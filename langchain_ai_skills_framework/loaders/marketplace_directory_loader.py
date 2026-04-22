from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Sequence, TYPE_CHECKING
from uuid import UUID, uuid4

from langchain_core.tools import BaseTool
from skillkit import SkillManager, SkillMetadata, Skill

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import MyScriptExecutor
from langchain_ai_skills_framework.executors.my_shell_executor import MyShellExecutor
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.github_directory_downloader import (
    GithubDirectoryDownloader,
)
from langchain_ai_skills_framework.loaders.marketplace_plugin_manager import (
    MarketplacePluginManager,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.snapshot_cache_mixin import (
    SnapshotCacheMixin,
)
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)
from langchain_ai_skills_framework.utilities.snapshot_serializer import (
    serialize_plugin_definition,
)

if TYPE_CHECKING:
    from key_value.aio.stores.base import BaseStore

logger = logging.getLogger(__name__)


class MarketplaceDirectoryLoader(SnapshotCacheMixin, SkillLoaderProtocol):
    """Loads Agent Skills from a Claude plugin marketplace GitHub repository.

    The marketplace structure is:
        plugins/<plugin-name>/skills/<skill-name>/SKILL.md
        plugins/<plugin-name>/scripts/<script-name>.py

    Each plugin's ``skills/`` subdirectory is scanned independently using
    skillkit's SkillManager and results are merged into a single snapshot.
    """

    def __init__(
        self,
        *,
        environment_variables: SkillLoaderEnvironmentVariables,
        github_directory_downloader: GithubDirectoryDownloader,
        plugin_manager: MarketplacePluginManager | None = None,
        snapshot_cache_store: BaseStore | None = None,
    ) -> None:
        self._identifier: UUID = uuid4()
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        self._environment_variables = environment_variables

        marketplace_uri = environment_variables.plugins_marketplace
        if not marketplace_uri or not marketplace_uri.strip():
            raise SkillValidationError("plugins_marketplace is not configured")
        self._marketplace_uri = marketplace_uri.strip()

        self._plugin_manager = plugin_manager or MarketplacePluginManager()

        if github_directory_downloader is None:
            raise SkillValidationError("github_directory_downloader is not configured")
        self._github_directory_downloader = github_directory_downloader

        self._snapshot_cache_store = snapshot_cache_store
        self._snapshot_cache_collection = environment_variables.snapshot_cache_plugins_collection
        self._plugins_collection = environment_variables.plugins_collection

        self._lock = RLock()
        self._snapshot: SkillSnapshot | None = None
        self._snapshot_loaded_at: float | None = None
        self._plugin_definitions: tuple[PluginDefinition, ...] = ()
        self._reload_ttl_seconds = self._resolve_reload_ttl_seconds(environment_variables)

        logger.info(
            "MarketplaceDirectoryLoader %s initialized for %s",
            self._identifier,
            self._marketplace_uri,
        )

    @property
    def _loader_display_name(self) -> str:
        return f"MarketplaceDirectoryLoader {self._identifier}"

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        snapshot = self._get_snapshot()
        return snapshot.ordered_summaries

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        snapshot = await self._get_snapshot_async()
        return snapshot.ordered_summaries

    def get_skill_details(self, skill_name: str, *, plugin_name: str = "") -> SkillDetails:
        normalized = normalize_skill_name(skill_name)
        snapshot = self._get_snapshot()
        try:
            return snapshot.details_by_name[normalized]
        except KeyError as exc:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in marketplace") from exc

    async def get_skill_details_for_user(self, *, user_id: str, plugin_name: str, skill_name: str) -> SkillDetails:
        normalized = normalize_skill_name(skill_name)
        snapshot = await self._get_snapshot_async()
        try:
            return snapshot.details_by_name[normalized]
        except KeyError as exc:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in marketplace") from exc

    def refresh(self) -> None:
        with self._lock:
            logger.info("MarketplaceDirectoryLoader refreshing cache")
            self._snapshot = self._build_snapshot(force_download=True)
            self._snapshot_loaded_at = time.monotonic()

    async def refresh_async(self) -> None:
        """Force reload and persist the new snapshot to MongoDB cache."""
        with self._lock:
            logger.info("MarketplaceDirectoryLoader refreshing cache (async)")
            self._snapshot = self._build_snapshot(force_download=True)
            self._snapshot_loaded_at = time.monotonic()
            snapshot = self._snapshot
            plugin_defs = self._plugin_definitions
        await self._write_to_snapshot_cache(snapshot)
        await self._write_plugins_to_collection(plugin_defs)

    async def get_instructions(self) -> str:
        # Marketplace plugins don't contribute system-prompt instructions,
        # but we trigger the async snapshot path here so the snapshot is
        # written to L2 (MongoDB) for cross-worker sharing.
        await self._get_snapshot_async()
        return ""

    def get_tools(self) -> list[BaseTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str, *, plugin_name: str = "") -> str:
        details = self.get_skill_details(skill_name)
        if details.source_path is None:
            raise SkillNotFoundError(f"Skill '{skill_name}' has no source path")
        references_dir = details.source_path.parent / "references"
        candidate_path = references_dir / resource_name
        try:
            resolved_references = references_dir.resolve()
            resolved_resource = candidate_path.resolve()
        except OSError as exc:
            raise SkillValidationError(
                f"Error resolving resource '{resource_name}' for skill '{skill_name}': {exc}"
            ) from exc

        try:
            resolved_resource.relative_to(resolved_references)
        except ValueError as exc:
            raise SkillValidationError(f"Invalid resource path '{resource_name}' for skill '{skill_name}'") from exc

        if not resolved_resource.is_file():
            raise SkillNotFoundError(f"Resource '{resource_name}' not found for skill '{skill_name}'")
        try:
            return resolved_resource.read_text(encoding="utf-8")
        except Exception as exc:
            raise SkillValidationError(
                f"Error reading resource '{resource_name}' for skill '{skill_name}': {exc}"
            ) from exc

    async def read_skill_resource_for_user(
        self, *, user_id: str, plugin_name: str, skill_name: str, resource_name: str
    ) -> str:
        return self.read_skill_resource(skill_name, resource_name, plugin_name=plugin_name)

    def list_skill_resource_names(self, skill_name: str, *, plugin_name: str = "") -> Sequence[str]:
        try:
            details = self.get_skill_details(skill_name)
        except SkillNotFoundError:
            return []
        if details.source_path is None:
            return []
        references_dir = details.source_path.parent / "references"
        if not references_dir.is_dir():
            return []
        return sorted(f.name for f in references_dir.iterdir() if f.is_file())

    async def list_skill_resource_names_for_user(
        self, *, user_id: str, plugin_name: str, skill_name: str
    ) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name, plugin_name=plugin_name)

    def list_skill_script_names(self, skill_name: str, *, plugin_name: str = "") -> Sequence[str]:
        try:
            details = self.get_skill_details(skill_name)
        except SkillNotFoundError:
            return []
        # Check for scripts in the skill's own directory
        if details.source_path is None:
            return []
        scripts_dir = details.source_path.parent / "scripts"
        if not scripts_dir.is_dir():
            # Also check the plugin-level scripts directory
            plugin_dir = self._find_plugin_dir_for_skill(details)
            if plugin_dir:
                scripts_dir = plugin_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        return sorted(f.stem for f in scripts_dir.iterdir() if f.is_file() and f.suffix in (".py", ".sh"))

    async def list_skill_script_names_for_user(
        self, *, user_id: str, plugin_name: str, skill_name: str
    ) -> Sequence[str]:
        return self.list_skill_script_names(skill_name, plugin_name=plugin_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None, *, plugin_name: str = ""
    ) -> MyScriptExecutionResult:
        details = self.get_skill_details(skill_name)
        script_path = self._resolve_script_path(details, script_name)
        if script_path is None:
            raise SkillNotFoundError(f"Script '{script_name}' not found for skill '{skill_name}'")

        if script_path.suffix == ".py":
            executor = MyScriptExecutor()
            normalized_arguments = {k.lower(): v for k, v in (arguments or {}).items()}
            script_content = script_path.read_text(encoding="utf-8")
            return await executor.execute_inline_script(
                script_name=script_name,
                script=script_content,
                arguments=normalized_arguments,
            )
        else:
            shell_executor = MyShellExecutor()
            return await shell_executor.execute(
                script_path=script_path,
                skill_base_dir=script_path.parent,
                arguments=arguments,
            )

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name, script_name, arguments, plugin_name=plugin_name)

    _SNAPSHOT_CACHE_KEY = "marketplace_snapshot"

    # --- Private implementation ------------------------------------------------

    def _get_snapshot(self) -> SkillSnapshot:
        with self._lock:
            if self._is_snapshot_valid_unlocked():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

        with self._lock:
            if self._is_snapshot_valid_unlocked():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

            logger.info(
                "MarketplaceDirectoryLoader cache expired; loading from %s",
                self._marketplace_uri,
            )
            self._snapshot = self._build_snapshot(force_download=False)
            self._snapshot_loaded_at = time.monotonic()
            return self._snapshot

    async def _get_snapshot_async(self) -> SkillSnapshot:
        """Async variant that checks MongoDB snapshot cache before building."""
        with self._lock:
            if self._is_snapshot_valid_unlocked():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

        # Check MongoDB snapshot cache
        snapshot = await self._read_from_snapshot_cache()
        if snapshot is not None:
            with self._lock:
                self._snapshot = snapshot
                self._snapshot_loaded_at = time.monotonic()
            return snapshot

        with self._lock:
            if self._is_snapshot_valid_unlocked():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

            logger.info(
                "MarketplaceDirectoryLoader cache expired; loading from %s",
                self._marketplace_uri,
            )
            self._snapshot = self._build_snapshot(force_download=False)
            self._snapshot_loaded_at = time.monotonic()
            await self._write_to_snapshot_cache(self._snapshot)
            await self._write_plugins_to_collection(self._plugin_definitions)
            return self._snapshot

    def get_plugin_mcp_configs(self) -> Sequence[PluginMcpServerEntry]:
        snapshot = self._get_snapshot()
        return snapshot.mcp_servers

    def _build_snapshot(self, *, force_download: bool) -> SkillSnapshot:
        cache_path = self._resolve_marketplace_path(force=force_download)

        details_map: dict[str, SkillDetails] = {}
        summaries: list[SkillSummary] = []
        all_mcp_servers: list[PluginMcpServerEntry] = []
        plugin_defs: list[PluginDefinition] = []

        excluded_skills = self._normalize_set(self._environment_variables.excluded_skills)
        excluded_groups = self._normalize_set(self._environment_variables.excluded_skill_groups)
        marketplace_include = self._normalize_set(self._environment_variables.plugins_marketplace_include) or None
        marketplace_exclude = self._normalize_set(self._environment_variables.plugins_marketplace_exclude)

        # Build combined exclude filter (marketplace-specific + global skill groups)
        combined_exclude = (marketplace_exclude | excluded_groups) if excluded_groups else marketplace_exclude

        # Discover plugins via MarketplacePluginManager
        plugin_entries = self._plugin_manager.discover_plugins(
            cache_path,
            include_filter=marketplace_include,
            exclude_filter=combined_exclude or None,
        )

        for entry in plugin_entries:
            # Collect MCP configs per plugin (avoids a second pass)
            plugin_mcp = self._plugin_manager.read_mcp_configs(entry)
            all_mcp_servers.extend(plugin_mcp)

            plugin_skills: list[SkillSummary] = []

            skills_dir = entry.path / "skills"
            if not skills_dir.is_dir():
                plugin_defs.append(
                    PluginDefinition(
                        name=entry.name,
                        description=entry.description,
                        mcp_servers=tuple(plugin_mcp),
                    )
                )
                continue

            try:
                manager = SkillManager(
                    project_skill_dir=skills_dir,
                    anthropic_config_dir="",
                    plugin_dirs=[],
                    additional_search_paths=[],
                )
                manager.discover()
            except Exception:
                logger.exception(
                    "Marketplace: failed to discover skills in plugin '%s'",
                    entry.name,
                )
                plugin_defs.append(
                    PluginDefinition(
                        name=entry.name,
                        description=entry.description,
                        mcp_servers=tuple(plugin_mcp),
                    )
                )
                continue

            metadata: SkillMetadata | str
            for metadata in manager.list_skills():
                if not isinstance(metadata, SkillMetadata):
                    continue
                try:
                    skill: Skill = manager.load_skill(name=metadata.name)
                    definition = self._map_skill(metadata=metadata, content=skill.content, plugin_name=entry.name)
                except Exception:
                    logger.exception(
                        "Marketplace: failed to load skill '%s' from plugin '%s'",
                        metadata.name if isinstance(metadata, SkillMetadata) else metadata,
                        entry.name,
                    )
                    continue

                if definition.name in excluded_skills:
                    logger.info("Marketplace: skipping excluded skill '%s'", definition.name)
                    continue
                if definition.name in details_map:
                    logger.warning(
                        "Marketplace: duplicate skill '%s' from plugin '%s'; keeping first occurrence",
                        definition.name,
                        entry.name,
                    )
                    continue

                details_map[definition.name] = definition
                summaries.append(definition.summary)
                plugin_skills.append(definition.summary)

            plugin_defs.append(
                PluginDefinition(
                    name=entry.name,
                    description=entry.description,
                    skills=tuple(sorted(plugin_skills, key=lambda s: s.name)),
                    mcp_servers=tuple(plugin_mcp),
                )
            )

        self._plugin_definitions = tuple(plugin_defs)

        ordered = tuple(sorted(summaries, key=lambda s: s.name))
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType(details_map),
            ordered_summaries=ordered,
            mcp_servers=tuple(all_mcp_servers),
        )
        logger.info(
            "Marketplace: loaded %d skills and %d MCP servers from %d plugins",
            len(ordered),
            len(all_mcp_servers),
            len(plugin_entries),
        )
        return snapshot

    async def _write_plugins_to_collection(
        self,
        plugin_definitions: tuple[PluginDefinition, ...],
    ) -> None:
        """Write each plugin as an individual document to the plugins collection."""
        if not self._snapshot_cache_store or not self._plugins_collection:
            return
        for plugin in plugin_definitions:
            try:
                data = serialize_plugin_definition(plugin)
                await self._snapshot_cache_store.put(
                    plugin.name,
                    data,
                    ttl=self._reload_ttl_seconds,
                    collection=self._plugins_collection,
                )
            except Exception:
                logger.debug(
                    "MarketplaceDirectoryLoader: failed to write plugin '%s' to collection",
                    plugin.name,
                    exc_info=True,
                )

    def _resolve_marketplace_path(self, *, force: bool) -> Path:
        """Resolve the marketplace to a local path.

        Supports both local filesystem paths and github:// URIs.
        Local paths are used directly; github:// URIs are downloaded via fsspec.
        """
        if not self._marketplace_uri.startswith("github://"):
            # Local filesystem path
            local_path = Path(self._marketplace_uri).expanduser().resolve()
            if not local_path.is_dir():
                raise SkillValidationError(f"Marketplace directory does not exist: '{self._marketplace_uri}'")
            return local_path

        # GitHub URI — download via the directory downloader with TTL-awareness.
        # When force=False, delegate freshness checks to the downloader's
        # cache_ttl_seconds parameter so expired caches trigger re-downloads.
        cache_path = Path(".marketplace-git-cache")
        cache_ttl = 0 if force else int(self._reload_ttl_seconds or 0)

        try:
            return self._github_directory_downloader.download(
                source_uri=self._marketplace_uri,
                github_token=self._environment_variables.skills_github_token,
                cache_path=cache_path,
                cache_ttl_seconds=cache_ttl,
            )
        except ValueError as exc:
            raise SkillValidationError(f"Failed to download marketplace from '{self._marketplace_uri}': {exc}") from exc

    def _find_plugin_dir_for_skill(self, details: SkillDetails) -> Path | None:
        """Walk up from the skill's source_path to find the plugin root."""
        if details.source_path is None:
            return None
        current = details.source_path.parent
        for _ in range(5):
            if (current / ".claude-plugin").is_dir() or (current / "scripts").is_dir():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None

    def _resolve_script_path(self, details: SkillDetails, script_name: str) -> Path | None:
        """Find a script file for the given skill."""
        if details.source_path is None:
            return None
        cleaned = script_name.removesuffix(".py").removesuffix(".sh")

        # Check skill-level scripts directory
        skill_scripts = details.source_path.parent / "scripts"
        for ext in (".py", ".sh"):
            candidate = skill_scripts / (cleaned + ext)
            if candidate.is_file():
                return candidate

        # Check plugin-level scripts directory
        plugin_dir = self._find_plugin_dir_for_skill(details)
        if plugin_dir:
            plugin_scripts = plugin_dir / "scripts"
            for ext in (".py", ".sh"):
                candidate = plugin_scripts / (cleaned + ext)
                if candidate.is_file():
                    return candidate

        return None

    @staticmethod
    def _map_skill(metadata: SkillMetadata, content: str, plugin_name: str = "") -> SkillDetails:
        normalized_name = normalize_skill_name(metadata.name)
        if not normalized_name:
            raise SkillValidationError("Skill name must not be empty")

        description = metadata.description.strip() if isinstance(metadata.description, str) else ""
        if not description:
            raise SkillValidationError(f"Skill {normalized_name} must include a non-empty description")

        summary = SkillSummary(
            name=normalized_name,
            description=description,
            plugin_name=plugin_name,
            source_path=metadata.skill_path,
            license=None,
            compatibility=None,
            metadata={"source": "marketplace", "plugin_name": plugin_name},
            allowed_tools=metadata.allowed_tools,
        )
        return SkillDetails(
            summary=summary,
            content=content,
            source_path=metadata.skill_path,
        )

    @staticmethod
    def _normalize_set(values: set[str] | None) -> frozenset[str]:
        if not values:
            return frozenset()
        return frozenset(normalize_skill_name(v) for v in values if isinstance(v, str) and v.strip())
