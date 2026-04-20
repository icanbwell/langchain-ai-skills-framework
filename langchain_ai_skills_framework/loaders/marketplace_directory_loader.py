import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Sequence

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
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PluginEntry:
    """A discovered plugin with its resolved filesystem path and canonical name."""

    name: str
    path: Path
    description: str | None = None


class MarketplaceDirectoryLoader(SkillLoaderProtocol):
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
    ) -> None:
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        self._environment_variables = environment_variables

        marketplace_uri = environment_variables.plugins_marketplace
        if not marketplace_uri or not marketplace_uri.strip():
            raise SkillValidationError("plugins_marketplace is not configured")
        self._marketplace_uri = marketplace_uri.strip()

        if github_directory_downloader is None:
            raise SkillValidationError("github_directory_downloader is not configured")
        self._github_directory_downloader = github_directory_downloader

        self._lock = RLock()
        self._snapshot: SkillSnapshot | None = None
        self._snapshot_loaded_at: float | None = None
        self._reload_ttl_seconds = self._resolve_reload_ttl_seconds(environment_variables)

        logger.info(
            "MarketplaceDirectoryLoader initialized for %s",
            self._marketplace_uri,
        )

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        snapshot = self._get_snapshot()
        return snapshot.ordered_summaries

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills)

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        normalized = normalize_skill_name(skill_name)
        snapshot = self._get_snapshot()
        try:
            return snapshot.details_by_name[normalized]
        except KeyError as exc:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in marketplace") from exc

    async def get_skill_details_for_user(self, *, user_id: str, skill_name: str) -> SkillDetails:
        return self.get_skill_details(skill_name)

    def refresh(self) -> None:
        with self._lock:
            logger.info("MarketplaceDirectoryLoader refreshing cache")
            self._snapshot = self._build_snapshot(force_download=True)
            self._snapshot_loaded_at = time.monotonic()

    async def get_instructions(self) -> str:
        return ""

    def get_tools(self) -> list[BaseTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        details = self.get_skill_details(skill_name)
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

    async def read_skill_resource_for_user(self, *, user_id: str, skill_name: str, resource_name: str) -> str:
        return self.read_skill_resource(skill_name, resource_name)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        try:
            details = self.get_skill_details(skill_name)
        except SkillNotFoundError:
            return []
        references_dir = details.source_path.parent / "references"
        if not references_dir.is_dir():
            return []
        return sorted(f.name for f in references_dir.iterdir() if f.is_file())

    async def list_skill_resource_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name)

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        try:
            details = self.get_skill_details(skill_name)
        except SkillNotFoundError:
            return []
        # Check for scripts in the skill's own directory
        scripts_dir = details.source_path.parent / "scripts"
        if not scripts_dir.is_dir():
            # Also check the plugin-level scripts directory
            plugin_dir = self._find_plugin_dir_for_skill(details)
            if plugin_dir:
                scripts_dir = plugin_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        return sorted(f.stem for f in scripts_dir.iterdir() if f.is_file() and f.suffix in (".py", ".sh"))

    async def list_skill_script_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_script_names(skill_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
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
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name, script_name, arguments)

    # --- Private implementation ------------------------------------------------

    def _get_snapshot(self) -> SkillSnapshot:
        with self._lock:
            if self._is_snapshot_valid():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

        with self._lock:
            if self._is_snapshot_valid():
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

    def _build_snapshot(self, *, force_download: bool) -> SkillSnapshot:
        cache_path = self._resolve_marketplace_path(force=force_download)

        details_map: dict[str, SkillDetails] = {}
        summaries: list[SkillSummary] = []

        excluded_skills = self._normalize_set(self._environment_variables.excluded_skills)
        excluded_groups = self._normalize_set(self._environment_variables.excluded_skill_groups)
        marketplace_include = self._normalize_set(self._environment_variables.plugins_marketplace_include) or None
        marketplace_exclude = self._normalize_set(self._environment_variables.plugins_marketplace_exclude)

        # Discover plugins via marketplace.json or directory scanning fallback
        plugin_entries = self._discover_plugins(cache_path)

        for entry in plugin_entries:
            plugin_name = entry.name
            normalized_plugin_name = normalize_skill_name(plugin_name)

            # Include-list takes precedence: if set, only named plugins are loaded
            if marketplace_include and normalized_plugin_name not in marketplace_include:
                logger.debug("Marketplace: skipping plugin '%s' (not in include list)", plugin_name)
                continue

            # Exclude-list (both marketplace-specific and the global skill groups)
            if normalized_plugin_name in marketplace_exclude:
                logger.info("Marketplace: skipping excluded plugin '%s'", plugin_name)
                continue
            if normalized_plugin_name in excluded_groups:
                logger.info("Marketplace: skipping excluded plugin group '%s'", plugin_name)
                continue

            skills_dir = entry.path / "skills"
            if not skills_dir.is_dir():
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
                    plugin_name,
                )
                continue

            metadata: SkillMetadata | str
            for metadata in manager.list_skills():
                if not isinstance(metadata, SkillMetadata):
                    continue
                try:
                    skill: Skill = manager.load_skill(name=metadata.name)
                    definition = self._map_skill(metadata=metadata, content=skill.content)
                except Exception:
                    logger.exception(
                        "Marketplace: failed to load skill '%s' from plugin '%s'",
                        metadata.name if isinstance(metadata, SkillMetadata) else metadata,
                        plugin_name,
                    )
                    continue

                if definition.name in excluded_skills:
                    logger.info("Marketplace: skipping excluded skill '%s'", definition.name)
                    continue
                if definition.name in details_map:
                    logger.warning(
                        "Marketplace: duplicate skill '%s' from plugin '%s'; keeping first occurrence",
                        definition.name,
                        plugin_name,
                    )
                    continue

                details_map[definition.name] = definition
                summaries.append(definition.summary)

        ordered = tuple(sorted(summaries, key=lambda s: s.name))
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType(details_map),
            ordered_summaries=ordered,
        )
        logger.info(
            "Marketplace: loaded %d skills from %d plugins",
            len(ordered),
            len(plugin_entries),
        )
        return snapshot

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

    @staticmethod
    def _discover_plugins(cache_path: Path) -> list[_PluginEntry]:
        """Discover plugins from the marketplace root.

        Reads ``.claude-plugin/marketplace.json`` when present (the canonical
        Claude Code plugin marketplace spec).  Falls back to directory scanning
        for backward compatibility with marketplaces that lack a manifest.

        marketplace.json schema (relevant fields)::

            {
              "name": "my-marketplace",
              "plugins": [
                {"name": "plugin-a", "source": "./plugins/plugin-a", "description": "..."}
              ],
              "metadata": {"pluginRoot": "optional/base/path"}
            }

        ``source`` paths starting with ``./`` are resolved relative to the
        marketplace root.  ``metadata.pluginRoot`` is prepended to relative
        source paths when present.
        """
        manifest_path = cache_path / ".claude-plugin" / "marketplace.json"
        if manifest_path.is_file():
            return MarketplaceDirectoryLoader._parse_marketplace_json(manifest_path, cache_path)

        # Fallback: directory-based discovery for legacy marketplaces
        return MarketplaceDirectoryLoader._discover_plugins_from_directories(cache_path)

    @staticmethod
    def _parse_marketplace_json(manifest_path: Path, marketplace_root: Path) -> list[_PluginEntry]:
        """Parse .claude-plugin/marketplace.json and resolve plugin paths."""
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read marketplace.json at %s: %s; falling back to directory scan",
                manifest_path,
                exc,
            )
            return MarketplaceDirectoryLoader._discover_plugins_from_directories(marketplace_root)

        if not isinstance(manifest, dict):
            logger.warning(
                "marketplace.json at %s is not a JSON object; falling back to directory scan",
                manifest_path,
            )
            return MarketplaceDirectoryLoader._discover_plugins_from_directories(marketplace_root)

        plugins_list = manifest.get("plugins", [])
        if not isinstance(plugins_list, list):
            logger.warning("marketplace.json 'plugins' is not an array; falling back to directory scan")
            return MarketplaceDirectoryLoader._discover_plugins_from_directories(marketplace_root)

        # metadata.pluginRoot is a base directory prepended to relative source paths
        metadata = manifest.get("metadata") or {}
        plugin_root_str = metadata.get("pluginRoot", "")

        entries: list[_PluginEntry] = []
        for plugin_spec in plugins_list:
            if not isinstance(plugin_spec, dict):
                continue

            name = plugin_spec.get("name")
            source = plugin_spec.get("source")
            if not name or not isinstance(name, str):
                logger.warning("Skipping marketplace plugin entry with missing/invalid 'name'")
                continue
            if not source or not isinstance(source, str):
                logger.warning("Skipping marketplace plugin '%s' with missing/invalid 'source'", name)
                continue

            resolved_path = MarketplaceDirectoryLoader._resolve_plugin_source(
                source=source,
                marketplace_root=marketplace_root,
                plugin_root=plugin_root_str,
            )
            if resolved_path is None:
                logger.warning(
                    "Skipping marketplace plugin '%s': unsupported source type '%s'",
                    name,
                    source,
                )
                continue

            if not resolved_path.is_dir():
                logger.warning(
                    "Skipping marketplace plugin '%s': resolved path does not exist: %s",
                    name,
                    resolved_path,
                )
                continue

            description = plugin_spec.get("description")
            entries.append(
                _PluginEntry(
                    name=name.strip(),
                    path=resolved_path,
                    description=description if isinstance(description, str) else None,
                )
            )

        logger.info(
            "marketplace.json: discovered %d plugins from %s",
            len(entries),
            manifest_path,
        )
        return sorted(entries, key=lambda e: e.name)

    @staticmethod
    def _resolve_plugin_source(*, source: str, marketplace_root: Path, plugin_root: str) -> Path | None:
        """Resolve a plugin source path from marketplace.json.

        Supports relative paths (starting with ``./``).  Other source types
        (git URLs, npm packages) are not yet supported and return None.
        """
        if source.startswith("./") or source.startswith("../"):
            # Relative path — resolve against marketplace root with optional pluginRoot
            base = marketplace_root
            if plugin_root:
                base = marketplace_root / plugin_root
            return (base / source).resolve()

        # Absolute local path (unusual but supported)
        if source.startswith("/"):
            return Path(source)

        # Bare relative path (no ./ prefix) — treat as relative to marketplace root
        base = marketplace_root
        if plugin_root:
            base = marketplace_root / plugin_root
        return (base / source).resolve()

    @staticmethod
    def _discover_plugins_from_directories(cache_path: Path) -> list[_PluginEntry]:
        """Legacy fallback: discover plugins by scanning directories.

        Supports two layouts:
        1. Direct: cache_path contains plugin subdirectories with skills/ folders
        2. Nested: cache_path/plugins/ contains plugin subdirectories
        """
        plugins_subdir = cache_path / "plugins"
        if plugins_subdir.is_dir():
            return sorted(
                (
                    _PluginEntry(name=d.name, path=d)
                    for d in plugins_subdir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ),
                key=lambda e: e.name,
            )

        return sorted(
            (
                _PluginEntry(name=d.name, path=d)
                for d in cache_path.iterdir()
                if d.is_dir() and not d.name.startswith(".") and (d / "skills").is_dir()
            ),
            key=lambda e: e.name,
        )

    def _find_plugin_dir_for_skill(self, details: SkillDetails) -> Path | None:
        """Walk up from the skill's source_path to find the plugin root."""
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

    def _is_snapshot_valid(self) -> bool:
        if self._snapshot is None:
            return False
        if self._reload_ttl_seconds is None:
            return True
        if self._snapshot_loaded_at is None:
            return False
        return (time.monotonic() - self._snapshot_loaded_at) < self._reload_ttl_seconds

    @staticmethod
    def _resolve_reload_ttl_seconds(
        environment_variables: SkillLoaderEnvironmentVariables,
    ) -> float | None:
        configured = environment_variables.skills_cache_timeout_seconds
        if isinstance(configured, bool):
            return 3600.0
        if not isinstance(configured, (int, float)):
            return 3600.0
        configured_seconds = float(configured)
        if configured_seconds <= 0:
            return None
        return configured_seconds

    @staticmethod
    def _map_skill(metadata: SkillMetadata, content: str) -> SkillDetails:
        normalized_name = normalize_skill_name(metadata.name)
        if not normalized_name:
            raise SkillValidationError("Skill name must not be empty")

        description = metadata.description.strip() if isinstance(metadata.description, str) else ""
        if not description:
            raise SkillValidationError(f"Skill {normalized_name} must include a non-empty description")

        summary = SkillSummary(
            name=normalized_name,
            description=description,
            source_path=metadata.skill_path,
            license=None,
            compatibility=None,
            metadata={"source": "marketplace"},
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
