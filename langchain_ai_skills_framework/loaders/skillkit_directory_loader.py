import logging
import re
import time
from html import escape
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Sequence, Any
from uuid import UUID, uuid4

from langchain_core.tools import StructuredTool
from skillkit import SkillManager, SkillMetadata, ScriptExecutionResult

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
    SkillSnapshot,
)
from langchain_ai_skills_framework.tools.read_skill_resource_tool import (
    ReadSkillResourceTool,
)
from langchain_ai_skills_framework.tools.run_skill_script_tool import RunSkillScriptTool
from langchain_ai_skills_framework.tools.load_skill_tool import LoadSkillTool
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["CONFIG"])


# Default instruction template for skills system prompt
_INSTRUCTION_SKILLS_HEADER = """\
You have access to a collection of skills containing domain-specific knowledge and capabilities.
Each skill provides specialized instructions, resources, and scripts for specific tasks.

<available_skills>
{skills_list}
</available_skills>

When a task falls within a skill's domain:
1. Use `load_skill` to read the complete skill instructions
2. Follow the skill's guidance to complete the task
3. Use any additional skill resources and scripts as needed

Use progressive disclosure: load only what you need, when you need it."""

# Template used by load_skill
LOAD_SKILL_TEMPLATE = """<skill>
<name>{skill_name}</name>
<description>{description}</description>
<uri>{uri}</uri>

<resources>
{resources_list}
</resources>

<scripts>
{scripts_list}
</scripts>

<instructions>
{content}
</instructions>
</skill>
"""


class SkillkitDirectoryLoader(SkillLoaderProtocol):
    """Loads Agent Skills from local directories using skillkit."""

    # Public API

    def __init__(
        self,
        *,
        environment_variables: SkillLoaderEnvironmentVariables,
        github_skill_downloader: GithubSkillDownloader,
    ) -> None:
        self._identifier: UUID = uuid4()
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        if not isinstance(environment_variables, SkillLoaderEnvironmentVariables):
            raise TypeError(
                "environment_variables must be an instance of SkillLoaderEnvironmentVariables: "
                f"{type(environment_variables)}"
            )
        self._environment_variables = environment_variables

        skills_directory = environment_variables.skills_directory
        if isinstance(skills_directory, Path):
            configured_directory = str(skills_directory)
        elif isinstance(skills_directory, str):
            configured_directory = skills_directory
        else:
            raise SkillValidationError(
                f"skills_directory must be a string or Path: {type(skills_directory)}"
            )

        if not configured_directory.strip():
            raise SkillValidationError("skills_directory is not configured")

        self._skills_directory = self._normalize_skills_directory(configured_directory)
        self._skills_root_path = self._initial_skills_root_path(self._skills_directory)
        self._github_skill_downloader = github_skill_downloader
        if github_skill_downloader is None:
            raise SkillValidationError("github_skill_downloader is not configured")
        if not isinstance(github_skill_downloader, GithubSkillDownloader):
            raise TypeError(
                "github_skill_downloader must be an instance of GithubSkillDownloader:"
                f" {type(github_skill_downloader)} "
            )
        self._lock = RLock()
        self._snapshot: SkillSnapshot | None = None
        self._snapshot_loaded_at: float | None = None
        self._manager: SkillManager = self._create_manager()
        self._reload_ttl_seconds = self._resolve_reload_ttl_seconds(
            environment_variables
        )

        logger.info(
            "SkillkitDirectoryLoader %s initialized for %s",
            self._identifier,
            self._skills_directory,
        )

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        """Return lightweight skill summaries from the current snapshot."""

        snapshot = self._get_snapshot()
        logger.debug(
            "SkillkitDirectoryLoader %s returning %d summaries",
            self._identifier,
            len(snapshot.ordered_summaries),
        )
        return snapshot.ordered_summaries

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        """Return full skill details for the normalized skill name."""

        normalized = self._normalize_skill_name(skill_name)
        snapshot = self._get_snapshot()
        try:
            return snapshot.details_by_name[normalized]
        except KeyError as exc:
            logger.warning(
                "SkillkitDirectoryLoader %s could not find skill '%s'",
                self._identifier,
                skill_name,
            )
            raise SkillNotFoundError(f"Skill '{skill_name}' not found") from exc

    def refresh(self) -> None:
        """Force an immediate reload regardless of TTL."""

        with self._lock:
            logger.info("SkillkitDirectoryLoader %s refreshing cache", self._identifier)
            self._reload_manager(force=True)
            self._snapshot = self._build_snapshot()
            self._snapshot_loaded_at = time.monotonic()

    def _build_skills_prompt(self, *, summaries: Sequence[SkillSummary]) -> str:
        if not summaries:
            return "No skills are currently configured."
        skills_block = " ".join(
            self._format_skill_entry(summary) for summary in summaries
        )
        return f"<available_skills> {skills_block} </available_skills>"

    @staticmethod
    def _format_skill_entry(summary: SkillSummary) -> str:
        escaped_name = escape(summary.name, quote=True)
        escaped_description = escape(summary.description.strip(), quote=True)
        return (
            "<skill>"
            f"<name> {escaped_name} </name> "
            f"<description> {escaped_description} </description> "
            "</skill>"
        )

    async def get_instructions(self) -> str:
        """Return `<available_skills>` block used by middleware system prompts."""

        # Build skills list in XML format
        skills_list_lines: list[str] = []
        skill: SkillSummary
        for skill in self.list_skill_summaries(allowed_skills=set()):
            skills_list_lines.append("<skill>")
            skills_list_lines.append(f"<name>{skill.name}</name>")
            skills_list_lines.append(f"<description>{skill.description}</description>")
            skills_list_lines.append("</skill>")
        skills_list = "\n".join(skills_list_lines)

        # Use custom template if provided, otherwise use default
        return _INSTRUCTION_SKILLS_HEADER.format(skills_list=skills_list)

    def get_tools(self) -> list[StructuredTool]:
        return [
            LoadSkillTool(
                skill_loader=self,
            ),
            ReadSkillResourceTool(
                skill_loader=self,
            ),
            RunSkillScriptTool(
                skill_loader=self,
            ),
        ]

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        """Read a specific resource from a skill, such as a file or script."""
        details = self.get_skill_details(skill_name=skill_name)
        resource_path = details.source_path.parent.joinpath(resource_name)
        if not resource_path.is_file():
            raise SkillNotFoundError(
                f"Resource '{resource_name}' not found for skill '{skill_name}'"
            )
        try:
            return resource_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise SkillValidationError(
                f"Error reading resource '{resource_name}' for skill '{skill_name}': {exc}"
            ) from exc

    def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> str:
        """Run a specific script from a skill and return its output."""
        result: ScriptExecutionResult = self._manager.execute_skill_script(
            skill_name=skill_name,
            script_name=script_name,
            arguments=arguments or {},
        )
        if result.exit_code != 0:
            return result.stdout

        return result.stdout

    # Snapshot lifecycle

    def _get_snapshot(self) -> SkillSnapshot:
        # Fast path: use the in-memory snapshot while TTL is still valid.
        with self._lock:
            if self._is_snapshot_valid_unlocked():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

        with self._lock:
            # Double-check after acquiring the lock in case another thread refreshed.
            if self._is_snapshot_valid_unlocked():
                snapshot = self._snapshot
                if snapshot is not None:
                    return snapshot

            logger.info(
                "SkillkitDirectoryLoader %s cache expired or empty; loading skills",
                self._identifier,
            )
            self._reload_manager(force=False)
            self._snapshot = self._build_snapshot()
            self._snapshot_loaded_at = time.monotonic()
            return self._snapshot

    def _build_snapshot(self) -> SkillSnapshot:
        """Build a normalized, exclusion-aware snapshot from the loaded manager."""

        logger.info(
            "SkillkitDirectoryLoader %s loading skills from %s",
            self._identifier,
            self._skills_directory,
        )
        new_details: dict[str, SkillDetails] = {}
        new_summaries: list[SkillSummary] = []

        excluded_skills = self._normalize_excluded_skills(
            self._environment_variables.excluded_skills
        )
        excluded_skill_groups = self._normalize_excluded_skill_groups(
            self._environment_variables.excluded_skill_groups
        )

        metadata: SkillMetadata | str
        for metadata in self._manager.list_skills():
            if isinstance(metadata, SkillMetadata):
                # skill: Skill = self._manager.load_skill(name=metadata.name)
                definition = self._map_skill(metadata=metadata, content="")
                skill_group = self._resolve_skill_group(
                    str(definition.source_path.parent)
                )
                if skill_group and skill_group in excluded_skill_groups:
                    logger.info("Skipping excluded skill group '%s'", skill_group)
                    continue
                if definition.name in excluded_skills:
                    logger.info("Skipping excluded skill '%s'", definition.name)
                    continue
                if definition.name in new_details:
                    raise SkillValidationError(
                        f"Duplicate skill name '{definition.name}' detected"
                    )
                new_details[definition.name] = definition
                new_summaries.append(definition.summary)

        ordered_summaries = tuple(
            sorted(new_summaries, key=lambda summary: summary.name)
        )
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType(new_details),
            ordered_summaries=ordered_summaries,
        )
        logger.info(
            "Loaded %d Agent Skills from %s",
            len(ordered_summaries),
            self._skills_directory,
        )
        return snapshot

    # Manager loading and reloading

    def _reload_manager(self, *, force: bool) -> None:
        """Refresh the underlying SkillManager when forced or when TTL expires."""

        try:
            if (
                force
                or self._snapshot is None
                or not self._is_snapshot_valid_unlocked()
            ):
                if self._skills_directory.startswith("github://"):
                    self._skills_root_path = self._github_skill_downloader.download(
                        cache_path=Path(".skills-git-cache"),
                        skills_directory=self._skills_directory,
                        github_token=self._environment_variables.skills_github_token,
                    )
                self._manager.discover()
        except Exception as exc:
            raise SkillValidationError(str(exc)) from exc

    def _create_manager(self) -> SkillManager:
        """Create a skillkit SkillManager from local or cached github:// skills."""

        if self._skills_directory.startswith("github://"):
            self._skills_root_path = self._github_skill_downloader.download(
                cache_path=Path(".skills-git-cache"),
                skills_directory=self._skills_directory,
                github_token=self._environment_variables.skills_github_token,
            )

        manager = SkillManager(
            project_skill_dir=self._skills_root_path,
            anthropic_config_dir="",
            plugin_dirs=[],
            additional_search_paths=[],
        )
        return manager

    # TTL helpers

    def _is_snapshot_valid_unlocked(self) -> bool:
        """Lock-held validity check based on snapshot presence and TTL age."""

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
        """Resolve loader TTL from environment, defaulting to one hour."""

        configured = getattr(
            environment_variables, "skills_cache_timeout_seconds", 3600
        )
        if isinstance(configured, bool):
            return 3600.0
        if not isinstance(configured, (int, float)):
            return 3600.0
        configured_seconds = float(configured)
        if configured_seconds <= 0:
            return None
        return configured_seconds

    # Skill mapping and validation

    def _map_skill(self, metadata: SkillMetadata, content: str) -> SkillDetails:
        """Map a skillkit skill into framework SkillDetails."""
        normalized_name = self._normalize_skill_name(metadata.name)
        if not normalized_name:
            raise SkillValidationError("Skill name must not be empty")

        description = (
            metadata.description.strip()
            if isinstance(metadata.description, str)
            else ""
        )
        if not description:
            raise SkillValidationError(
                f"Skill {normalized_name} must include a non-empty description"
            )

        summary = SkillSummary(
            name=normalized_name,
            description=description,
            source_path=metadata.skill_path,
            license=None,
            compatibility=None,
            metadata={},
            allowed_tools=metadata.allowed_tools,
        )
        return SkillDetails(
            summary=summary, content=content, source_path=metadata.skill_path
        )

    # Source path and name normalization

    def _resolve_skill_group(self, skill_dir: str) -> str | None:
        try:
            relative = PurePosixPath(skill_dir).relative_to(
                PurePosixPath(str(self._skills_root_path))
            )
        except ValueError:
            return None
        if len(relative.parts) < 2:
            return None
        return self._normalize_skill_name(relative.parts[0])

    @staticmethod
    def _normalize_skills_directory(value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("github://"):
            return normalized
        return str(Path(normalized).expanduser())

    @staticmethod
    def _initial_skills_root_path(skills_directory: str) -> Path:
        if skills_directory.startswith("github://"):
            return Path(skills_directory)
        return Path(skills_directory).expanduser().resolve()

    @classmethod
    def _normalize_excluded_skill_groups(
        cls, values: set[str] | None
    ) -> frozenset[str]:
        if not values:
            return frozenset()
        normalized = {
            cls._normalize_skill_name(value)
            for value in values
            if isinstance(value, str) and value.strip()
        }
        return frozenset(item for item in normalized if item)

    @classmethod
    def _normalize_excluded_skills(cls, values: set[str] | None) -> frozenset[str]:
        if not values:
            return frozenset()
        normalized = {
            cls._normalize_skill_name(value)
            for value in values
            if isinstance(value, str) and value.strip()
        }
        return frozenset(item for item in normalized if item)

    @staticmethod
    def _normalize_skill_name(value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        normalized = re.sub(r"-+", "-", normalized)
        return normalized.strip("-")
