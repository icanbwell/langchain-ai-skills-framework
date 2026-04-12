from __future__ import annotations

import logging
import re
import time
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Mapping, Sequence, cast, Any
from uuid import UUID, uuid4

from langchain_core.tools import BaseTool
from pydantic_ai_skills import SkillsToolset
from pydantic_ai_skills.exceptions import (
    SkillRegistryError as PydanticSkillRegistryError,
    SkillValidationError as PydanticSkillValidationError,
)
from pydantic_ai_skills.types import Skill

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GitLocation,
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
from langchain_ai_skills_framework.tools.load_skill_tool import LoadSkillTool
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["CONFIG"])


class SkillDirectoryLoader(SkillLoaderProtocol):
    """Loads Agent Skills from local directories or GitHub repositories."""

    _github_uri_example = "github://my-org/private-skills/skills?ref=main"

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
                f"environment_variables must be an instance of SkillLoaderEnvironmentVariables: {type(environment_variables)}"
            )
        self._environment_variables = environment_variables

        self._github_skill_downloader = github_skill_downloader
        if github_skill_downloader is None:
            raise SkillValidationError("github_skill_downloader is not configured")
        if not isinstance(github_skill_downloader, GithubSkillDownloader):
            raise TypeError(
                "github_skill_downloader must be an instance of GithubSkillDownloader:"
                f" {type(github_skill_downloader)} "
            )
        skills_directory = environment_variables.skills_directory
        if isinstance(skills_directory, Path):
            configured_directory = str(skills_directory)
        elif isinstance(skills_directory, str):
            configured_directory = skills_directory
        else:
            raise SkillValidationError("skills_directory must be a string or Path")

        if not configured_directory.strip():
            raise SkillValidationError("skills_directory is not configured")

        self._skills_directory = self._normalize_skills_directory(configured_directory)
        self._skills_root_path = self._initial_skills_root_path(self._skills_directory)
        self._lock = RLock()
        self._snapshot: SkillSnapshot | None = None
        self._snapshot_loaded_at: float | None = None
        self._skills_toolset: SkillsToolset = self._create_toolset()
        self._reload_ttl_seconds = self._resolve_reload_ttl_seconds(
            environment_variables
        )

        logger.info(
            "SkillDirectoryLoader %s initialized for %s",
            self._identifier,
            self._skills_directory,
        )

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        """Return lightweight skill summaries from the current snapshot."""

        snapshot = self._get_snapshot()
        logger.debug(
            "SkillDirectoryLoader %s returning %d summaries",
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
                "SkillDirectoryLoader %s could not find skill '%s'",
                self._identifier,
                skill_name,
            )
            raise SkillNotFoundError(f"Skill '{skill_name}' not found") from exc

    async def list_all_summaries(
        self, *, user_id: str, allowed_skills: set[str]
    ) -> Sequence[SkillSummary]:
        """Fallback: directory loader has no user skills, delegate to shared."""
        return self.list_skill_summaries(allowed_skills)

    async def get_skill_details_for_user(
        self, *, user_id: str, skill_name: str
    ) -> SkillDetails:
        """Fallback: directory loader has no user skills, delegate to shared."""
        return self.get_skill_details(skill_name)

    def refresh(self) -> None:
        """Force an immediate reload regardless of TTL."""

        with self._lock:
            logger.info("SkillDirectoryLoader %s refreshing cache", self._identifier)
            self._reload_toolset(force=True)
            self._snapshot = self._build_snapshot()
            self._snapshot_loaded_at = time.monotonic()

    async def get_instructions(self) -> str:
        return await self._skills_toolset.get_instructions(ctx=None)  # type: ignore[arg-type, return-value]

    def get_tools(self) -> list[BaseTool]:
        return [
            LoadSkillTool(
                skill_loader=self,
            ),
        ]

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        raise NotImplementedError()

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError()

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return []

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return []

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
                "SkillDirectoryLoader %s cache expired or empty; loading skills",
                self._identifier,
            )
            self._reload_toolset(force=False)
            self._snapshot = self._build_snapshot()
            self._snapshot_loaded_at = time.monotonic()
            return self._snapshot

    def _build_snapshot(self) -> SkillSnapshot:
        """Build a normalized, exclusion-aware snapshot from the loaded toolset."""

        logger.info(
            "SkillDirectoryLoader %s loading skills from %s",
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

        if self._skills_toolset is None:
            raise SkillValidationError("Skills toolset is not initialized")

        skills_by_name = cast(Mapping[str, Skill], self._skills_toolset.skills)
        skills = tuple(skills_by_name.values())
        for skill in skills:
            definition = self._map_skill(skill)
            skill_group = self._resolve_skill_group(str(definition.source_path.parent))
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

    # Toolset loading and reloading

    def _reload_toolset(self, *, force: bool) -> None:
        """Refresh the underlying SkillsToolset when forced or when TTL expires."""

        try:
            if self._skills_toolset is None:
                self._skills_toolset = self._create_toolset()
                return
            if (
                force
                or self._snapshot is None
                or not self._is_snapshot_valid_unlocked()
            ):
                self._skills_toolset.reload(include_registries=True)
        except (
            PydanticSkillValidationError,
            PydanticSkillRegistryError,
            AttributeError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            raise SkillValidationError(str(exc)) from exc

    def _create_toolset(self) -> SkillsToolset:
        """Create a SkillsToolset from filesystem or github:// source."""

        if self._skills_directory.startswith("github://"):
            self._skills_root_path = self._github_skill_downloader.download(
                cache_path=Path(".skills-git-cache"),
                skills_directory=self._skills_directory,
                github_token=self._environment_variables.skills_github_token,
            )
        else:
            self._skills_root_path = Path(self._skills_directory).expanduser().resolve()
        return SkillsToolset(directories=[str(self._skills_root_path)])

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

        configured = environment_variables.skills_cache_timeout_seconds
        if isinstance(configured, bool):
            return 3600.0
        if not isinstance(configured, (int, float)):
            return 3600.0
        configured_seconds = float(configured)
        if configured_seconds <= 0:
            return None
        return configured_seconds

    # Skill mapping and validation

    def _map_skill(self, skill: Skill) -> SkillDetails:
        """Map a pydantic-ai-skills Skill into framework SkillDetails."""

        normalized_name = self._normalize_skill_name(skill.name)
        if not normalized_name:
            raise SkillValidationError("Skill name must not be empty")

        description = (
            skill.description.strip() if isinstance(skill.description, str) else ""
        )
        if not description:
            raise SkillValidationError(
                f"Skill {normalized_name} must include a non-empty description"
            )

        compatibility_value = skill.compatibility
        if compatibility_value is not None and not isinstance(compatibility_value, str):
            raise SkillValidationError(
                f"Skill {normalized_name} compatibility must be a string when provided"
            )

        license_value = skill.license
        if license_value is not None and not isinstance(license_value, str):
            raise SkillValidationError(
                f"Skill {normalized_name} license must be a string when provided"
            )

        metadata_value: Mapping[str, object] = (
            skill.metadata if isinstance(skill.metadata, Mapping) else {}
        )
        metadata = self._normalize_metadata(
            skill_name=normalized_name,
            value=metadata_value.get("metadata"),
        )
        allowed_tools = self._normalize_allowed_tools(
            skill_name=normalized_name,
            value=metadata_value.get("allowed-tools"),
        )

        source_dir = (
            Path(skill.uri)
            if isinstance(skill.uri, str) and skill.uri.strip()
            else self._skills_root_path / normalized_name
        )
        source_path = source_dir / "SKILL.md"

        summary = SkillSummary(
            name=normalized_name,
            description=description,
            source_path=source_path,
            license=license_value.strip() if isinstance(license_value, str) else None,
            compatibility=compatibility_value.strip()
            if isinstance(compatibility_value, str)
            else None,
            metadata=metadata,
            allowed_tools=allowed_tools,
        )
        content = skill.content if isinstance(skill.content, str) else ""
        return SkillDetails(summary=summary, content=content, source_path=source_path)

    @staticmethod
    def _normalize_metadata(
        *,
        skill_name: str,
        value: object,
    ) -> dict[str, object]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise SkillValidationError(f"Skill {skill_name} metadata must be a mapping")
        metadata: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SkillValidationError(
                    f"Skill {skill_name} metadata keys must be strings"
                )
            metadata[key] = item
        return metadata

    @staticmethod
    def _normalize_allowed_tools(*, skill_name: str, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, str):
            raise SkillValidationError(
                f"Skill {skill_name} allowed-tools must be a space-delimited string"
            )
        return tuple(tool for tool in value.split() if tool)

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
        if "://" in normalized or "::" in normalized:
            return normalized
        return str(Path(normalized).expanduser())

    @staticmethod
    def _build_git_target_dir(git_location: GitLocation) -> Path:
        branch_suffix = f"-{git_location.branch}" if git_location.branch else ""
        directory_name = (
            f"{git_location.owner}-{git_location.repository}{branch_suffix}"
        )
        return Path(".skills-git-cache") / directory_name

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
