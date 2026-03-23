import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Mapping, Sequence
from uuid import UUID, uuid4

import yaml
from skillkit import SkillManager, SkillMetadata

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["CONFIG"])

_FRONTMATTER_PATTERN = re.compile(
    r"^---[\r\n]+(.*?)[\r\n]+---", re.DOTALL | re.MULTILINE
)


@dataclass(frozen=True, slots=True)
class _SkillSnapshot:
    """Immutable, already-filtered view of skills used by public loader calls."""

    details_by_name: Mapping[str, SkillDetails]
    ordered_summaries: tuple[SkillSummary, ...]


class SkillkitDirectoryLoader(SkillLoaderProtocol):
    """Loads Agent Skills from local directories using skillkit."""

    # Public API

    def __init__(
        self,
        *,
        environment_variables: SkillLoaderEnvironmentVariables,
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
            raise SkillValidationError("skills_directory must be a string or Path")

        if not configured_directory.strip():
            raise SkillValidationError("skills_directory is not configured")

        self._skills_directory = self._normalize_skills_directory(configured_directory)
        self._skills_root_path = Path(self._skills_directory).expanduser().resolve()
        self._lock = RLock()
        self._snapshot: _SkillSnapshot | None = None
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

    def list_skill_summaries(
        self, *, allowed_skills: set[str]
    ) -> Sequence[SkillSummary]:
        """Return lightweight skill summaries from the current snapshot."""

        del allowed_skills
        snapshot = self._get_snapshot()
        logger.debug(
            "SkillkitDirectoryLoader %s returning %d summaries",
            self._identifier,
            len(snapshot.ordered_summaries),
        )
        return snapshot.ordered_summaries

    def get_skill_details(self, *, skill_name: str) -> SkillDetails:
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

    async def get_instructions(self) -> str:
        """Return `<available_skills>` block used by middleware system prompts."""

        snapshot = self._get_snapshot()
        skills_lines = "".join(
            (
                "<skill><name> "
                f"{summary.name}"
                " </name><description> "
                f"{summary.description}"
                " </description></skill>"
            )
            for summary in snapshot.ordered_summaries
        )
        return f"\n\n<available_skills>{skills_lines}</available_skills>\n\n"

    # Snapshot lifecycle

    def _get_snapshot(self) -> _SkillSnapshot:
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

    def _build_snapshot(self) -> _SkillSnapshot:
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
        snapshot = _SkillSnapshot(
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
                self._manager.discover()
        except Exception as exc:
            raise SkillValidationError(str(exc)) from exc

    def _create_manager(self) -> SkillManager:
        """Create a skillkit SkillManager from a local directory."""

        if self._skills_directory.startswith("github://"):
            raise SkillValidationError(
                "SkillkitDirectoryLoader does not support github:// skill directories; "
                "use a local path"
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
            metadata=metadata.__dict__,
            allowed_tools=metadata.allowed_tools,
        )
        return SkillDetails(
            summary=summary, content=content, source_path=metadata.skill_path
        )

    @staticmethod
    def _extract_frontmatter_and_content(
        *,
        skill_name: str,
        source_path: Path,
        skill_content: str,
    ) -> tuple[Mapping[str, object], str]:
        if not isinstance(skill_content, str):
            raise SkillValidationError(f"Skill {skill_name} content must be a string")

        match = _FRONTMATTER_PATTERN.match(skill_content)
        if match is None:
            raise SkillValidationError(
                f"Skill {skill_name} at {source_path} is missing YAML frontmatter delimiters"
            )

        try:
            parsed_frontmatter = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise SkillValidationError(
                f"Skill {skill_name} at {source_path} has invalid YAML frontmatter"
            ) from exc

        if not isinstance(parsed_frontmatter, Mapping):
            raise SkillValidationError(
                f"Skill {skill_name} at {source_path} frontmatter must be a mapping"
            )

        content = skill_content[match.end() :].lstrip("\r\n")
        normalized_frontmatter = {
            str(key): value for key, value in parsed_frontmatter.items()
        }
        return normalized_frontmatter, content

    @staticmethod
    def _normalize_optional_string(
        *,
        skill_name: str,
        field_name: str,
        value: object,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise SkillValidationError(
                f"Skill {skill_name} {field_name} must be a string when provided"
            )
        normalized = value.strip()
        return normalized or None

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
        return str(Path(value.strip()).expanduser())

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
