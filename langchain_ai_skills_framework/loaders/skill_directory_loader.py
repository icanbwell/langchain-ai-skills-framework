from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Mapping, Sequence, cast
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

from pydantic_ai_skills import SkillsToolset
from pydantic_ai_skills.exceptions import (
    SkillRegistryError as PydanticSkillRegistryError,
    SkillValidationError as PydanticSkillValidationError,
)
from pydantic_ai_skills.registries import GitCloneOptions, GitSkillsRegistry
from pydantic_ai_skills.types import Skill

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


@dataclass(frozen=True, slots=True)
class _SkillSnapshot:
    """Immutable, already-filtered view of skills used by public loader calls."""

    details_by_name: Mapping[str, SkillDetails]
    ordered_summaries: tuple[SkillSummary, ...]


@dataclass(frozen=True, slots=True)
class _GitLocation:
    repo_url: str
    owner: str
    repository: str
    path: str
    branch: str | None


class SkillDirectoryLoader(SkillLoaderProtocol):
    """Loads Agent Skills from local directories or GitHub repositories."""

    _github_uri_example = "github://my-org/private-skills/skills?ref=main"

    # Public API

    def __init__(
        self,
        *,
        environment_variables: SkillLoaderEnvironmentVariables,
    ) -> None:
        self._identifier: UUID = uuid4()
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")

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
        self._snapshot: _SkillSnapshot | None = None
        self._snapshot_loaded_at: float | None = None
        self._skills_toolset: SkillsToolset | None = None
        self._reload_ttl_seconds = self._resolve_reload_ttl_seconds(
            environment_variables
        )
        self._environment_variables = environment_variables

        logger.info(
            "SkillDirectoryLoader %s initialized for %s",
            self._identifier,
            self._skills_directory,
        )

    def list_skill_summaries(self) -> Sequence[SkillSummary]:
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

    def refresh(self) -> None:
        """Force an immediate reload regardless of TTL."""

        with self._lock:
            logger.info("SkillDirectoryLoader %s refreshing cache", self._identifier)
            self._reload_toolset(force=True)
            self._snapshot = self._build_snapshot()
            self._snapshot_loaded_at = time.monotonic()

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
                "SkillDirectoryLoader %s cache expired or empty; loading skills",
                self._identifier,
            )
            self._reload_toolset(force=False)
            self._snapshot = self._build_snapshot()
            self._snapshot_loaded_at = time.monotonic()
            return self._snapshot

    def _build_snapshot(self) -> _SkillSnapshot:
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
                reload_method = getattr(self._skills_toolset, "reload", None)
                if callable(reload_method):
                    # Keep remote registries fresh on TTL refresh.
                    reload_method(include_registries=True)
                else:
                    # Backward-compatible fallback for older pydantic-ai-skills builds.
                    self._skills_toolset = self._create_toolset()
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
            git_location = self._parse_github_uri(self._skills_directory)
            registry = GitSkillsRegistry(
                repo_url=git_location.repo_url,
                path=git_location.path,
                target_dir=self._build_git_target_dir(git_location),
                token=self._environment_variables.skills_github_token,
                clone_options=GitCloneOptions(
                    depth=1,
                    branch=git_location.branch,
                    single_branch=True,
                ),
            )
            self._skills_root_path = registry._skills_root()
            return SkillsToolset(registries=[registry])

        self._skills_root_path = Path(self._skills_directory).expanduser().resolve()
        return SkillsToolset(directories=[str(self._skills_root_path)])

    # TTL helpers

    def _is_snapshot_valid(self) -> bool:
        """Thread-safe snapshot validity check."""

        with self._lock:
            return self._is_snapshot_valid_unlocked()

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
    ) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise SkillValidationError(
                f"Skill {skill_name} metadata must be a mapping of string keys to string values"
            )
        metadata: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise SkillValidationError(
                    f"Skill {skill_name} metadata values must be strings"
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

    @classmethod
    def _parse_github_uri(cls, skills_directory: str) -> _GitLocation:
        parsed = urlsplit(skills_directory)
        if parsed.scheme != "github":
            raise SkillValidationError(
                "GitHub skill directory must match github://<owner>/<repo>/<path>?ref=<branch>"
            )
        if parsed.fragment:
            raise SkillValidationError(
                "GitHub skill directory must not include a fragment"
            )

        query_values = parse_qs(parsed.query, keep_blank_values=True)
        unsupported_query_params = set(query_values.keys()) - {"ref"}
        if unsupported_query_params:
            unsupported = ", ".join(sorted(unsupported_query_params))
            raise SkillValidationError(
                f"GitHub skill directory supports only '?ref=' query parameter; got: {unsupported}"
            )

        ref_values = query_values.get("ref")
        if ref_values and len(ref_values) > 1:
            raise SkillValidationError(
                "GitHub skill directory must include a single '?ref=' value"
            )
        if ref_values is not None and not ref_values[0].strip():
            raise SkillValidationError(
                "GitHub skill directory '?ref=' value must not be empty"
            )
        branch_from_query = ref_values[0].strip() if ref_values else None

        owner = parsed.netloc.strip()
        path_parts = [part for part in parsed.path.split("/") if part]

        # Backward compatibility for the legacy owner:repo style while callers migrate.
        if ":" in owner:
            repository_without_ref, separator, branch = owner.partition("@")
            if ":" not in repository_without_ref:
                raise SkillValidationError(
                    f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
                )
            legacy_owner, repo = repository_without_ref.split(":", 1)
            if not legacy_owner or not repo:
                raise SkillValidationError(
                    f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
                )
            if (
                branch_from_query is not None
                and separator
                and branch
                and branch_from_query != branch
            ):
                raise SkillValidationError(
                    "GitHub skill directory ref mismatch between legacy '@branch' and '?ref='"
                )
            owner = legacy_owner
            path_value = "/".join(path_parts)
            normalized_branch = (
                branch_from_query
                if branch_from_query is not None
                else (branch if separator and branch else None)
            )
        else:
            if not owner or not path_parts:
                raise SkillValidationError(
                    f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
                )
            repo = path_parts[0]
            path_value = "/".join(path_parts[1:])
            normalized_branch = branch_from_query

        if not owner or not repo:
            raise SkillValidationError(
                f"GitHub skill directory must include owner and repo, e.g. {cls._github_uri_example}"
            )

        return _GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            owner=owner,
            repository=repo,
            path=path_value,
            branch=normalized_branch,
        )

    @staticmethod
    def _build_git_target_dir(git_location: _GitLocation) -> Path:
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
