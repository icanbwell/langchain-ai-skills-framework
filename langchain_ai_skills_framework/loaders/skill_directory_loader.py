from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Mapping, Sequence
from uuid import UUID, uuid4

from pydantic_ai_skills import SkillsDirectory
from pydantic_ai_skills.exceptions import (
    SkillRegistryError as PydanticSkillRegistryError,
    SkillValidationError as PydanticSkillValidationError,
)
from pydantic_ai_skills.registries import GitCloneOptions, GitSkillsRegistry
from pydantic_ai_skills.types import Skill
import yaml

from langchain_ai_skills_framework.cache.skill_cache import (
    SkillCache,
    SkillCacheSnapshot,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["CONFIG"])


@dataclass(frozen=True, slots=True)
class _GitLocation:
    repo_url: str
    path: str
    branch: str | None


class SkillDirectoryLoader(SkillLoaderProtocol):
    """Loads Agent Skills from local directories or GitHub repositories."""

    _skill_name_pattern = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    _ordered_skill_directory_pattern = re.compile(
        r"^(?P<prefix>\d+)-(?P<skill_name>[a-z0-9]+(?:-[a-z0-9]+)*)$"
    )
    _github_uri_pattern = re.compile(
        r"^github://(?P<repo_spec>[^/]+)(?:/(?P<skills_path>.*))?$"
    )

    def __init__(
        self,
        *,
        cache: SkillCache,
        environment_variables: SkillLoaderEnvironmentVariables,
    ) -> None:
        self._identifier: UUID = uuid4()
        if cache is None:
            raise ValueError("cache must not be None")
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
        self._skills_path = self._skills_directory
        self._skills_root_path = self._resolve_skills_root_path(self._skills_directory)
        self._lock = RLock()
        self._cache = cache
        self._snapshot: SkillCacheSnapshot | None = None
        self._excluded_skills = self._normalize_excluded_skills(
            environment_variables.excluded_skills
        )
        self._environment_variables = environment_variables
        logger.info(
            "SkillDirectoryLoader %s initialized for %s",
            self._identifier,
            self._skills_directory,
        )

    def list_skill_summaries(self) -> Sequence[SkillSummary]:
        snapshot = self._get_snapshot()
        logger.debug(
            "SkillDirectoryLoader %s returning %d summaries",
            self._identifier,
            len(snapshot.ordered_summaries),
        )
        return snapshot.ordered_summaries

    def get_skill_details(self, skill_name: str) -> SkillDetails:
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
        with self._lock:
            logger.info("SkillDirectoryLoader %s refreshing cache", self._identifier)
            self._cache.clear()
            self._snapshot = None
            snapshot = self._build_snapshot()
            self._cache.set(snapshot)
            self._snapshot = snapshot

    def _get_snapshot(self) -> SkillCacheSnapshot:
        if self._snapshot is not None:
            logger.debug(
                "SkillDirectoryLoader %s using instance snapshot",
                self._identifier,
            )
            return self._snapshot

        cached_snapshot = self._cache.get()
        if cached_snapshot is not None:
            logger.debug(
                "SkillDirectoryLoader %s using shared cached snapshot",
                self._identifier,
            )
            self._snapshot = cached_snapshot
            return cached_snapshot

        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
            cached_snapshot = self._cache.get()
            if cached_snapshot is not None:
                logger.debug(
                    "SkillDirectoryLoader %s observed cache fill while waiting",
                    self._identifier,
                )
                self._snapshot = cached_snapshot
                return cached_snapshot

            logger.info(
                "SkillDirectoryLoader %s cache miss; loading skills",
                self._identifier,
            )
            snapshot = self._build_snapshot()
            self._cache.set(snapshot)
            self._snapshot = snapshot
            return snapshot

    def _build_snapshot(self) -> SkillCacheSnapshot:
        logger.info(
            "SkillDirectoryLoader %s scanning directory %s",
            self._identifier,
            self._skills_directory,
        )
        new_details: dict[str, SkillDetails] = {}
        new_summaries: list[SkillSummary] = []
        excluded_skills = self._get_excluded_skills()
        excluded_skill_groups = self._get_excluded_skill_groups()
        skills = self._load_skills_from_source()
        if not skills and not self._skills_directory.startswith("github://"):
            logger.warning(
                "Skills directory %s does not exist or has no valid skills.",
                self._skills_directory,
            )
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
        snapshot = SkillCacheSnapshot(
            details_by_name=MappingProxyType(new_details),
            ordered_summaries=ordered_summaries,
        )
        logger.info(
            "Loaded %d Agent Skills from %s",
            len(ordered_summaries),
            self._skills_directory,
        )
        return snapshot

    def _load_skills_from_source(self) -> tuple[Skill, ...]:
        try:
            if self._skills_directory.startswith("github://"):
                git_location = self._parse_github_uri(self._skills_directory)
                registry = GitSkillsRegistry(
                    repo_url=git_location.repo_url,
                    path=git_location.path,
                    token=self._environment_variables.skills_github_token,
                    clone_options=GitCloneOptions(
                        depth=1,
                        branch=git_location.branch,
                        single_branch=git_location.branch is not None,
                    ),
                )
                self._skills_root_path = registry._skills_root()
                self._skills_path = str(self._skills_root_path)
                self._validate_skill_markdown_files(self._skills_root_path)
                return tuple(registry.get_skills())
            self._skills_root_path = Path(self._skills_directory).expanduser().resolve()
            self._skills_path = str(self._skills_root_path)
            if not self._skills_root_path.exists():
                return ()
            if not self._skills_root_path.is_dir():
                raise SkillValidationError(
                    f"Configured skills path '{self._skills_directory}' is not a directory"
                )
            self._validate_skill_markdown_files(self._skills_root_path)
            source = SkillsDirectory(
                path=self._skills_directory, validate=True, max_depth=2
            )
            loaded = source.get_skills()
            return tuple(loaded.values())
        except (
            PydanticSkillValidationError,
            PydanticSkillRegistryError,
            AttributeError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            raise SkillValidationError(str(exc)) from exc

    @staticmethod
    def _validate_skill_markdown_files(skills_root: Path) -> None:
        for skill_file in skills_root.rglob("SKILL.md"):
            raw_content = skill_file.read_text(encoding="utf-8")
            normalized = raw_content.replace("\r\n", "\n")
            if not normalized.startswith("---\n"):
                raise SkillValidationError(
                    f"Skill {skill_file.parent.name} missing YAML frontmatter header"
                )
            match = re.match(
                r"^---\n(?P<frontmatter>.*?)\n---(?:\n|$)",
                normalized,
                re.DOTALL,
            )
            if match is None:
                raise SkillValidationError(
                    f"Skill {skill_file.parent.name} missing YAML frontmatter terminator"
                )
            frontmatter_text = match.group("frontmatter")
            try:
                loaded_frontmatter = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError as exc:
                raise SkillValidationError("Invalid YAML frontmatter") from exc
            if not isinstance(loaded_frontmatter, Mapping):
                raise SkillValidationError("Frontmatter must be a mapping")
            skill_name = loaded_frontmatter.get("name")
            if not isinstance(skill_name, str):
                raise SkillValidationError(
                    f"Skill {skill_file.parent.name} is missing the required 'name' field"
                )

    def _map_skill(self, skill: Skill) -> SkillDetails:
        normalized_name = self._normalize_skill_name(skill.name)
        if skill.name != normalized_name:
            raise SkillValidationError(
                "Skill names must be lowercase and use hyphens only"
            )
        if len(normalized_name) > 64:
            raise SkillValidationError(
                f"Skill skill_name '{normalized_name}' exceeds 64 characters"
            )
        if not self._skill_name_pattern.fullmatch(normalized_name):
            raise SkillValidationError(
                f"Skill skill_name '{normalized_name}' contains invalid characters"
            )

        source_dir = (
            Path(skill.uri)
            if isinstance(skill.uri, str)
            else self._skills_root_path / normalized_name
        )
        normalized_directory_name = self._normalize_skill_name(source_dir.name)
        if not self._directory_matches_skill_name(
            normalized_directory_name=normalized_directory_name,
            normalized_skill_name=normalized_name,
        ):
            logger.warning(
                "Skill name '%s' does not match directory '%s'; using frontmatter name",
                skill.name,
                source_dir.name,
            )

        description = skill.description
        if not isinstance(description, str) or not description.strip():
            raise SkillValidationError(
                f"Skill {normalized_name} must include a non-empty description"
            )
        if len(description) > 1024:
            raise SkillValidationError(
                f"Skill {normalized_name} description exceeds 1024 characters"
            )

        compatibility = skill.compatibility
        if compatibility is not None:
            if not isinstance(compatibility, str) or not compatibility.strip():
                raise SkillValidationError(
                    f"Skill {normalized_name} compatibility must be a non-empty string when provided"
                )
            if len(compatibility) > 500:
                raise SkillValidationError(
                    f"Skill {normalized_name} compatibility exceeds 500 characters"
                )

        license_value = skill.license
        if license_value is not None and not isinstance(license_value, str):
            raise SkillValidationError(
                f"Skill {normalized_name} license must be a string when provided"
            )

        metadata_value = skill.metadata if isinstance(skill.metadata, Mapping) else {}
        frontmatter_metadata = metadata_value.get("metadata")
        metadata = self._normalize_metadata(
            skill_name=normalized_name,
            value=frontmatter_metadata,
        )
        allowed_tools_value = metadata_value.get("allowed-tools")
        allowed_tools = self._normalize_allowed_tools(
            skill_name=normalized_name,
            value=allowed_tools_value,
        )

        source_path = source_dir / "SKILL.md"
        summary = SkillSummary(
            name=normalized_name,
            description=description.strip(),
            source_path=source_path,
            license=license_value.strip() if isinstance(license_value, str) else None,
            compatibility=compatibility.strip()
            if isinstance(compatibility, str)
            else None,
            metadata=metadata,
            allowed_tools=allowed_tools,
        )
        content = skill.content if isinstance(skill.content, str) else ""
        if not content.strip():
            logger.warning("Skill %s has empty body content", normalized_name)
        return SkillDetails(summary=summary, content=content, source_path=source_path)

    @classmethod
    def _normalize_metadata(
        cls,
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
            if not isinstance(key, str):
                raise SkillValidationError(
                    f"Skill {skill_name} metadata keys must be strings: {type(key)}"
                )
            if isinstance(item, str):
                metadata[key] = item
                continue
            if isinstance(item, datetime):
                metadata[key] = item.isoformat()
                continue
            if isinstance(item, date):
                metadata[key] = item.isoformat()
                continue
            if isinstance(item, list) and all(isinstance(entry, str) for entry in item):
                metadata[key] = ", ".join(item)
                continue
            raise SkillValidationError(
                f"Skill {skill_name} metadata values must be strings or lists of strings: {type(item)}"
            )
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

    def _get_excluded_skills(self) -> frozenset[str]:
        if self._environment_variables is None:
            return self._excluded_skills
        return self._normalize_excluded_skills(
            self._environment_variables.excluded_skills
        )

    def _get_excluded_skill_groups(self) -> frozenset[str]:
        if self._environment_variables is None:
            return frozenset()
        return self._normalize_excluded_skill_groups(
            self._environment_variables.excluded_skill_groups
        )

    def _resolve_skill_group(self, skill_dir: str) -> str | None:
        try:
            relative = PurePosixPath(skill_dir).relative_to(
                PurePosixPath(self._skills_path)
            )
        except ValueError:
            return None
        if len(relative.parts) < 2:
            return None
        return self._normalize_skill_group(relative.parts[0])

    @staticmethod
    def _normalize_skills_directory(value: str) -> str:
        normalized = value.strip()
        if "://" in normalized or "::" in normalized:
            return normalized
        return str(Path(normalized).expanduser())

    def _path_exists(self, path: str) -> bool:
        return Path(path).exists()

    def _is_dir(self, path: str) -> bool:
        return Path(path).is_dir()

    @classmethod
    def _parse_github_uri(cls, skills_directory: str) -> _GitLocation:
        match = cls._github_uri_pattern.fullmatch(skills_directory)
        if match is None:
            raise SkillValidationError(
                "GitHub skill directory must match github://<owner>:<repo>[@branch]/<path>"
            )
        repository_spec = match.group("repo_spec")
        path_value = (match.group("skills_path") or "").strip("/")
        repository_without_ref, separator, branch = repository_spec.partition("@")
        if ":" not in repository_without_ref:
            raise SkillValidationError(
                "GitHub skill directory must include owner and repo, e.g. github://my-org:private-skills@main/skills"
            )
        owner, repo = repository_without_ref.split(":", 1)
        if not owner or not repo:
            raise SkillValidationError(
                "GitHub skill directory must include owner and repo, e.g. github://my-org:private-skills@main/skills"
            )
        normalized_branch = branch if separator and branch else None
        return _GitLocation(
            repo_url=f"https://github.com/{owner}/{repo}.git",
            path=path_value,
            branch=normalized_branch,
        )

    @staticmethod
    def _resolve_skills_root_path(skills_directory: str) -> Path:
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
            cls._normalize_skill_group(value)
            for value in values
            if isinstance(value, str) and value.strip()
        }
        return frozenset(item for item in normalized if item)

    @classmethod
    def _normalize_skill_group(cls, value: str) -> str:
        return cls._normalize_skill_name(value)

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

    @classmethod
    def _directory_matches_skill_name(
        cls, *, normalized_directory_name: str, normalized_skill_name: str
    ) -> bool:
        if normalized_directory_name == normalized_skill_name:
            return True
        ordered_match = cls._ordered_skill_directory_pattern.fullmatch(
            normalized_directory_name
        )
        if ordered_match is None:
            return False
        return ordered_match.group("skill_name") == normalized_skill_name
