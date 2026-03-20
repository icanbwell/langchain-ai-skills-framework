from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Mapping, MutableMapping, Sequence, cast
from uuid import UUID, uuid4

import fsspec
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


class SkillDirectoryLoader(SkillLoaderProtocol):
    """Loads Agent Skills from a directory following the AgentSkills specification."""

    _skill_name_pattern = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

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
        self._filesystem, self._skills_path = fsspec.core.url_to_fs(
            self._skills_directory
        )
        self._lock = RLock()
        self._cache = cache
        self._snapshot: SkillCacheSnapshot | None = None
        self._excluded_skills = self._normalize_excluded_skills(
            environment_variables.excluded_skills
        )
        self._environment_variables = environment_variables
        logger.info(
            f"SkillDirectoryLoader {self._identifier}initialized for {self._skills_directory}"
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
                f"SkillDirectoryLoader {self._identifier} cache miss; loading skills"
            )
            snapshot = self._build_snapshot()
            self._cache.set(snapshot)
            self._snapshot = snapshot
            return snapshot

    def _build_snapshot(self) -> SkillCacheSnapshot:
        logger.info(
            f"SkillDirectoryLoader {self._identifier} scanning directory {self._skills_directory}"
        )
        if not self._path_exists(self._skills_path):
            logger.warning(
                "Skills directory %s does not exist. No skills will be available.",
                self._skills_directory,
            )
            return SkillCacheSnapshot(
                details_by_name=MappingProxyType({}), ordered_summaries=()
            )

        if not self._is_dir(self._skills_path):
            raise SkillValidationError(
                f"Configured skills path '{self._skills_directory}' is not a directory"
            )

        new_details: dict[str, SkillDetails] = {}
        new_summaries: list[SkillSummary] = []
        excluded_skills = self._get_excluded_skills()
        excluded_skill_groups = self._get_excluded_skill_groups()
        for skill_dir, skill_file in self._iter_skill_files(self._skills_path):
            skill_group = self._resolve_skill_group(skill_dir)
            if skill_group and skill_group in excluded_skill_groups:
                logger.info("Skipping excluded skill group '%s'", skill_group)
                continue
            definition = self._parse_skill(Path(skill_dir).name, skill_file)
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
            f"Loaded {len(ordered_summaries)} Agent Skills from {self._skills_directory}"
        )
        return snapshot

    def _iter_skill_files(self, skills_directory: str) -> Sequence[tuple[str, str]]:
        skill_files: list[tuple[str, str]] = []
        for entry in self._list_directories(skills_directory):
            skill_file = self._join_path(entry, "SKILL.md")
            if self._is_file(skill_file):
                skill_files.append((entry, skill_file))
                continue
            nested_skill_files: list[tuple[str, str]] = []
            for nested_entry in self._list_directories(entry):
                nested_skill_file = self._join_path(nested_entry, "SKILL.md")
                if self._is_file(nested_skill_file):
                    nested_skill_files.append((nested_entry, nested_skill_file))
            if nested_skill_files:
                skill_files.extend(nested_skill_files)
                continue
            logger.warning(
                "Skipping skill directory %s because SKILL.md is missing",
                entry,
            )
        return tuple(skill_files)

    def _parse_skill(self, directory_name: str, skill_file: str) -> SkillDetails:
        raw_content = self._read_text(skill_file)
        normalized = raw_content.replace("\r\n", "\n")
        if not normalized.startswith("---\n"):
            raise SkillValidationError(
                f"Skill {directory_name} missing YAML frontmatter header"
            )
        closing_index = normalized.find("\n---", 4)
        if closing_index == -1:
            raise SkillValidationError(
                f"Skill {directory_name} missing YAML frontmatter terminator"
            )
        frontmatter_text = normalized[4:closing_index]
        body = normalized[closing_index + len("\n---") :].lstrip("\n")
        data = self._load_frontmatter(frontmatter_text)
        skill_name: str | None = cast(str | None, data.get("name"))
        description: str | None = cast(str | None, data.get("description"))
        license_value = data.get("license")
        compatibility_value = data.get("compatibility")
        metadata_value = data.get("metadata", {})
        allowed_tools_value = data.get("allowed-tools")
        if not isinstance(skill_name, str):
            raise SkillValidationError(
                f"Skill {directory_name} is missing the required 'name' field"
            )
        normalized_name = self._normalize_skill_name(skill_name)
        if skill_name != normalized_name:
            raise SkillValidationError(
                "Skill names must be lowercase and use hyphens only"
            )
        if len(skill_name) > 64:
            raise SkillValidationError(
                f"Skill skill_name '{skill_name}' exceeds 64 characters"
            )
        if not self._skill_name_pattern.fullmatch(normalized_name):
            raise SkillValidationError(
                f"Skill skill_name '{skill_name}' contains invalid characters"
            )
        if not isinstance(description, str) or not description.strip():
            raise SkillValidationError(
                f"Skill {skill_name} must include a non-empty description"
            )
        if len(description) > 1024:
            raise SkillValidationError(
                f"Skill {skill_name} description exceeds 1024 characters"
            )
        if compatibility_value is not None:
            if (
                not isinstance(compatibility_value, str)
                or not compatibility_value.strip()
            ):
                raise SkillValidationError(
                    f"Skill {skill_name} compatibility must be a non-empty string when provided"
                )
            if len(compatibility_value) > 500:
                raise SkillValidationError(
                    f"Skill {skill_name} compatibility exceeds 500 characters"
                )
        if license_value is not None and not isinstance(license_value, str):
            raise SkillValidationError(
                f"Skill {skill_name} license must be a string when provided"
            )
        metadata: MutableMapping[str, str] = {}
        if metadata_value is not None:
            if not isinstance(metadata_value, Mapping):
                raise SkillValidationError(
                    f"Skill {skill_name} metadata must be a mapping of string keys to string values"
                )
            for key, value in metadata_value.items():
                if not isinstance(key, str):
                    raise SkillValidationError(
                        f"Skill {skill_name} metadata entries must be strings: {type(key)}, {type(value)}"
                    )
                metadata[key] = str(value)
        allowed_tools: tuple[str, ...] = ()
        if isinstance(allowed_tools_value, str):
            allowed_tools = tuple(tool for tool in allowed_tools_value.split() if tool)
        elif allowed_tools_value is not None:
            raise SkillValidationError(
                f"Skill {skill_name} allowed-tools must be a space-delimited string"
            )
        source_path = Path(skill_file)
        summary = SkillSummary(
            name=normalized_name,
            description=description.strip(),
            source_path=source_path,
            license=license_value.strip() if isinstance(license_value, str) else None,
            compatibility=(
                compatibility_value.strip()
                if isinstance(compatibility_value, str)
                else None
            ),
            metadata=dict(metadata),
            allowed_tools=allowed_tools,
        )
        if not body.strip():
            logger.warning("Skill %s has empty body content", normalized_name)
        return SkillDetails(summary=summary, content=body, source_path=source_path)

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
        return bool(self._filesystem.exists(path))

    def _is_dir(self, path: str) -> bool:
        return bool(self._filesystem.isdir(path))

    def _is_file(self, path: str) -> bool:
        return bool(self._filesystem.isfile(path))

    def _list_directories(self, path: str) -> tuple[str, ...]:
        entries = self._filesystem.ls(path, detail=True)
        items: Sequence[object]
        if isinstance(entries, Mapping):
            items = tuple(entries.values())
        else:
            items = tuple(entries)
        directories: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                name = item.get("name")
                if not isinstance(name, str):
                    continue
                entry_type = item.get("type")
                if entry_type in {"directory", "dir"} or self._is_dir(name):
                    directories.append(name)
                continue
            if isinstance(item, str) and self._is_dir(item):
                directories.append(item)
        return tuple(sorted(directories))

    @staticmethod
    def _join_path(parent: str, child: str) -> str:
        if not parent:
            return child
        return f"{parent.rstrip('/')}/{child}"

    def _read_text(self, path: str) -> str:
        with self._filesystem.open(path, mode="rb") as file_handle:
            data = file_handle.read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
        raise SkillValidationError(f"Skill file '{path}' content is not text")

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
    def _load_frontmatter(frontmatter_text: str) -> MutableMapping[str, object]:
        try:
            loaded = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as exc:
            raise SkillValidationError("Invalid YAML frontmatter") from exc
        if not isinstance(loaded, MutableMapping):
            raise SkillValidationError("Frontmatter must be a mapping")
        return loaded

    @staticmethod
    def _normalize_skill_name(value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        normalized = re.sub(r"-+", "-", normalized)
        return normalized.strip("-")
