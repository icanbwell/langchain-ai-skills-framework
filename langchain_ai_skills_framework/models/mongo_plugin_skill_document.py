"""MongoDB document models for plugin-scoped skills, resources, and scripts.

Each document carries a ``path`` field following the **Materialized Paths**
tree-structure pattern (see
https://www.mongodb.com/docs/manual/applications/data-models-tree-structures/).
The path mirrors the on-disk plugin layout from the Claude Code plugins
specification, enabling tree-style queries such as "all files in a plugin"
or "all resources for a skill".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Path builders — Materialized Paths pattern
# ---------------------------------------------------------------------------


def _normalize_folder(folder: str | None) -> str | None:
    """Coerce empty-string folder to None for consistent truthiness checks."""
    if folder is None:
        return None
    stripped = folder.strip()
    return stripped if stripped else None


def _skill_base_path(*, plugin_name: str, skill_name: str, folder: str | None = None) -> str:
    """Return the base path prefix for a skill: ``plugin/skills/[folder/]name``."""
    folder = _normalize_folder(folder)
    if folder is not None:
        return f"{plugin_name}/skills/{folder}/{skill_name}"
    return f"{plugin_name}/skills/{skill_name}"


def normalize_folder(folder: str | None) -> str | None:
    """Public alias for folder normalization (coerces '' to None)."""
    return _normalize_folder(folder)


def build_skill_path(*, plugin_name: str, skill_name: str, folder: str | None = None) -> str:
    """Return the canonical path for a skill's ``SKILL.md``."""
    return f"{_skill_base_path(plugin_name=plugin_name, skill_name=skill_name, folder=folder)}/SKILL.md"


def build_resource_path(*, plugin_name: str, skill_name: str, resource_name: str, folder: str | None = None) -> str:
    """Return the canonical path for a skill resource file."""
    return f"{_skill_base_path(plugin_name=plugin_name, skill_name=skill_name, folder=folder)}/{resource_name}"


def build_script_path(*, plugin_name: str, skill_name: str, script_name: str, folder: str | None = None) -> str:
    """Return the canonical path for a skill script file."""
    return f"{_skill_base_path(plugin_name=plugin_name, skill_name=skill_name, folder=folder)}/scripts/{script_name}"


# ---------------------------------------------------------------------------
# Skill document
# ---------------------------------------------------------------------------


class MongoPluginSkillDocument(BaseModel):
    """A skill stored in the ``plugin_skills`` collection."""

    model_config = ConfigDict(extra="ignore")

    SCHEMA_VERSION: ClassVar[int] = 2

    plugin_name: str = Field(description="Plugin that owns this skill")
    skill_name: str = Field(description="Normalized name of the skill")
    folder: str | None = Field(default=None, description="Optional subfolder path within the plugin")
    path: str = Field(default="", description="Materialized path: plugin/skills/name/SKILL.md")
    description: str = Field(default="", description="Short description of what the skill does")
    content: str = Field(default="", description="Full skill content (SKILL.md body)")
    allowed_tools: tuple[str, ...] = Field(
        default=(),
        description="Tool names this skill is allowed to use",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata from the skill frontmatter",
    )
    author: str = Field(description="'system' for marketplace-synced, actual user id for user-saved")
    state: str = Field(default="draft", description="Skill lifecycle state: draft, staging, in_review, or published")
    published_date: datetime | None = Field(
        default=None,
        description="When the skill was last published (or unpublished)",
    )
    published_branch: str | None = Field(
        default=None,
        description="Git branch used for the marketplace publish PR",
    )
    modified_by: str = Field(default="", description="ID of the user who last modified this skill")
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was first saved",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["allowed_tools"] = list(self.allowed_tools)
        return data

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginSkillDocument:
        normalized = dict(data)
        if "author" not in normalized and "user_id" in normalized:
            normalized["author"] = normalized.pop("user_id")
        if "state" not in normalized:
            if normalized.get("published") or normalized.get("shared"):
                normalized["state"] = "published"
            else:
                normalized["state"] = "draft"
        return cls.model_validate(normalized)


# ---------------------------------------------------------------------------
# Resource document
# ---------------------------------------------------------------------------


class MongoPluginResourceDocument(BaseModel):
    """A resource file stored in the ``plugin_references`` collection."""

    model_config = ConfigDict(extra="ignore")

    plugin_name: str = Field(description="Plugin that owns this resource")
    skill_name: str = Field(description="Normalized name of the parent skill")
    resource_name: str = Field(description="Name of the resource file")
    path: str = Field(default="", description="Materialized path: plugin/skills/name/resource")
    content: str = Field(default="", description="Content of the resource file")
    author: str = Field(description="'system' for marketplace-synced, actual user id for user-saved")
    modified_by: str = Field(default="", description="ID of the user who last modified this resource")
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the resource was first saved",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the resource was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginResourceDocument:
        normalized = dict(data)
        if "author" not in normalized and "user_id" in normalized:
            normalized["author"] = normalized.pop("user_id")
        return cls.model_validate(normalized)


# ---------------------------------------------------------------------------
# Script document
# ---------------------------------------------------------------------------


class MongoPluginScriptDocument(BaseModel):
    """An executable script stored in the ``plugin_scripts`` collection."""

    model_config = ConfigDict(extra="ignore")

    plugin_name: str = Field(description="Plugin that owns this script")
    skill_name: str = Field(description="Normalized name of the parent skill")
    script_name: str = Field(description="Name of the script file")
    path: str = Field(default="", description="Materialized path: plugin/skills/name/scripts/script")
    content: str = Field(default="", description="Content of the script file")
    author: str = Field(description="'system' for marketplace-synced, actual user id for user-saved")
    modified_by: str = Field(default="", description="ID of the user who last modified this script")
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the script was first saved",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the script was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginScriptDocument:
        normalized = dict(data)
        if "author" not in normalized and "user_id" in normalized:
            normalized["author"] = normalized.pop("user_id")
        return cls.model_validate(normalized)


# ---------------------------------------------------------------------------
# Usage document (unchanged structure, kept for continuity)
# ---------------------------------------------------------------------------


class MongoPluginDefinitionDocument(BaseModel):
    """A plugin definition stored in the ``plugins`` collection.

    Captures the plugin manifest metadata, skill list, and MCP server
    configuration so the gateway has a queryable catalog of available plugins.
    """

    model_config = ConfigDict(extra="ignore")

    plugin_name: str = Field(description="Canonical plugin name (directory name)")
    description: str = Field(default="", description="Human-readable description")
    skills: list[str] = Field(default_factory=list, description="Skill names provided by this plugin")
    mcp_servers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="MCP server configurations from the plugin's .mcp.json",
    )
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the plugin was first registered",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the plugin was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginDefinitionDocument:
        plugin_name = data.get("plugin_name") or data.get("name", "")
        if not plugin_name:
            raise KeyError("plugin_name")
        normalized = dict(data)
        normalized["plugin_name"] = plugin_name
        return cls.model_validate(normalized)


class MongoPluginSkillUsageDocument(BaseModel):
    """A single usage event for a skill within a plugin."""

    model_config = ConfigDict(extra="ignore")

    plugin_name: str = Field(description="Plugin containing the skill")
    skill_name: str = Field(description="Name of the skill that was used")
    author: str = Field(description="ID of the user who used the skill")
    date_used: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was used",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginSkillUsageDocument:
        normalized = dict(data)
        if "author" not in normalized and "user_id" in normalized:
            normalized["author"] = normalized.pop("user_id")
        return cls.model_validate(normalized)
