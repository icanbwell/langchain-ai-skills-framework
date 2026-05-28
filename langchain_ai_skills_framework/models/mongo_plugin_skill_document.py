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
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Path builders — Materialized Paths pattern
# ---------------------------------------------------------------------------


def build_skill_path(*, plugin_name: str, skill_name: str) -> str:
    """Return the canonical path for a skill's ``SKILL.md``."""
    return f"{plugin_name}/skills/{skill_name}/SKILL.md"


def build_resource_path(*, plugin_name: str, skill_name: str, resource_name: str) -> str:
    """Return the canonical path for a skill resource file."""
    return f"{plugin_name}/skills/{skill_name}/{resource_name}"


def build_script_path(*, plugin_name: str, skill_name: str, script_name: str) -> str:
    """Return the canonical path for a skill script file."""
    return f"{plugin_name}/skills/{skill_name}/scripts/{script_name}"


# ---------------------------------------------------------------------------
# Skill document
# ---------------------------------------------------------------------------


class MongoPluginSkillDocument(BaseModel):
    """A skill stored in the ``plugin_skills`` collection."""

    model_config = ConfigDict(extra="forbid")

    plugin_name: str = Field(description="Plugin that owns this skill")
    skill_name: str = Field(description="Normalized name of the skill")
    path: str = Field(description="Materialized path: plugin/skills/name/SKILL.md")
    description: str = Field(description="Short description of what the skill does")
    content: str = Field(description="Full skill content (SKILL.md body)")
    allowed_tools: tuple[str, ...] = Field(
        default=(),
        description="Tool names this skill is allowed to use",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata from the skill frontmatter",
    )
    user_id: str = Field(description="'system' for marketplace-synced, actual user id for user-saved")
    published: bool = Field(
        default=False,
        description="When True, this skill is visible to all users",
    )
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
        # Convert tuple to list for MongoDB compatibility
        data["allowed_tools"] = list(self.allowed_tools)
        return data

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginSkillDocument:
        allowed_tools_raw = data.get("allowed_tools", ())
        if isinstance(allowed_tools_raw, list):
            allowed_tools = tuple(allowed_tools_raw)
        else:
            allowed_tools = tuple(allowed_tools_raw) if allowed_tools_raw else ()

        return cls(
            plugin_name=data["plugin_name"],
            skill_name=data["skill_name"],
            path=data.get("path", ""),
            description=data.get("description", ""),
            content=data.get("content", ""),
            allowed_tools=allowed_tools,
            metadata=data.get("metadata"),
            user_id=data["user_id"],
            published=data.get("published", data.get("shared", False)),
            published_date=data.get("published_date"),
            published_branch=data.get("published_branch"),
            modified_by=data.get("modified_by", ""),
            date_created=data.get("date_created", datetime.now(timezone.utc)),
            date_modified=data.get("date_modified", datetime.now(timezone.utc)),
        )


# ---------------------------------------------------------------------------
# Resource document
# ---------------------------------------------------------------------------


class MongoPluginResourceDocument(BaseModel):
    """A resource file stored in the ``plugin_references`` collection."""

    model_config = ConfigDict(extra="forbid")

    plugin_name: str = Field(description="Plugin that owns this resource")
    skill_name: str = Field(description="Normalized name of the parent skill")
    resource_name: str = Field(description="Name of the resource file")
    path: str = Field(description="Materialized path: plugin/skills/name/resource")
    content: str = Field(description="Content of the resource file")
    user_id: str = Field(description="'system' for marketplace-synced, actual user id for user-saved")
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
        return cls(
            plugin_name=data["plugin_name"],
            skill_name=data["skill_name"],
            resource_name=data["resource_name"],
            path=data.get("path", ""),
            content=data.get("content", ""),
            user_id=data["user_id"],
            modified_by=data.get("modified_by", ""),
            date_created=data.get("date_created", datetime.now(timezone.utc)),
            date_modified=data.get("date_modified", datetime.now(timezone.utc)),
        )


# ---------------------------------------------------------------------------
# Script document
# ---------------------------------------------------------------------------


class MongoPluginScriptDocument(BaseModel):
    """An executable script stored in the ``plugin_scripts`` collection."""

    model_config = ConfigDict(extra="forbid")

    plugin_name: str = Field(description="Plugin that owns this script")
    skill_name: str = Field(description="Normalized name of the parent skill")
    script_name: str = Field(description="Name of the script file")
    path: str = Field(description="Materialized path: plugin/skills/name/scripts/script")
    content: str = Field(description="Content of the script file")
    user_id: str = Field(description="'system' for marketplace-synced, actual user id for user-saved")
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
        return cls(
            plugin_name=data["plugin_name"],
            skill_name=data["skill_name"],
            script_name=data["script_name"],
            path=data.get("path", ""),
            content=data.get("content", ""),
            user_id=data["user_id"],
            modified_by=data.get("modified_by", ""),
            date_created=data.get("date_created", datetime.now(timezone.utc)),
            date_modified=data.get("date_modified", datetime.now(timezone.utc)),
        )


# ---------------------------------------------------------------------------
# Usage document (unchanged structure, kept for continuity)
# ---------------------------------------------------------------------------


class MongoPluginDefinitionDocument(BaseModel):
    """A plugin definition stored in the ``plugins`` collection.

    Captures the plugin manifest metadata, skill list, and MCP server
    configuration so the gateway has a queryable catalog of available plugins.
    """

    model_config = ConfigDict(extra="forbid")

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
        return cls(
            plugin_name=plugin_name,
            description=data.get("description", ""),
            skills=data.get("skills", []),
            mcp_servers=data.get("mcp_servers", []),
            date_created=data.get("date_created", datetime.now(timezone.utc)),
            date_modified=data.get("date_modified", datetime.now(timezone.utc)),
        )


class MongoPluginSkillUsageDocument(BaseModel):
    """A single usage event for a skill within a plugin."""

    model_config = ConfigDict(extra="forbid")

    plugin_name: str = Field(description="Plugin containing the skill")
    skill_name: str = Field(description="Name of the skill that was used")
    user_id: str = Field(description="ID of the user who used the skill")
    date_used: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was used",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> MongoPluginSkillUsageDocument:
        return cls(
            plugin_name=data["plugin_name"],
            skill_name=data["skill_name"],
            user_id=data["user_id"],
            date_used=data.get("date_used", datetime.now(timezone.utc)),
        )
