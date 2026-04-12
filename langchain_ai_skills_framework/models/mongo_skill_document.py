from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


class MongoSkillResourceDocument(BaseModel):
    """Represents a resource file belonging to a skill stored in MongoDB."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="ID of the user who owns this resource")
    skill_name: str = Field(description="Normalized name of the parent skill")
    resource_name: str = Field(description="Name of the resource file")
    content: str = Field(description="Content of the resource file")
    modified_by: str = Field(
        default="",
        description="ID of the user who last modified this resource",
    )
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the resource was first saved",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the resource was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        """Return a dict suitable for MongoDB insertion."""
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> "MongoSkillResourceDocument":
        """Construct from a MongoDB document, tolerating missing fields."""
        return cls(
            user_id=data["user_id"],
            skill_name=data["skill_name"],
            resource_name=data["resource_name"],
            content=data["content"],
            modified_by=data.get("modified_by", ""),
            date_created=data.get("date_created", datetime.now(timezone.utc)),
            date_modified=data.get("date_modified", datetime.now(timezone.utc)),
        )


class MongoSkillScriptDocument(BaseModel):
    """Represents an executable script belonging to a skill stored in MongoDB."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="ID of the user who owns this script")
    skill_name: str = Field(description="Normalized name of the parent skill")
    script_name: str = Field(description="Name of the script file")
    content: str = Field(description="Content of the script file")
    modified_by: str = Field(
        default="",
        description="ID of the user who last modified this script",
    )
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the script was first saved",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the script was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        """Return a dict suitable for MongoDB insertion."""
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> "MongoSkillScriptDocument":
        """Construct from a MongoDB document, tolerating missing fields."""
        return cls(
            user_id=data["user_id"],
            skill_name=data["skill_name"],
            script_name=data["script_name"],
            content=data["content"],
            modified_by=data.get("modified_by", ""),
            date_created=data.get("date_created", datetime.now(timezone.utc)),
            date_modified=data.get("date_modified", datetime.now(timezone.utc)),
        )


class MongoSkillDocument(BaseModel):
    """Represents a user-owned skill stored in MongoDB."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(description="ID of the user who owns this skill")
    skill_name: str = Field(description="Normalized name of the skill")
    description: str = Field(description="Short description of what the skill does")
    content: str = Field(description="Full skill content (SKILL.md body)")
    shared: bool = Field(
        default=False,
        description="When True, this skill is available to all users",
    )
    modified_by: str = Field(
        default="",
        description="ID of the user who last modified this skill",
    )
    date_created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was first saved",
    )
    date_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        """Return a dict suitable for MongoDB insertion."""
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> "MongoSkillDocument":
        """Construct from a MongoDB document, tolerating missing fields."""
        return cls(
            user_id=data["user_id"],
            skill_name=data["skill_name"],
            description=data["description"],
            content=data["content"],
            shared=data.get("shared", False),
            modified_by=data.get("modified_by", ""),
            date_created=data.get(
                "date_created", data.get("created_at", datetime.now(timezone.utc))
            ),
            date_modified=data.get(
                "date_modified", data.get("updated_at", datetime.now(timezone.utc))
            ),
        )
