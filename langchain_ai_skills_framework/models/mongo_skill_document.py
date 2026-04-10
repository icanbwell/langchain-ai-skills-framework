from datetime import datetime, timezone
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field


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
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was first saved",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the skill was last updated",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        """Return a dict suitable for MongoDB insertion."""
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> "MongoSkillDocument":
        """Construct from a MongoDB document, tolerating missing timestamps."""
        return cls(
            user_id=data["user_id"],
            skill_name=data["skill_name"],
            description=data["description"],
            content=data["content"],
            shared=data.get("shared", False),
            created_at=data.get("created_at", datetime.now(timezone.utc)),
            updated_at=data.get("updated_at", datetime.now(timezone.utc)),
        )
