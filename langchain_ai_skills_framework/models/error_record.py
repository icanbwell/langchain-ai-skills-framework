"""Pydantic model for skill operation error records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


class ErrorRecord(BaseModel):
    """A single error entry for troubleshooting failed skill operations."""

    model_config = ConfigDict(extra="ignore")

    operation: Literal["save", "publish", "retrieve", "delete"] = Field(description="The operation that failed")
    error_type: str = Field(default="", description="Exception class name")
    error_message: str = Field(default="", description="Human-readable error description")
    traceback: str = Field(default="", description="Full traceback if available")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the error occurred (UTC)",
    )
    user_id: str = Field(default="", description="User who triggered the operation")
    plugin_name: str = Field(default="", description="Plugin involved in the operation")
    skill_name: str = Field(default="", description="Skill involved in the operation")
    resource_name: str | None = Field(default=None, description="Resource name if applicable")
    script_name: str | None = Field(default=None, description="Script name if applicable")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the failed operation",
    )

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> ErrorRecord:
        return cls.model_validate(data)
