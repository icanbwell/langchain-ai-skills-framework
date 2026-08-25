"""Pydantic model for document change history records."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryRecord(BaseModel):
    """A single history entry representing a document state change."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["created", "updated", "deleted", "published", "unpublished", "state_changed"] = Field(
        description="Type of mutation that produced this record"
    )
    document_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Full document state (after for create/update, before for delete)",
    )
    changed_by: str = Field(default="", description="User who triggered the change")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the change occurred (UTC)",
    )
    source_collection: str = Field(
        description="Origin collection: plugin_skills, plugin_scripts, plugin_references, or plugins"
    )
    user_id: str = Field(description="Owner of the document")
    plugin_name: str = Field(description="Plugin the document belongs to")
    skill_name: str = Field(default="", description="Skill name (empty for plugin-level history)")
    resource_name: str | None = Field(default=None, description="Resource name (for reference history)")
    script_name: str | None = Field(default=None, description="Script name (for script history)")

    def to_mongo_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mongo_dict(cls, data: Mapping[str, Any]) -> HistoryRecord:
        return cls.model_validate(data)
