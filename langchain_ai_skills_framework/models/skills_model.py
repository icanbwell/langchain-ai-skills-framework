from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry


@dataclass(frozen=True, slots=True)
class SkillSummary:
    """Lightweight metadata describing an Agent Skill."""

    name: str
    description: str
    plugin_name: str | None = None
    folder: str | None = None
    path: str = ""
    state: str = "published"
    source_path: Path | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    date_modified: datetime | None = None


@dataclass(frozen=True, slots=True)
class SkillDetails:
    """Full Agent Skill definition including resolved content."""

    summary: SkillSummary
    content: str
    source_path: Path | None = None

    @property
    def name(self) -> str:
        return self.summary.name

    @property
    def description(self) -> str:
        return self.summary.description


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    """Immutable, already-filtered view of skills used by public loader calls."""

    details_by_name: Mapping[str, SkillDetails]
    ordered_summaries: tuple[SkillSummary, ...]
    mcp_servers: tuple[PluginMcpServerEntry, ...] = ()
