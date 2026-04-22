from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry


@dataclass(frozen=True, slots=True)
class SkillSummary:
    """Lightweight metadata describing an Agent Skill."""

    name: str
    description: str
    plugin_name: str = ""
    source_path: Path | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()


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
