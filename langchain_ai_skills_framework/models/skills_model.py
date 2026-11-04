from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry


@dataclass(frozen=True, slots=True)
class ManifestFileEntry:
    """One file's content-integrity record within a skill's manifest (SEP-2640)."""

    path: str
    digest: str
    size: int


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
    required_external_servers: tuple[str, ...] = ()
    """``.mcp.json`` server keys (``PluginMcpServerEntry.server_key``) this skill
    needs even though they're marked ``visibility="external"``. Empty (the
    default) means the skill needs no external server — every existing skill's
    current, unchanged behavior."""
    manifest: tuple[ManifestFileEntry, ...] | Literal["dynamic"] | None = None
    """Per-file SHA-256 digest + size manifest, computed at write time by
    ``MongoPluginSkillLoader``. ``None`` means not yet computed (e.g. content
    written before this field existed and not yet resaved/resynced).
    ``"dynamic"`` means the skill opted out of manifest computation.
    Script entries' ``path`` uses the script's stored stem name with no file
    extension (e.g. ``scripts/run`` for a script saved from ``run.py``) — do
    not assume ``scripts/<name>`` maps 1:1 to an on-disk filename."""
    digest: str | None = None
    """``sha256:<hex>`` digest of the SKILL.md content only, computed at write time. It does not cover references or
    scripts (see ``manifest`` for those), and it is the same for any author or tenant with identical content, so it is
    not a tenant-scoped identifier. A pure function of the content, so it is identical across pods and changes
    whenever SKILL.md is edited. ``None`` for skills with no stored digest
    (marketplace directory skills, or content written before the digest existed)."""


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


@dataclass(frozen=True, slots=True)
class AllAuthorsSkillSnapshot:
    """Immutable view of every author's skills, keyed by ``(author, plugin_name, skill_name)``.

    ``SkillSnapshot.details_by_name`` is keyed by name alone, so same-named skills from
    different authors would collapse. This view keeps them apart.
    """

    details_by_key: Mapping[tuple[str, str, str], SkillDetails]
    ordered_summaries: tuple[SkillSummary, ...]


@dataclass(frozen=True, slots=True)
class AllAuthorsSummaryPage:
    """One page of every author's skill summaries, in stable ``(plugin_name, skill_name, author)`` order.

    Summaries carry no ``manifest`` (``None``): resource and script documents are not read to build a listing page.
    ``next_cursor`` is opaque; pass it back unchanged to fetch the following page. ``None`` means this is the last page.
    """

    summaries: tuple[SkillSummary, ...]
    next_cursor: str | None = None
