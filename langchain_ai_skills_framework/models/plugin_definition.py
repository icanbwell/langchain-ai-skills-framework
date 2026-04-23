"""Data model for individual plugin entries stored in the plugins collection.

Each :class:`PluginDefinition` represents a single marketplace plugin with
its metadata, skill summaries, and MCP server configurations.  These are
persisted as individual documents (keyed by plugin name) in a dedicated
MongoDB collection so that the gateway has a queryable catalog of available
plugins.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import SkillSummary


@dataclass(frozen=True, slots=True)
class PluginDefinition:
    """A complete plugin definition for storage in the plugins collection."""

    name: str
    """Canonical plugin name (directory name / marketplace.json key)."""

    description: str | None = None
    """Optional human-readable description from the marketplace manifest."""

    skills: tuple[SkillSummary, ...] = ()
    """Skill summaries provided by this plugin."""

    mcp_servers: tuple[PluginMcpServerEntry, ...] = ()
    """MCP server entries declared in this plugin's ``.mcp.json``."""
