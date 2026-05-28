"""Serialization helpers for SkillSnapshot to/from dict.

Handles non-JSON-native types:
- ``Path`` ↔ ``str``
- ``tuple`` ↔ ``list``
- ``MappingProxyType`` ↔ ``dict``
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)


def serialize_snapshot(*, snapshot: SkillSnapshot) -> dict[str, Any]:
    """Convert a SkillSnapshot to a JSON-serializable dict."""
    return {
        "details_by_name": {
            name: _serialize_details(details=details) for name, details in snapshot.details_by_name.items()
        },
        "ordered_summaries": [_serialize_summary(summary=s) for s in snapshot.ordered_summaries],
        "mcp_servers": [_serialize_mcp_entry(entry=e) for e in snapshot.mcp_servers],
    }


def deserialize_snapshot(*, data: dict[str, Any]) -> SkillSnapshot:
    """Reconstruct a SkillSnapshot from a serialized dict."""
    details_by_name: dict[str, SkillDetails] = {
        name: _deserialize_details(data=d) for name, d in data.get("details_by_name", {}).items()
    }
    ordered_summaries = tuple(_deserialize_summary(data=s) for s in data.get("ordered_summaries", []))
    mcp_servers = tuple(_deserialize_mcp_entry(data=e) for e in data.get("mcp_servers", []))
    return SkillSnapshot(
        details_by_name=MappingProxyType(details_by_name),
        ordered_summaries=ordered_summaries,
        mcp_servers=mcp_servers,
    )


# --- Private helpers --------------------------------------------------------


def _serialize_summary(*, summary: SkillSummary) -> dict[str, Any]:
    return {
        "name": summary.name,
        "description": summary.description,
        "plugin_name": summary.plugin_name,
        "source_path": str(summary.source_path) if summary.source_path else None,
        "license": summary.license,
        "compatibility": summary.compatibility,
        "metadata": dict(summary.metadata),
        "allowed_tools": list(summary.allowed_tools),
    }


def _deserialize_summary(*, data: dict[str, Any]) -> SkillSummary:
    metadata_raw = data.get("metadata", {})
    metadata: Mapping[str, object] = metadata_raw if isinstance(metadata_raw, dict) else {}
    source_path_raw = data.get("source_path")
    return SkillSummary(
        name=data["name"],
        description=data["description"],
        plugin_name=data.get("plugin_name"),
        source_path=Path(source_path_raw) if source_path_raw else None,
        license=data.get("license"),
        compatibility=data.get("compatibility"),
        metadata=metadata,
        allowed_tools=tuple(data.get("allowed_tools", [])),
    )


def _serialize_details(*, details: SkillDetails) -> dict[str, Any]:
    return {
        "summary": _serialize_summary(summary=details.summary),
        "content": details.content,
        "source_path": str(details.source_path) if details.source_path else None,
    }


def _deserialize_details(*, data: dict[str, Any]) -> SkillDetails:
    source_path_raw = data.get("source_path")
    return SkillDetails(
        summary=_deserialize_summary(data=data["summary"]),
        content=data["content"],
        source_path=Path(source_path_raw) if source_path_raw else None,
    )


def _serialize_mcp_entry(*, entry: PluginMcpServerEntry) -> dict[str, Any]:
    return {
        "server_key": entry.server_key,
        "plugin_name": entry.plugin_name,
        "plugin_root": str(entry.plugin_root),
        "url": entry.url,
        "command": entry.command,
        "args": list(entry.args),
        "env": dict(entry.env),
        "headers": dict(entry.headers),
        "description": entry.description,
        "display_name": entry.display_name,
        "auth": entry.auth,
        "oauth": entry.oauth,
    }


def _deserialize_mcp_entry(*, data: dict[str, Any]) -> PluginMcpServerEntry:
    return PluginMcpServerEntry(
        server_key=data["server_key"],
        plugin_name=data["plugin_name"],
        plugin_root=Path(data["plugin_root"]),
        url=data.get("url"),
        command=data.get("command"),
        args=tuple(data.get("args", [])),
        env=data.get("env", {}),
        headers=data.get("headers", {}),
        description=data.get("description"),
        display_name=data.get("display_name"),
        auth=data.get("auth"),
        oauth=data.get("oauth"),
    )


# --- Plugin definition serialization ----------------------------------------


def serialize_plugin_definition(*, plugin: PluginDefinition) -> dict[str, Any]:
    """Convert a PluginDefinition to a JSON-serializable dict."""
    return {
        "name": plugin.name,
        "description": plugin.description,
        "skills": [_serialize_summary(summary=s) for s in plugin.skills],
        "mcp_servers": [_serialize_mcp_entry(entry=e) for e in plugin.mcp_servers],
    }


def deserialize_plugin_definition(*, data: dict[str, Any]) -> PluginDefinition:
    """Reconstruct a PluginDefinition from a serialized dict."""
    return PluginDefinition(
        name=data["name"],
        description=data.get("description"),
        skills=tuple(_deserialize_summary(data=s) for s in data.get("skills", [])),
        mcp_servers=tuple(_deserialize_mcp_entry(data=e) for e in data.get("mcp_servers", [])),
    )
