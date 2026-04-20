"""Tests for snapshot serializer round-trip fidelity."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.snapshot_serializer import (
    deserialize_snapshot,
    serialize_snapshot,
)


def _make_summary(name: str = "test_skill") -> SkillSummary:
    return SkillSummary(
        name=name,
        description="A test skill",
        source_path=Path("/plugins/test-plugin/skills/test_skill/SKILL.md"),
        license="MIT",
        compatibility="1.0",
        metadata={"source": "marketplace", "version": 2},
        allowed_tools=("search_tool", "call_tool"),
    )


def _make_details(name: str = "test_skill") -> SkillDetails:
    summary = _make_summary(name)
    return SkillDetails(
        summary=summary,
        content="# Test Skill\n\nDo something useful.",
        source_path=summary.source_path,
    )


def _make_mcp_entry() -> PluginMcpServerEntry:
    return PluginMcpServerEntry(
        server_key="my-server",
        plugin_name="test-plugin",
        plugin_root=Path("/plugins/test-plugin"),
        url="http://localhost:8080/mcp",
        command="/plugins/test-plugin/bin/run",
        args=("--port", "8080"),
        env={"HOME": "/plugins/test-plugin"},
        headers={"X-Custom": "value"},
        description="Test MCP server",
        display_name="My Server",
        auth="oauth2",
    )


def _make_snapshot() -> SkillSnapshot:
    details = _make_details()
    return SkillSnapshot(
        details_by_name=MappingProxyType({"test_skill": details}),
        ordered_summaries=(details.summary,),
        mcp_servers=(_make_mcp_entry(),),
    )


class TestSnapshotSerializer:
    """Round-trip serialization tests."""

    def test_round_trip_full_snapshot(self) -> None:
        original = _make_snapshot()
        serialized = serialize_snapshot(original)
        restored = deserialize_snapshot(serialized)

        assert restored.ordered_summaries[0].name == original.ordered_summaries[0].name
        assert restored.ordered_summaries[0].description == original.ordered_summaries[0].description
        assert restored.ordered_summaries[0].source_path == original.ordered_summaries[0].source_path
        assert restored.ordered_summaries[0].license == original.ordered_summaries[0].license
        assert restored.ordered_summaries[0].allowed_tools == original.ordered_summaries[0].allowed_tools
        assert restored.ordered_summaries[0].metadata == original.ordered_summaries[0].metadata

    def test_round_trip_details(self) -> None:
        original = _make_snapshot()
        serialized = serialize_snapshot(original)
        restored = deserialize_snapshot(serialized)

        assert "test_skill" in restored.details_by_name
        detail = restored.details_by_name["test_skill"]
        assert detail.content == "# Test Skill\n\nDo something useful."
        assert detail.source_path == Path("/plugins/test-plugin/skills/test_skill/SKILL.md")

    def test_round_trip_mcp_servers(self) -> None:
        original = _make_snapshot()
        serialized = serialize_snapshot(original)
        restored = deserialize_snapshot(serialized)

        assert len(restored.mcp_servers) == 1
        entry = restored.mcp_servers[0]
        assert entry.server_key == "my-server"
        assert entry.plugin_name == "test-plugin"
        assert entry.plugin_root == Path("/plugins/test-plugin")
        assert entry.url == "http://localhost:8080/mcp"
        assert entry.command == "/plugins/test-plugin/bin/run"
        assert entry.args == ("--port", "8080")
        assert entry.env == {"HOME": "/plugins/test-plugin"}
        assert entry.headers == {"X-Custom": "value"}
        assert entry.description == "Test MCP server"
        assert entry.display_name == "My Server"
        assert entry.auth == "oauth2"
        assert entry.namespaced_key == "test-plugin__my-server"
        assert entry.is_http is True

    def test_serialized_is_json_compatible(self) -> None:
        """Serialized output should contain only JSON-native types."""
        import json

        original = _make_snapshot()
        serialized = serialize_snapshot(original)
        # Should not raise
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)

    def test_empty_snapshot(self) -> None:
        empty = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(empty)
        restored = deserialize_snapshot(serialized)

        assert len(restored.details_by_name) == 0
        assert len(restored.ordered_summaries) == 0
        assert len(restored.mcp_servers) == 0

    def test_mcp_entry_without_optional_fields(self) -> None:
        entry = PluginMcpServerEntry(
            server_key="minimal",
            plugin_name="plugin",
            plugin_root=Path("/tmp"),
            url=None,
            command=None,
        )
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(entry,),
        )
        serialized = serialize_snapshot(snapshot)
        restored = deserialize_snapshot(serialized)

        assert restored.mcp_servers[0].url is None
        assert restored.mcp_servers[0].command is None
        assert restored.mcp_servers[0].is_http is False
