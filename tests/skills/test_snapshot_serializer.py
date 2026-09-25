"""Tests for snapshot serializer round-trip fidelity."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    ManifestFileEntry,
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.snapshot_serializer import (
    deserialize_plugin_definition,
    deserialize_snapshot,
    serialize_plugin_definition,
    serialize_snapshot,
)


def _make_summary(name: str = "test_skill") -> SkillSummary:
    return SkillSummary(
        name=name,
        description="A test skill",
        plugin_name="test-plugin",
        source_path=Path("/plugins/test-plugin/skills/test_skill/SKILL.md"),
        license="MIT",
        compatibility="1.0",
        metadata={"source": "marketplace", "version": 2},
        allowed_tools=("search_tool", "call_tool"),
        required_external_servers=("partner-server",),
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
        visibility="external",
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
        serialized = serialize_snapshot(snapshot=original)
        restored = deserialize_snapshot(data=serialized)

        assert restored.ordered_summaries[0].name == original.ordered_summaries[0].name
        assert restored.ordered_summaries[0].description == original.ordered_summaries[0].description
        assert restored.ordered_summaries[0].source_path == original.ordered_summaries[0].source_path
        assert restored.ordered_summaries[0].license == original.ordered_summaries[0].license
        assert restored.ordered_summaries[0].allowed_tools == original.ordered_summaries[0].allowed_tools
        assert (
            restored.ordered_summaries[0].required_external_servers
            == original.ordered_summaries[0].required_external_servers
        )
        assert restored.ordered_summaries[0].metadata == original.ordered_summaries[0].metadata

    def test_round_trip_details(self) -> None:
        original = _make_snapshot()
        serialized = serialize_snapshot(snapshot=original)
        restored = deserialize_snapshot(data=serialized)

        assert "test_skill" in restored.details_by_name
        detail = restored.details_by_name["test_skill"]
        assert detail.content == "# Test Skill\n\nDo something useful."
        assert detail.source_path == Path("/plugins/test-plugin/skills/test_skill/SKILL.md")

    def test_round_trip_mcp_servers(self) -> None:
        original = _make_snapshot()
        serialized = serialize_snapshot(snapshot=original)
        restored = deserialize_snapshot(data=serialized)

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
        assert entry.visibility == "external"
        assert entry.namespaced_key == "test-plugin__my-server"
        assert entry.is_http is True

    def test_serialized_is_json_compatible(self) -> None:
        """Serialized output should contain only JSON-native types."""
        import json

        original = _make_snapshot()
        serialized = serialize_snapshot(snapshot=original)
        # Should not raise
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)

    def test_empty_snapshot(self) -> None:
        empty = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=empty)
        restored = deserialize_snapshot(data=serialized)

        assert len(restored.details_by_name) == 0
        assert len(restored.ordered_summaries) == 0
        assert len(restored.mcp_servers) == 0

    def test_round_trip_plugin_definition(self) -> None:
        plugin = PluginDefinition(
            name="test-plugin",
            description="A test plugin",
            skills=(_make_summary("skill-a"), _make_summary("skill-b")),
            mcp_servers=(_make_mcp_entry(),),
        )
        serialized = serialize_plugin_definition(plugin=plugin)
        restored = deserialize_plugin_definition(data=serialized)

        assert restored.name == "test-plugin"
        assert restored.description == "A test plugin"
        assert len(restored.skills) == 2
        assert restored.skills[0].name == "skill-a"
        assert restored.skills[1].name == "skill-b"
        assert len(restored.mcp_servers) == 1
        assert restored.mcp_servers[0].server_key == "my-server"

    def test_round_trip_plugin_definition_minimal(self) -> None:
        plugin = PluginDefinition(name="bare-plugin")
        serialized = serialize_plugin_definition(plugin=plugin)
        restored = deserialize_plugin_definition(data=serialized)

        assert restored.name == "bare-plugin"
        assert restored.description is None
        assert restored.skills == ()
        assert restored.mcp_servers == ()

    def test_mcp_entry_without_optional_fields(self) -> None:
        entry = PluginMcpServerEntry(
            server_key="minimal",
            plugin_name="plugin",
            plugin_root=Path("/plugins/minimal"),
            url=None,
            command=None,
        )
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(entry,),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        restored = deserialize_snapshot(data=serialized)

        assert restored.mcp_servers[0].url is None
        assert restored.mcp_servers[0].command is None
        assert restored.mcp_servers[0].is_http is False

    def test_deserialize_summary_without_required_external_servers_key(self) -> None:
        """A snapshot cached before this field existed (no key at all, not just
        an empty list) must still deserialize, defaulting to an empty tuple."""
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        legacy_summary_dict = {
            "name": "legacy_skill",
            "description": "A skill serialized before required_external_servers existed",
        }
        serialized["ordered_summaries"] = [legacy_summary_dict]

        restored = deserialize_snapshot(data=serialized)

        assert restored.ordered_summaries[0].name == "legacy_skill"
        assert restored.ordered_summaries[0].required_external_servers == ()

    def test_round_trip_manifest_none(self) -> None:
        summary = _make_summary()
        assert summary.manifest is None
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(summary,),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        restored = deserialize_snapshot(data=serialized)

        assert restored.ordered_summaries[0].manifest is None

    def test_round_trip_manifest_entries(self) -> None:
        summary = SkillSummary(
            name="manifest_skill",
            description="A skill with a manifest",
            manifest=(
                ManifestFileEntry(path="SKILL.md", digest="sha256:abc", size=123),
                ManifestFileEntry(path="references/foo.md", digest="sha256:def", size=456),
            ),
        )
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(summary,),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        restored = deserialize_snapshot(data=serialized)

        assert restored.ordered_summaries[0].manifest == summary.manifest

    def test_round_trip_manifest_dynamic(self) -> None:
        summary = SkillSummary(
            name="dynamic_skill",
            description="A dynamic skill",
            manifest="dynamic",
        )
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(summary,),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        restored = deserialize_snapshot(data=serialized)

        assert restored.ordered_summaries[0].manifest == "dynamic"

    def test_deserialize_summary_without_manifest_key(self) -> None:
        """A snapshot serialized before `manifest` existed (no key at all) must
        still deserialize, defaulting to None rather than raising."""
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        legacy_summary_dict = {
            "name": "legacy_skill",
            "description": "A skill serialized before manifest existed",
        }
        serialized["ordered_summaries"] = [legacy_summary_dict]

        restored = deserialize_snapshot(data=serialized)

        assert restored.ordered_summaries[0].name == "legacy_skill"
        assert restored.ordered_summaries[0].manifest is None

    def test_deserialize_mcp_entry_without_visibility_key(self) -> None:
        """A cached MCP entry from before `visibility` existed must default to
        "internal", not fail or silently become "external"."""
        legacy_entry_dict = {
            "server_key": "legacy-server",
            "plugin_name": "test-plugin",
            "plugin_root": "/plugins/test-plugin",
        }
        snapshot = SkillSnapshot(
            details_by_name=MappingProxyType({}),
            ordered_summaries=(),
            mcp_servers=(),
        )
        serialized = serialize_snapshot(snapshot=snapshot)
        serialized["mcp_servers"] = [legacy_entry_dict]

        restored = deserialize_snapshot(data=serialized)

        assert restored.mcp_servers[0].server_key == "legacy-server"
        assert restored.mcp_servers[0].visibility == "internal"
