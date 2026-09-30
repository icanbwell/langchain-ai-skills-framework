"""Tests for SkillSummary.required_external_servers and SkillSummary.manifest."""

from __future__ import annotations

from langchain_ai_skills_framework.models.skills_model import ManifestFileEntry, SkillSummary


def _make_summary(**overrides: object) -> SkillSummary:
    defaults: dict[str, object] = {
        "name": "test_skill",
        "description": "A test skill",
    }
    defaults.update(overrides)
    return SkillSummary(**defaults)  # type: ignore[arg-type]


class TestSkillSummaryRequiredExternalServers:
    """Backward-compatibility and explicit-value coverage for the new field."""

    def test_defaults_to_empty_tuple(self) -> None:
        """A summary constructed the way every existing caller already does
        (no `required_external_servers` kwarg) must keep working unchanged."""
        summary = _make_summary()

        assert summary.required_external_servers == ()

    def test_explicit_value(self) -> None:
        summary = _make_summary(required_external_servers=("partner-server", "another-server"))

        assert summary.required_external_servers == ("partner-server", "another-server")

    def test_other_fields_unaffected(self) -> None:
        summary = _make_summary(
            plugin_name="test-plugin",
            allowed_tools=("search_tool",),
            required_external_servers=("partner-server",),
        )

        assert summary.name == "test_skill"
        assert summary.plugin_name == "test-plugin"
        assert summary.allowed_tools == ("search_tool",)
        assert summary.required_external_servers == ("partner-server",)


class TestSkillSummaryManifest:
    """Backward-compatibility and explicit-value coverage for the manifest field."""

    def test_defaults_to_none(self) -> None:
        summary = _make_summary()

        assert summary.manifest is None

    def test_explicit_manifest_tuple(self) -> None:
        entries = (
            ManifestFileEntry(path="SKILL.md", digest="sha256:" + "a" * 64, size=42),
            ManifestFileEntry(path="references/checklist.md", digest="sha256:" + "b" * 64, size=10),
        )

        summary = _make_summary(manifest=entries)

        assert summary.manifest == entries

    def test_dynamic_literal(self) -> None:
        summary = _make_summary(manifest="dynamic")

        assert summary.manifest == "dynamic"
