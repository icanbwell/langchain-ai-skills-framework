"""Tests for MongoPluginSkillDocument.required_external_servers."""

from __future__ import annotations

from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginSkillDocument,
)


def _make_document(**overrides: object) -> MongoPluginSkillDocument:
    defaults: dict[str, object] = {
        "plugin_name": "test-plugin",
        "skill_name": "test_skill",
        "author": "system",
    }
    defaults.update(overrides)
    return MongoPluginSkillDocument(**defaults)  # type: ignore[arg-type]


class TestRequiredExternalServers:
    """Backward-compatibility and round-trip coverage for the new field."""

    def test_defaults_to_empty_tuple(self) -> None:
        doc = _make_document()

        assert doc.required_external_servers == ()

    def test_explicit_value_round_trips_through_mongo_dict(self) -> None:
        doc = _make_document(required_external_servers=("partner-server",))

        mongo_dict = doc.to_mongo_dict()
        assert mongo_dict["required_external_servers"] == ["partner-server"]

        restored = MongoPluginSkillDocument.from_mongo_dict(mongo_dict)
        assert restored.required_external_servers == ("partner-server",)

    def test_from_mongo_dict_without_key_defaults_to_empty_tuple(self) -> None:
        """A document persisted before this field existed (no key in the raw
        Mongo dict at all) must still load, defaulting to an empty tuple."""
        legacy_raw = {
            "plugin_name": "test-plugin",
            "skill_name": "test_skill",
            "author": "system",
        }

        restored = MongoPluginSkillDocument.from_mongo_dict(legacy_raw)

        assert restored.required_external_servers == ()

    def test_schema_version_bumped_for_this_field_addition(self) -> None:
        """Per this repo's convention, adding a stored field must bump
        SCHEMA_VERSION so existing cached documents are ignored on read and
        re-synced rather than partially matching the new shape."""
        assert MongoPluginSkillDocument.SCHEMA_VERSION == 4


class TestManifestFields:
    """Backward-compatibility coverage for digest/size/is_dynamic."""

    def test_skill_document_defaults(self) -> None:
        doc = _make_document()

        assert doc.digest is None
        assert doc.size is None
        assert doc.is_dynamic is False

    def test_skill_document_explicit_values_round_trip(self) -> None:
        doc = _make_document(digest="sha256:" + "a" * 64, size=42, is_dynamic=True)

        mongo_dict = doc.to_mongo_dict()
        restored = MongoPluginSkillDocument.from_mongo_dict(mongo_dict)

        assert restored.digest == "sha256:" + "a" * 64
        assert restored.size == 42
        assert restored.is_dynamic is True

    def test_from_mongo_dict_without_manifest_keys_defaults(self) -> None:
        """A document persisted before this field existed must still load."""
        legacy_raw = {"plugin_name": "test-plugin", "skill_name": "test_skill", "author": "system"}

        restored = MongoPluginSkillDocument.from_mongo_dict(legacy_raw)

        assert restored.digest is None
        assert restored.size is None
        assert restored.is_dynamic is False

    def test_schema_version_bumped_for_manifest_fields(self) -> None:
        assert MongoPluginSkillDocument.SCHEMA_VERSION == 4
