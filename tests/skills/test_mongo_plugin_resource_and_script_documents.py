"""Tests for MongoPluginResourceDocument/MongoPluginScriptDocument manifest fields."""

from __future__ import annotations

from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
)


def test_resource_document_digest_size_default_to_none() -> None:
    doc = MongoPluginResourceDocument(
        plugin_name="test-plugin", skill_name="test_skill", resource_name="data.json", author="system"
    )

    assert doc.digest is None
    assert doc.size is None


def test_script_document_digest_size_default_to_none() -> None:
    doc = MongoPluginScriptDocument(
        plugin_name="test-plugin", skill_name="test_skill", script_name="run.py", author="system"
    )

    assert doc.digest is None
    assert doc.size is None
