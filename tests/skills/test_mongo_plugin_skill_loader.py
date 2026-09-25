"""Tests for MongoPluginSkillLoader's digest/size computation at write time."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.models.skills_model import ManifestFileEntry


def _expected_digest_and_size(content: str) -> tuple[str, int]:
    encoded = content.encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(encoded)


class _AsyncIter:
    """Minimal async iterable/iterator wrapping a plain list, for `async for` over a fake Motor cursor."""

    def __init__(self, items: list[dict[str, object]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> dict[str, object]:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def test_save_skill_computes_digest_and_size(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)
    content = "# Skill content\n\nSome body text."
    expected_digest, expected_size = _expected_digest_and_size(content)

    async def fake_find_one_and_update(
        filter_: dict[str, object], update: dict[str, dict[str, object]], **kwargs: object
    ) -> dict[str, object]:
        return {
            "author": "user-1",
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            **update["$set"],
            **update["$setOnInsert"],
        }

    mock_mongo_database["plugin_skills"].find_one_and_update = fake_find_one_and_update

    doc = await loader.save_skill(author="user-1", plugin_name="my-plugin", skill_name="my-skill", content=content)

    assert doc.digest == expected_digest
    assert doc.size == expected_size
    assert doc.is_dynamic is False


async def test_save_skill_is_dynamic_true(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)

    async def fake_find_one_and_update(
        filter_: dict[str, object], update: dict[str, dict[str, object]], **kwargs: object
    ) -> dict[str, object]:
        return {
            "author": "user-1",
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            **update["$set"],
            **update["$setOnInsert"],
        }

    mock_mongo_database["plugin_skills"].find_one_and_update = fake_find_one_and_update

    doc = await loader.save_skill(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", content="x", is_dynamic=True
    )

    assert doc.is_dynamic is True


async def test_save_resource_computes_digest_and_size(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)
    content = '{"key": "value"}'
    expected_digest, expected_size = _expected_digest_and_size(content)

    async def fake_find_one_and_update(
        filter_: dict[str, object], update: dict[str, dict[str, object]], **kwargs: object
    ) -> dict[str, object]:
        return {
            "author": "user-1",
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            "resource_name": "data.json",
            **update["$set"],
            **update["$setOnInsert"],
        }

    mock_mongo_database["plugin_references"].find_one_and_update = fake_find_one_and_update

    doc = await loader.save_resource(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", resource_name="data.json", content=content
    )

    assert doc.digest == expected_digest
    assert doc.size == expected_size


async def test_save_script_computes_digest_and_size(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)
    content = "print('hello')"
    expected_digest, expected_size = _expected_digest_and_size(content)

    async def fake_find_one_and_update(
        filter_: dict[str, object], update: dict[str, dict[str, object]], **kwargs: object
    ) -> dict[str, object]:
        return {
            "author": "user-1",
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            "script_name": "run.py",
            **update["$set"],
            **update["$setOnInsert"],
        }

    mock_mongo_database["plugin_scripts"].find_one_and_update = fake_find_one_and_update

    doc = await loader.save_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py", content=content
    )

    assert doc.digest == expected_digest
    assert doc.size == expected_size


def test_digest_and_size_counts_utf8_bytes_not_characters() -> None:
    """A non-ASCII character can be >1 byte in UTF-8; size must reflect that."""
    digest, size = MongoPluginSkillLoader._digest_and_size("café")

    assert size == len("café".encode())
    assert size != len("café")
    assert digest == f"sha256:{hashlib.sha256('café'.encode()).hexdigest()}"


class TestBuildManifest:
    """Unit tests for the pure manifest-assembly helper (no Mongo I/O)."""

    def _skill_doc(self, **overrides: object) -> MongoPluginSkillDocument:
        defaults: dict[str, object] = {
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            "author": "system",
            "digest": "sha256:" + "a" * 64,
            "size": 10,
        }
        defaults.update(overrides)
        return MongoPluginSkillDocument(**defaults)  # type: ignore[arg-type]

    def test_dynamic_skill_short_circuits(self) -> None:
        doc = self._skill_doc(is_dynamic=True)

        manifest = MongoPluginSkillLoader._build_manifest(doc=doc, resources=[], scripts=[])

        assert manifest == "dynamic"

    def test_missing_digest_returns_none(self) -> None:
        doc = self._skill_doc(digest=None, size=None)

        manifest = MongoPluginSkillLoader._build_manifest(doc=doc, resources=[], scripts=[])

        assert manifest is None

    def test_skill_with_no_resources_or_scripts(self) -> None:
        doc = self._skill_doc()

        manifest = MongoPluginSkillLoader._build_manifest(doc=doc, resources=[], scripts=[])

        assert manifest == (ManifestFileEntry(path="SKILL.md", digest="sha256:" + "a" * 64, size=10),)

    def test_skill_with_resources_and_scripts_sorted_by_path(self) -> None:
        doc = self._skill_doc()
        resource = MongoPluginResourceDocument(
            plugin_name="my-plugin",
            skill_name="my-skill",
            resource_name="checklist.md",
            author="system",
            digest="sha256:" + "b" * 64,
            size=5,
        )
        script = MongoPluginScriptDocument(
            plugin_name="my-plugin",
            skill_name="my-skill",
            script_name="run.py",
            author="system",
            digest="sha256:" + "c" * 64,
            size=7,
        )

        manifest = MongoPluginSkillLoader._build_manifest(doc=doc, resources=[resource], scripts=[script])

        assert manifest == (
            ManifestFileEntry(path="SKILL.md", digest="sha256:" + "a" * 64, size=10),
            ManifestFileEntry(path="references/checklist.md", digest="sha256:" + "b" * 64, size=5),
            ManifestFileEntry(path="scripts/run.py", digest="sha256:" + "c" * 64, size=7),
        )

    def test_resource_missing_digest_returns_none(self) -> None:
        """One un-hashed file invalidates the whole manifest rather than emitting a partial one."""
        doc = self._skill_doc()
        resource = MongoPluginResourceDocument(
            plugin_name="my-plugin", skill_name="my-skill", resource_name="checklist.md", author="system"
        )

        manifest = MongoPluginSkillLoader._build_manifest(doc=doc, resources=[resource], scripts=[])

        assert manifest is None


async def test_get_skill_details_populates_manifest(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)
    mock_mongo_database["plugin_skills"].find_one = AsyncMock(
        return_value={
            "author": "user-1",
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            "content": "# Skill",
            "digest": "sha256:" + "a" * 64,
            "size": 8,
        }
    )
    mock_mongo_database["plugin_references"].find.return_value.to_list = AsyncMock(return_value=[])
    mock_mongo_database["plugin_scripts"].find.return_value.to_list = AsyncMock(return_value=[])

    details = await loader.get_skill_details(author="user-1", plugin_name="my-plugin", skill_name="my-skill")

    assert details.summary.manifest == (ManifestFileEntry(path="SKILL.md", digest="sha256:" + "a" * 64, size=8),)


async def test_build_snapshot_scopes_resources_by_plugin_and_skill_name(mock_mongo_database: MagicMock) -> None:
    """Two plugins with a same-named skill must not cross-contaminate manifests."""
    loader = MongoPluginSkillLoader(database=mock_mongo_database)

    skill_docs = [
        {
            "author": "system",
            "plugin_name": "plugin-a",
            "skill_name": "shared-name",
            "content": "A",
            "digest": "sha256:" + "a" * 64,
            "size": 1,
        },
        {
            "author": "system",
            "plugin_name": "plugin-b",
            "skill_name": "shared-name",
            "content": "B",
            "digest": "sha256:" + "b" * 64,
            "size": 1,
        },
    ]

    mock_mongo_database["plugin_skills"].find = MagicMock(return_value=_AsyncIter(skill_docs))

    resource_docs = [
        {
            "author": "system",
            "plugin_name": "plugin-a",
            "skill_name": "shared-name",
            "resource_name": "only-a.md",
            "content": "x",
            "digest": "sha256:" + "c" * 64,
            "size": 1,
        }
    ]
    mock_mongo_database["plugin_references"].find = MagicMock(return_value=_AsyncIter(resource_docs))
    mock_mongo_database["plugin_scripts"].find = MagicMock(return_value=_AsyncIter([]))

    snapshot = await loader.load_shared_snapshot()

    by_plugin = {s.plugin_name: s for s in snapshot.ordered_summaries}
    manifest_a = by_plugin["plugin-a"].manifest
    manifest_b = by_plugin["plugin-b"].manifest
    assert isinstance(manifest_a, tuple)
    assert any(entry.path == "references/only-a.md" for entry in manifest_a)
    assert isinstance(manifest_b, tuple)
    assert not any(entry.path == "references/only-a.md" for entry in manifest_b)
