"""Tests for MongoPluginSkillLoader's digest/size computation at write time."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader


def _expected_digest_and_size(content: str) -> tuple[str, int]:
    encoded = content.encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(encoded)


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
