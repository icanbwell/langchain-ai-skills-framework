from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.mongo_user_skill_loader import (
    MongoUserSkillLoader,
)


def _make_collection() -> AsyncMock:
    """Create a mock AsyncIOMotorCollection."""
    collection = AsyncMock()
    # Motor's find() returns a cursor synchronously (not a coroutine),
    # so it must be a regular MagicMock to support async iteration.
    collection.find = MagicMock()
    return collection


def _make_database(
    *,
    skills_collection: AsyncMock | None = None,
    resources_collection: AsyncMock | None = None,
    scripts_collection: AsyncMock | None = None,
    usage_collection: AsyncMock | None = None,
) -> MagicMock:
    """Create a mock AsyncIOMotorDatabase with configurable collections."""
    db = MagicMock()
    collections = {
        MongoUserSkillLoader.RESOURCES_COLLECTION_NAME: resources_collection or _make_collection(),
        MongoUserSkillLoader.SCRIPTS_COLLECTION_NAME: scripts_collection or _make_collection(),
        MongoUserSkillLoader.USAGE_COLLECTION_NAME: usage_collection or _make_collection(),
    }
    db.__getitem__ = MagicMock(side_effect=lambda name: collections.get(name, _make_collection()))
    return db


def _make_raw_doc(
    user_id: str = "user-1",
    skill_name: str = "my-skill",
    description: str = "A test skill",
    content: str = "# My Skill\nDo the thing.",
    modified_by: str = "user-1",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "skill_name": skill_name,
        "description": description,
        "content": content,
        "modified_by": modified_by,
        "date_created": now,
        "date_modified": now,
    }


def _make_raw_resource_doc(
    user_id: str = "user-1",
    skill_name: str = "my-skill",
    resource_name: str = "FORMS.md",
    content: str = "# Forms\nSome content",
    modified_by: str = "user-1",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "skill_name": skill_name,
        "resource_name": resource_name,
        "content": content,
        "modified_by": modified_by,
        "date_created": now,
        "date_modified": now,
    }


def _make_raw_script_doc(
    user_id: str = "user-1",
    skill_name: str = "my-skill",
    script_name: str = "analyze.py",
    content: str = "print('hello')",
    modified_by: str = "user-1",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "skill_name": skill_name,
        "script_name": script_name,
        "content": content,
        "modified_by": modified_by,
        "date_created": now,
        "date_modified": now,
    }


def _make_loader(
    *,
    collection: AsyncMock | None = None,
    resources_collection: AsyncMock | None = None,
    scripts_collection: AsyncMock | None = None,
    usage_collection: AsyncMock | None = None,
) -> MongoUserSkillLoader:
    """Create a MongoUserSkillLoader with mock collections."""
    coll = collection or _make_collection()
    db = _make_database(
        skills_collection=coll,
        resources_collection=resources_collection,
        scripts_collection=scripts_collection,
        usage_collection=usage_collection,
    )
    return MongoUserSkillLoader(collection=coll, database=db)


class TestMongoUserSkillLoaderInit:
    def test_rejects_none_collection(self) -> None:
        with pytest.raises(ValueError, match="collection must not be None"):
            MongoUserSkillLoader(collection=None)  # type: ignore[arg-type]


class TestSaveSkill:
    @pytest.mark.asyncio
    async def test_upserts_document(self) -> None:
        collection = _make_collection()
        collection.find_one_and_update.return_value = _make_raw_doc(
            user_id="user-1",
            skill_name="my-skill",
            description="Test skill",
            content="---\ndescription: Test skill\n---\n# Content",
        )
        loader = _make_loader(collection=collection)

        doc = await loader.save_skill(
            user_id="user-1",
            skill_name="My Skill",
            content="---\ndescription: Test skill\n---\n# Content",
        )

        assert doc.user_id == "user-1"
        assert doc.skill_name == "my-skill"
        assert doc.description == "Test skill"
        collection.find_one_and_update.assert_awaited_once()
        call_args = collection.find_one_and_update.call_args
        assert call_args[0][0] == {"user_id": "user-1", "skill_name": "my-skill"}
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_uses_date_created_and_date_modified(self) -> None:
        collection = _make_collection()
        collection.find_one_and_update.return_value = _make_raw_doc()
        loader = _make_loader(collection=collection)

        await loader.save_skill(user_id="user-1", skill_name="test", content="content")

        call_args = collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert "date_modified" in update_doc["$set"]
        assert "date_created" in update_doc["$setOnInsert"]

    @pytest.mark.asyncio
    async def test_stores_modified_by_defaults_to_user_id(self) -> None:
        collection = _make_collection()
        collection.find_one_and_update.return_value = _make_raw_doc()
        loader = _make_loader(collection=collection)

        await loader.save_skill(user_id="user-1", skill_name="test", content="content")

        call_args = collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert update_doc["$set"]["modified_by"] == "user-1"

    @pytest.mark.asyncio
    async def test_stores_explicit_modified_by(self) -> None:
        collection = _make_collection()
        collection.find_one_and_update.return_value = _make_raw_doc(modified_by="admin")
        loader = _make_loader(collection=collection)

        await loader.save_skill(
            user_id="user-1",
            skill_name="test",
            content="content",
            modified_by="admin",
        )

        call_args = collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert update_doc["$set"]["modified_by"] == "admin"

    @pytest.mark.asyncio
    async def test_extracts_description_from_first_line_when_no_frontmatter(
        self,
    ) -> None:
        collection = _make_collection()
        collection.find_one_and_update.return_value = _make_raw_doc(
            user_id="user-1",
            skill_name="test",
            description="Hello World",
            content="# Hello World\nBody here",
        )
        loader = _make_loader(collection=collection)

        doc = await loader.save_skill(
            user_id="user-1",
            skill_name="test",
            content="# Hello World\nBody here",
        )

        assert doc.description == "Hello World"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_id",
        ["", "   "],
    )
    async def test_rejects_empty_user_id(self, user_id: str) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await loader.save_skill(user_id=user_id, skill_name="test", content="content")

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="skill_name must be a non-empty string"):
            await loader.save_skill(user_id="user-1", skill_name="", content="content")


class TestDeleteSkill:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self) -> None:
        collection = _make_collection()
        collection.delete_one.return_value = MagicMock(deleted_count=1)
        resources_collection = _make_collection()
        scripts_collection = _make_collection()
        loader = _make_loader(
            collection=collection,
            resources_collection=resources_collection,
            scripts_collection=scripts_collection,
        )

        result = await loader.delete_skill(user_id="user-1", skill_name="my-skill")

        assert result is True
        collection.delete_one.assert_awaited_once_with({"user_id": "user-1", "skill_name": "my-skill"})
        # Also deletes associated resources and scripts
        resources_collection.delete_many.assert_awaited_once_with({"user_id": "user-1", "skill_name": "my-skill"})
        scripts_collection.delete_many.assert_awaited_once_with({"user_id": "user-1", "skill_name": "my-skill"})

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self) -> None:
        collection = _make_collection()
        collection.delete_one.return_value = MagicMock(deleted_count=0)
        loader = _make_loader(collection=collection)

        result = await loader.delete_skill(user_id="user-1", skill_name="nope")

        assert result is False


class TestSkillExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        collection = _make_collection()
        collection.count_documents.return_value = 1
        loader = _make_loader(collection=collection)

        result = await loader.skill_exists(user_id="user-1", skill_name="my-skill")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_missing(self) -> None:
        collection = _make_collection()
        collection.count_documents.return_value = 0
        loader = _make_loader(collection=collection)

        result = await loader.skill_exists(user_id="user-1", skill_name="nope")

        assert result is False


class TestLoadSnapshot:
    @pytest.mark.asyncio
    async def test_loads_all_user_skills(self) -> None:
        raw_docs = [
            _make_raw_doc(skill_name="alpha", description="First"),
            _make_raw_doc(skill_name="beta", description="Second"),
        ]
        collection = _make_collection()
        collection.find.return_value = AsyncIterator(raw_docs)
        loader = _make_loader(collection=collection)

        snapshot = await loader.load_snapshot(user_id="user-1")

        assert len(snapshot.ordered_summaries) == 2
        names = [s.name for s in snapshot.ordered_summaries]
        assert names == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_returns_empty_snapshot_for_no_skills(self) -> None:
        collection = _make_collection()
        collection.find.return_value = AsyncIterator([])
        loader = _make_loader(collection=collection)

        snapshot = await loader.load_snapshot(user_id="user-1")

        assert len(snapshot.ordered_summaries) == 0
        assert len(snapshot.details_by_name) == 0

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await loader.load_snapshot(user_id="")


class TestGetSkillDetails:
    @pytest.mark.asyncio
    async def test_returns_details_for_existing_skill(self) -> None:
        raw = _make_raw_doc(skill_name="my-skill", content="The content")
        collection = _make_collection()
        collection.find_one.return_value = raw
        loader = _make_loader(collection=collection)

        detail = await loader.get_skill_details(user_id="user-1", skill_name="my-skill")

        assert detail.name == "my-skill"
        assert detail.content == "The content"

    @pytest.mark.asyncio
    async def test_raises_not_found_for_missing_skill(self) -> None:
        collection = _make_collection()
        collection.find_one.return_value = None
        loader = _make_loader(collection=collection)

        with pytest.raises(SkillNotFoundError):
            await loader.get_skill_details(user_id="user-1", skill_name="nonexistent")


class TestNormalizeSkillName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("My Skill", "my-skill"),
            ("my_skill", "my-skill"),
            ("  --my---skill--  ", "my-skill"),
            ("UPPER_CASE", "upper-case"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert MongoUserSkillLoader._normalize_skill_name(raw) == expected


class TestExtractDescription:
    def test_from_frontmatter(self) -> None:
        content = "---\ndescription: My fancy skill\n---\n# Body"
        assert MongoUserSkillLoader._extract_description(content) == "My fancy skill"

    def test_from_first_line_when_no_frontmatter(self) -> None:
        content = "# Hello World\nSome body"
        assert MongoUserSkillLoader._extract_description(content) == "Hello World"

    def test_fallback_when_empty(self) -> None:
        assert MongoUserSkillLoader._extract_description("") == "User-saved skill"


# --- Resource tests ----------------------------------------------------------


class TestSaveResource:
    @pytest.mark.asyncio
    async def test_upserts_resource(self) -> None:
        resources_collection = _make_collection()
        resources_collection.find_one_and_update.return_value = _make_raw_resource_doc()
        loader = _make_loader(resources_collection=resources_collection)

        doc = await loader.save_resource(
            user_id="user-1",
            skill_name="my-skill",
            resource_name="FORMS.md",
            content="# Forms\nSome content",
        )

        assert doc.resource_name == "FORMS.md"
        assert doc.skill_name == "my-skill"
        resources_collection.find_one_and_update.assert_awaited_once()
        call_args = resources_collection.find_one_and_update.call_args
        assert call_args[0][0] == {
            "user_id": "user-1",
            "skill_name": "my-skill",
            "resource_name": "FORMS.md",
        }
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_stores_date_created_and_date_modified(self) -> None:
        resources_collection = _make_collection()
        resources_collection.find_one_and_update.return_value = _make_raw_resource_doc()
        loader = _make_loader(resources_collection=resources_collection)

        await loader.save_resource(
            user_id="user-1",
            skill_name="my-skill",
            resource_name="FORMS.md",
            content="content",
        )

        call_args = resources_collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert "date_modified" in update_doc["$set"]
        assert "date_created" in update_doc["$setOnInsert"]

    @pytest.mark.asyncio
    async def test_stores_modified_by(self) -> None:
        resources_collection = _make_collection()
        resources_collection.find_one_and_update.return_value = _make_raw_resource_doc(modified_by="system")
        loader = _make_loader(resources_collection=resources_collection)

        await loader.save_resource(
            user_id="user-1",
            skill_name="my-skill",
            resource_name="FORMS.md",
            content="content",
            modified_by="system",
        )

        call_args = resources_collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert update_doc["$set"]["modified_by"] == "system"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("user_id", ["", "   "])
    async def test_rejects_empty_user_id(self, user_id: str) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await loader.save_resource(
                user_id=user_id,
                skill_name="test",
                resource_name="r.md",
                content="x",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_resource_name(self) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="resource_name must be a non-empty string"):
            await loader.save_resource(
                user_id="user-1",
                skill_name="test",
                resource_name="",
                content="x",
            )


class TestReadResource:
    @pytest.mark.asyncio
    async def test_returns_content(self) -> None:
        resources_collection = _make_collection()
        resources_collection.find_one.return_value = _make_raw_resource_doc(content="Resource content here")
        loader = _make_loader(resources_collection=resources_collection)

        content = await loader.read_resource(user_id="user-1", skill_name="my-skill", resource_name="FORMS.md")

        assert content == "Resource content here"

    @pytest.mark.asyncio
    async def test_raises_not_found(self) -> None:
        resources_collection = _make_collection()
        resources_collection.find_one.return_value = None
        loader = _make_loader(resources_collection=resources_collection)

        with pytest.raises(SkillNotFoundError):
            await loader.read_resource(user_id="user-1", skill_name="my-skill", resource_name="nope.md")


class TestListResourceNames:
    @pytest.mark.asyncio
    async def test_returns_sorted_names(self) -> None:
        resources_collection = _make_collection()
        resources_collection.find.return_value = AsyncIterator(
            [
                {"resource_name": "ZEBRA.md"},
                {"resource_name": "ALPHA.md"},
            ]
        )
        loader = _make_loader(resources_collection=resources_collection)

        names = await loader.list_resource_names(user_id="user-1", skill_name="my-skill")

        assert list(names) == ["ALPHA.md", "ZEBRA.md"]


class TestDeleteResource:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self) -> None:
        resources_collection = _make_collection()
        resources_collection.delete_one.return_value = MagicMock(deleted_count=1)
        loader = _make_loader(resources_collection=resources_collection)

        result = await loader.delete_resource(user_id="user-1", skill_name="my-skill", resource_name="FORMS.md")

        assert result is True


class TestResourceExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        resources_collection = _make_collection()
        resources_collection.count_documents.return_value = 1
        loader = _make_loader(resources_collection=resources_collection)

        result = await loader.resource_exists(user_id="user-1", skill_name="my-skill", resource_name="FORMS.md")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_missing(self) -> None:
        resources_collection = _make_collection()
        resources_collection.count_documents.return_value = 0
        loader = _make_loader(resources_collection=resources_collection)

        result = await loader.resource_exists(user_id="user-1", skill_name="my-skill", resource_name="nope.md")

        assert result is False


# --- Script tests ------------------------------------------------------------


class TestSaveScript:
    @pytest.mark.asyncio
    async def test_upserts_script(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.find_one_and_update.return_value = _make_raw_script_doc()
        loader = _make_loader(scripts_collection=scripts_collection)

        doc = await loader.save_script(
            user_id="user-1",
            skill_name="my-skill",
            script_name="analyze.py",
            content="print('hello')",
        )

        assert doc.script_name == "analyze.py"
        assert doc.skill_name == "my-skill"
        scripts_collection.find_one_and_update.assert_awaited_once()
        call_args = scripts_collection.find_one_and_update.call_args
        assert call_args[0][0] == {
            "user_id": "user-1",
            "skill_name": "my-skill",
            "script_name": "analyze.py",
        }
        assert call_args[1]["upsert"] is True

    @pytest.mark.asyncio
    async def test_stores_date_created_and_date_modified(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.find_one_and_update.return_value = _make_raw_script_doc()
        loader = _make_loader(scripts_collection=scripts_collection)

        await loader.save_script(
            user_id="user-1",
            skill_name="my-skill",
            script_name="analyze.py",
            content="print('hello')",
        )

        call_args = scripts_collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert "date_modified" in update_doc["$set"]
        assert "date_created" in update_doc["$setOnInsert"]

    @pytest.mark.asyncio
    async def test_stores_modified_by(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.find_one_and_update.return_value = _make_raw_script_doc(modified_by="system")
        loader = _make_loader(scripts_collection=scripts_collection)

        await loader.save_script(
            user_id="user-1",
            skill_name="my-skill",
            script_name="analyze.py",
            content="print('hello')",
            modified_by="system",
        )

        call_args = scripts_collection.find_one_and_update.call_args
        update_doc = call_args[0][1]
        assert update_doc["$set"]["modified_by"] == "system"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("user_id", ["", "   "])
    async def test_rejects_empty_user_id(self, user_id: str) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await loader.save_script(
                user_id=user_id,
                skill_name="test",
                script_name="s.py",
                content="x",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_script_name(self) -> None:
        loader = _make_loader()

        with pytest.raises(ValueError, match="script_name must be a non-empty string"):
            await loader.save_script(
                user_id="user-1",
                skill_name="test",
                script_name="",
                content="x",
            )


class TestReadScript:
    @pytest.mark.asyncio
    async def test_returns_content(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.find_one.return_value = _make_raw_script_doc(content="import sys\nprint(sys.argv)")
        loader = _make_loader(scripts_collection=scripts_collection)

        content = await loader.read_script(user_id="user-1", skill_name="my-skill", script_name="analyze.py")

        assert content == "import sys\nprint(sys.argv)"

    @pytest.mark.asyncio
    async def test_raises_not_found(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.find_one.return_value = None
        loader = _make_loader(scripts_collection=scripts_collection)

        with pytest.raises(SkillNotFoundError):
            await loader.read_script(user_id="user-1", skill_name="my-skill", script_name="nope.py")


class TestListScriptNames:
    @pytest.mark.asyncio
    async def test_returns_sorted_names(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.find.return_value = AsyncIterator(
            [
                {"script_name": "z_script.py"},
                {"script_name": "a_script.py"},
            ]
        )
        loader = _make_loader(scripts_collection=scripts_collection)

        names = await loader.list_script_names(user_id="user-1", skill_name="my-skill")

        assert list(names) == ["a_script.py", "z_script.py"]


class TestDeleteScript:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.delete_one.return_value = MagicMock(deleted_count=1)
        loader = _make_loader(scripts_collection=scripts_collection)

        result = await loader.delete_script(user_id="user-1", skill_name="my-skill", script_name="analyze.py")

        assert result is True


class TestScriptExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_exists(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.count_documents.return_value = 1
        loader = _make_loader(scripts_collection=scripts_collection)

        result = await loader.script_exists(user_id="user-1", skill_name="my-skill", script_name="analyze.py")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_missing(self) -> None:
        scripts_collection = _make_collection()
        scripts_collection.count_documents.return_value = 0
        loader = _make_loader(scripts_collection=scripts_collection)

        result = await loader.script_exists(user_id="user-1", skill_name="my-skill", script_name="nope.py")

        assert result is False


# --- Usage tracking tests ----------------------------------------------------


class TestRecordSkillUsage:
    @pytest.mark.asyncio
    async def test_inserts_usage_document(self) -> None:
        usage_collection = _make_collection()
        loader = _make_loader(usage_collection=usage_collection)

        doc = await loader.record_skill_usage(skill_name="my-skill", user_id="user-1")

        assert doc.skill_name == "my-skill"
        assert doc.user_id == "user-1"
        usage_collection.insert_one.assert_awaited_once()


class TestGetSkillUsageCount:
    @pytest.mark.asyncio
    async def test_returns_count(self) -> None:
        usage_collection = _make_collection()
        usage_collection.count_documents.return_value = 42
        loader = _make_loader(usage_collection=usage_collection)

        count = await loader.get_skill_usage_count(skill_name="my-skill")

        assert count == 42
        usage_collection.count_documents.assert_awaited_once_with({"skill_name": "my-skill"})

    @pytest.mark.asyncio
    async def test_returns_zero_for_unused_skill(self) -> None:
        usage_collection = _make_collection()
        usage_collection.count_documents.return_value = 0
        loader = _make_loader(usage_collection=usage_collection)

        count = await loader.get_skill_usage_count(skill_name="never-used")

        assert count == 0


class AsyncIterator:
    """Helper to mock an async cursor."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self._index = 0

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item
