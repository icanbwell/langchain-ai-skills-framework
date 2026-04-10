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


def _make_raw_doc(
    user_id: str = "user-1",
    skill_name: str = "my-skill",
    description: str = "A test skill",
    content: str = "# My Skill\nDo the thing.",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "skill_name": skill_name,
        "description": description,
        "content": content,
        "created_at": now,
        "updated_at": now,
    }


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
        loader = MongoUserSkillLoader(collection=collection)

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
        loader = MongoUserSkillLoader(collection=collection)

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
        collection = _make_collection()
        loader = MongoUserSkillLoader(collection=collection)

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await loader.save_skill(
                user_id=user_id, skill_name="test", content="content"
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        collection = _make_collection()
        loader = MongoUserSkillLoader(collection=collection)

        with pytest.raises(ValueError, match="skill_name must be a non-empty string"):
            await loader.save_skill(user_id="user-1", skill_name="", content="content")


class TestDeleteSkill:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self) -> None:
        collection = _make_collection()
        collection.delete_one.return_value = MagicMock(deleted_count=1)
        loader = MongoUserSkillLoader(collection=collection)

        result = await loader.delete_skill(user_id="user-1", skill_name="my-skill")

        assert result is True
        collection.delete_one.assert_awaited_once_with(
            {"user_id": "user-1", "skill_name": "my-skill"}
        )

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self) -> None:
        collection = _make_collection()
        collection.delete_one.return_value = MagicMock(deleted_count=0)
        loader = MongoUserSkillLoader(collection=collection)

        result = await loader.delete_skill(user_id="user-1", skill_name="nope")

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
        loader = MongoUserSkillLoader(collection=collection)

        snapshot = await loader.load_snapshot(user_id="user-1")

        assert len(snapshot.ordered_summaries) == 2
        names = [s.name for s in snapshot.ordered_summaries]
        assert names == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_returns_empty_snapshot_for_no_skills(self) -> None:
        collection = _make_collection()
        collection.find.return_value = AsyncIterator([])
        loader = MongoUserSkillLoader(collection=collection)

        snapshot = await loader.load_snapshot(user_id="user-1")

        assert len(snapshot.ordered_summaries) == 0
        assert len(snapshot.details_by_name) == 0

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        collection = _make_collection()
        loader = MongoUserSkillLoader(collection=collection)

        with pytest.raises(ValueError, match="user_id must be a non-empty string"):
            await loader.load_snapshot(user_id="")


class TestGetSkillDetails:
    @pytest.mark.asyncio
    async def test_returns_details_for_existing_skill(self) -> None:
        raw = _make_raw_doc(skill_name="my-skill", content="The content")
        collection = _make_collection()
        collection.find_one.return_value = raw
        loader = MongoUserSkillLoader(collection=collection)

        detail = await loader.get_skill_details(user_id="user-1", skill_name="my-skill")

        assert detail.name == "my-skill"
        assert detail.content == "The content"

    @pytest.mark.asyncio
    async def test_raises_not_found_for_missing_skill(self) -> None:
        collection = _make_collection()
        collection.find_one.return_value = None
        loader = MongoUserSkillLoader(collection=collection)

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
