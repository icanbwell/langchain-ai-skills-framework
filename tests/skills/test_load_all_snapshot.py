"""Tests for MongoPluginSkillLoader.load_all_snapshot (BAI-1500)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.loaders.null_plugin_skill_store import NullPluginSkillStore


class _AsyncIter:
    def __init__(self, items: list[dict[str, object]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> dict[str, object]:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _skill(*, author: str, plugin: str, name: str, state: str = "published") -> dict[str, object]:
    return {
        "author": author,
        "plugin_name": plugin,
        "skill_name": name,
        "content": f"content of {author}",
        "state": state,
        "digest": "sha256:" + "a" * 64,
        "size": 1,
    }


def _loader(database: MagicMock, skills: list[dict[str, object]]) -> MongoPluginSkillLoader:
    database["plugin_skills"].find = MagicMock(return_value=_AsyncIter(skills))
    database["plugin_references"].find = MagicMock(return_value=_AsyncIter([]))
    database["plugin_scripts"].find = MagicMock(return_value=_AsyncIter([]))
    return MongoPluginSkillLoader(database=database)


async def test_returns_every_authors_skills_including_drafts(mock_mongo_database: MagicMock) -> None:
    loader = _loader(
        mock_mongo_database,
        [
            _skill(author="author-a@example.com", plugin="p1", name="one", state="draft"),
            _skill(author="author-b@example.com", plugin="p1", name="two", state="published"),
            _skill(author="system", plugin="p2", name="three", state="in_review"),
        ],
    )

    snapshot = await loader.load_all_snapshot()

    assert set(snapshot.details_by_key) == {
        ("author-a@example.com", "p1", "one"),
        ("author-b@example.com", "p1", "two"),
        ("system", "p2", "three"),
    }
    assert snapshot.details_by_key[("author-a@example.com", "p1", "one")].summary.state == "draft"
    assert len(snapshot.ordered_summaries) == 3


async def test_same_named_skill_from_two_authors_stays_two_entries(mock_mongo_database: MagicMock) -> None:
    loader = _loader(
        mock_mongo_database,
        [
            _skill(author="author-a@example.com", plugin="p1", name="same", state="draft"),
            _skill(author="author-b@example.com", plugin="p1", name="same", state="draft"),
        ],
    )

    snapshot = await loader.load_all_snapshot()

    assert len(snapshot.details_by_key) == 2
    assert len(snapshot.ordered_summaries) == 2
    owners = {s.metadata["user_id"] for s in snapshot.ordered_summaries}
    assert owners == {"author-a@example.com", "author-b@example.com"}
    contents = {d.content for d in snapshot.details_by_key.values()}
    assert contents == {"content of author-a@example.com", "content of author-b@example.com"}


@pytest.mark.parametrize(
    ("include_staging", "expected_state_filter"),
    [(False, {"$ne": "staging"}), (True, None)],
)
async def test_staging_flag_controls_state_filter(
    mock_mongo_database: MagicMock, include_staging: bool, expected_state_filter: dict[str, str] | None
) -> None:
    loader = _loader(mock_mongo_database, [])

    await loader.load_all_snapshot(include_staging=include_staging)

    query = mock_mongo_database["plugin_skills"].find.call_args.args[0]
    assert query.get("state") == expected_state_filter
    assert "author" not in query


async def test_schema_version_filter_is_applied(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, [])

    await loader.load_all_snapshot()

    query = mock_mongo_database["plugin_skills"].find.call_args.args[0]
    assert query[MongoPluginSkillLoader.SCHEMA_VERSION_FIELD] == loader._schema_version


async def test_null_store_returns_empty_snapshot() -> None:
    snapshot = await NullPluginSkillStore().load_all_snapshot()

    assert dict(snapshot.details_by_key) == {}
    assert snapshot.ordered_summaries == ()
