"""SkillSummary.frontmatter: the raw SKILL.md frontmatter mapping, verbatim (BAI-1905)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginSkillDocument
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.snapshot_serializer import deserialize_snapshot, serialize_snapshot

IDENTITY: dict[str, Any] = {"author": "user-1", "plugin_name": "my-plugin", "skill_name": "my-skill"}


class _AsyncIter:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


async def _snapshot_summary_for(database: MagicMock, content: str) -> SkillSummary:
    skill_doc = MongoPluginSkillDocument(**IDENTITY, content=content, path="p").to_mongo_dict()
    database["plugin_skills"].find = MagicMock(return_value=_AsyncIter([skill_doc]))
    database["plugin_references"].find = MagicMock(return_value=_AsyncIter([]))
    database["plugin_scripts"].find = MagicMock(return_value=_AsyncIter([]))
    snapshot = await MongoPluginSkillLoader(database=database).load_snapshot(author="user-1")
    return snapshot.details_by_name["my-skill"].summary


async def test_frontmatter_keeps_original_keys_unknown_keys_and_value_types(mock_mongo_database: MagicMock) -> None:
    content = (
        "---\n"
        "name: my-skill\n"
        "description: Does a thing\n"
        "allowed-tools: Read Grep\n"
        "x-extra:\n"
        "  nested: [1, 2]\n"
        "  flag: true\n"
        "---\n"
        "# Body\n"
    )

    summary = await _snapshot_summary_for(mock_mongo_database, content)

    assert summary.frontmatter == {
        "name": "my-skill",
        "description": "Does a thing",
        "allowed-tools": "Read Grep",
        "x-extra": {"nested": [1, 2], "flag": True},
    }


@pytest.mark.parametrize(
    "content",
    [
        "# Just a body, no frontmatter\n",
        "---\nname: [unclosed\n  bad: : yaml\n---\nbody\n",
        "---\n- a\n- b\n---\nbody\n",
        "---\njust a scalar\n---\nbody\n",
        "",
    ],
    ids=["no-frontmatter", "malformed-yaml", "list-not-mapping", "scalar-not-mapping", "empty-content"],
)
async def test_frontmatter_is_none_when_absent_or_unparseable(mock_mongo_database: MagicMock, content: str) -> None:
    summary = await _snapshot_summary_for(mock_mongo_database, content)

    assert summary.frontmatter is None


def test_frontmatter_defaults_to_none() -> None:
    assert SkillSummary(name="s", description="d").frontmatter is None


def _snapshot_with(frontmatter: dict[str, object] | None) -> SkillSnapshot:
    summary = SkillSummary(name="s", description="d", frontmatter=frontmatter)
    return SkillSnapshot(
        details_by_name={"s": SkillDetails(summary=summary, content="body")}, ordered_summaries=(summary,)
    )


def test_frontmatter_round_trips_through_snapshot_serializer() -> None:
    frontmatter: dict[str, object] = {"allowed-tools": "Read Grep", "x-extra": {"n": [1, 2]}}

    restored = deserialize_snapshot(data=serialize_snapshot(snapshot=_snapshot_with(frontmatter)))

    assert restored.ordered_summaries[0].frontmatter == frontmatter


def test_none_frontmatter_round_trips_and_old_payloads_deserialize_as_none() -> None:
    serialized = serialize_snapshot(snapshot=_snapshot_with(None))
    assert deserialize_snapshot(data=serialized).ordered_summaries[0].frontmatter is None

    for entry in serialized["ordered_summaries"]:
        entry.pop("frontmatter")
    assert deserialize_snapshot(data=serialized).ordered_summaries[0].frontmatter is None
