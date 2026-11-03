"""``all_authors`` listing (BAI-1500)."""

from __future__ import annotations

from types import MappingProxyType
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.loaders.composite_skill_loader import CompositeSkillLoader
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import SkillLoaderProtocol
from langchain_ai_skills_framework.models.skills_model import (
    AllAuthorsSkillSnapshot,
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.services.list_skills_service import ListSkillsService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


def _summary(name: str, *, author: str | None = None, state: str = "published") -> SkillSummary:
    return SkillSummary(
        name=name,
        description="d",
        plugin_name="p",
        state=state,
        metadata={"user_id": author} if author else {},
    )


async def test_service_default_does_not_pass_all_authors() -> None:
    loader = AsyncMock(spec=SkillLoaderProtocol)
    loader.list_all_summaries.return_value = [_summary("a")]

    await ListSkillsService(skill_loader=loader).execute(user_id="u")

    loader.list_all_summaries.assert_awaited_once_with(user_id="u", allowed_skills=set(), include_staging=False)


async def test_service_all_authors_returns_per_author_entries_with_author_set() -> None:
    loader = AsyncMock(spec=SkillLoaderProtocol)
    loader.list_all_summaries.return_value = [
        _summary("same", author="author-a@example.com", state="draft"),
        _summary("same", author="author-b@example.com", state="draft"),
        _summary("sys", author="system"),
    ]

    result = await ListSkillsService(skill_loader=loader).execute(user_id="admin@example.com", all_authors=True)

    assert loader.list_all_summaries.await_args.kwargs["all_authors"] is True
    assert [(r.name, r.author) for r in result if r.name == "same"] == [
        ("same", "author-a@example.com"),
        ("same", "author-b@example.com"),
    ]
    assert next(r for r in result if r.name == "sys").author is None


def _composite(
    *, mine: list[SkillSummary], everyone: list[SkillSummary], marketplace: list[SkillSummary]
) -> tuple[CompositeSkillLoader, AsyncMock]:
    shared = AsyncMock(spec=SkillLoaderProtocol)
    shared.list_skill_summaries = lambda *, allowed_skills: marketplace
    shared.get_skill_details = lambda *, skill_name, plugin_name=None: SkillDetails(
        summary=next(m for m in marketplace if m.name == skill_name), content="c"
    )
    store = AsyncMock(spec=PluginSkillStore)
    store.load_all_snapshot = AsyncMock()
    details = {s.name: SkillDetails(summary=s, content="c") for s in mine}
    store.load_snapshot.return_value = SkillSnapshot(
        details_by_name=MappingProxyType(details), ordered_summaries=tuple(mine)
    )
    store.load_shared_snapshot.return_value = SkillSnapshot(details_by_name=MappingProxyType({}), ordered_summaries=())
    store.load_all_snapshot.return_value = AllAuthorsSkillSnapshot(
        details_by_key=MappingProxyType(
            {
                (str(s.metadata["user_id"]), s.plugin_name or "", s.name): SkillDetails(summary=s, content="c")
                for s in everyone
            }
        ),
        ordered_summaries=tuple(everyone),
    )
    return CompositeSkillLoader(shared_loader=shared, user_loader=store), store


async def test_composite_default_is_unchanged_and_ignores_other_authors() -> None:
    mine = [_summary("mine", author="me@example.com")]
    other = [_summary("theirs", author="other@example.com", state="draft")]
    composite, store = _composite(mine=mine, everyone=mine + other, marketplace=[_summary("market")])

    result = await composite.list_all_summaries(user_id="me@example.com", allowed_skills=set())

    assert [s.name for s in result] == ["market", "mine"]
    store.load_all_snapshot.assert_not_awaited()


async def test_composite_all_authors_returns_every_authors_skills_and_marketplace() -> None:
    everyone = [
        _summary("same", author="author-a@example.com", state="draft"),
        _summary("same", author="author-b@example.com", state="draft"),
    ]
    composite, store = _composite(mine=[], everyone=everyone, marketplace=[_summary("market"), _summary("same")])

    result = await composite.list_all_summaries(user_id="admin@example.com", allowed_skills=set(), all_authors=True)

    store.load_all_snapshot.assert_awaited_once_with(include_staging=False)
    assert sorted((s.name, str(s.metadata.get("user_id"))) for s in result) == [
        ("market", "None"),
        ("same", "author-a@example.com"),
        ("same", "author-b@example.com"),
    ]


async def test_composite_all_authors_raises_when_store_lacks_load_all_snapshot() -> None:
    """A PluginSkillStore written before BAI-1500 must fail clearly, not with AttributeError."""

    class LegacyStore:
        pass

    shared = MagicMock()
    composite = CompositeSkillLoader(shared_loader=shared, user_loader=LegacyStore())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="load_all_snapshot"):
        await composite.list_all_summaries(user_id="admin@example.com", allowed_skills=set(), all_authors=True)


@pytest.mark.parametrize("include_staging", [True, False])
async def test_composite_all_authors_passes_include_staging_through(include_staging: bool) -> None:
    composite, store = _composite(mine=[], everyone=[], marketplace=[])

    await composite.list_all_summaries(
        user_id="admin@example.com", allowed_skills=set(), include_staging=include_staging, all_authors=True
    )

    store.load_all_snapshot.assert_awaited_once_with(include_staging=include_staging)


@pytest.mark.parametrize("user_id", ["", "   ", None])
async def test_service_all_authors_without_user_id_raises(user_id: str | None) -> None:
    loader = AsyncMock(spec=SkillLoaderProtocol)

    with pytest.raises(SkillOperationError, match="user_id"):
        await ListSkillsService(skill_loader=loader).execute(user_id=cast(str, user_id), all_authors=True)

    loader.list_all_summaries.assert_not_awaited()
