"""The marketplace (shared) loader's synchronous snapshot read can download; it must not run on the event loop."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from types import MappingProxyType
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.composite_skill_loader import CompositeSkillLoader
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.models.skills_model import (
    AllAuthorsSkillSnapshot,
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)


class _ThreadRecordingSharedLoader:
    """Records the thread of each blocking call, as the real loader's download would run on it."""

    def __init__(self) -> None:
        self.call_threads: list[int] = []
        self._summary = SkillSummary(name="market", description="d", plugin_name="p")

    def list_skill_summaries(self, *, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        self.call_threads.append(threading.get_ident())
        return [self._summary]

    def get_skill_details(self, *, skill_name: str, plugin_name: str | None = None) -> SkillDetails:
        self.call_threads.append(threading.get_ident())
        return SkillDetails(summary=self._summary, content="c")


def _composite(shared: _ThreadRecordingSharedLoader) -> CompositeSkillLoader:
    empty = SkillSnapshot(details_by_name=MappingProxyType({}), ordered_summaries=())
    store = AsyncMock(spec=PluginSkillStore)
    store.load_snapshot.return_value = empty
    store.load_shared_snapshot.return_value = empty
    store.load_all_snapshot = AsyncMock(
        return_value=AllAuthorsSkillSnapshot(details_by_key=MappingProxyType({}), ordered_summaries=())
    )
    return CompositeSkillLoader(
        shared_loader=shared,  # type: ignore[arg-type]
        user_loader=store,
    )


@pytest.mark.parametrize("all_authors", [False, True])
async def test_shared_loader_is_read_off_the_event_loop_thread(all_authors: bool) -> None:
    shared = _ThreadRecordingSharedLoader()
    composite = _composite(shared)

    result = await composite.list_all_summaries(user_id="u", allowed_skills=set(), all_authors=all_authors)

    assert [s.name for s in result] == ["market"]
    assert shared.call_threads
    assert threading.get_ident() not in shared.call_threads

