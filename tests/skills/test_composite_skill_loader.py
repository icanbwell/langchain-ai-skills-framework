from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.mongo_user_skill_loader import (
    MongoUserSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)


def _make_skill(
    name: str, *, content: str = "Skill content", source: str = "shared"
) -> SkillDetails:
    source_path = Path(f"/{source}/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path,
        metadata={"source": source},
    )
    return SkillDetails(summary=summary, content=content, source_path=source_path)


class _StubSharedLoader(SkillLoaderProtocol):
    def __init__(self, details: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details)

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return [d.summary for d in self._details.values()]

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        try:
            return self._details[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError(f"'{skill_name}' not found") from exc

    def refresh(self) -> None:
        pass

    async def get_instructions(self) -> str:
        return "<available_skills></available_skills>"

    def get_tools(self) -> list[BaseTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        raise NotImplementedError

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return []

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return []


def _make_user_loader_mock(
    user_skills: dict[str, SkillDetails] | None = None,
) -> MongoUserSkillLoader:
    """Create a mock MongoUserSkillLoader that returns a snapshot."""
    loader = AsyncMock(spec=MongoUserSkillLoader)
    skills = user_skills or {}
    snapshot = SkillSnapshot(
        details_by_name=MappingProxyType(skills),
        ordered_summaries=tuple(
            sorted([d.summary for d in skills.values()], key=lambda s: s.name)
        ),
    )
    loader.load_snapshot.return_value = snapshot
    loader.get_skill_details.side_effect = lambda *, user_id, skill_name: (
        skills[skill_name]
        if skill_name in skills
        else (_ for _ in ()).throw(SkillNotFoundError(f"'{skill_name}' not found"))
    )
    return loader


class TestCompositeSkillLoaderInit:
    def test_rejects_none_shared_loader(self) -> None:
        user_loader = _make_user_loader_mock()
        with pytest.raises(ValueError, match="shared_loader must not be None"):
            CompositeSkillLoader(shared_loader=None, user_loader=user_loader)  # type: ignore[arg-type]

    def test_rejects_none_user_loader(self) -> None:
        shared = _StubSharedLoader({})
        with pytest.raises(ValueError, match="user_loader must not be None"):
            CompositeSkillLoader(shared_loader=shared, user_loader=None)  # type: ignore[arg-type]


class TestListAllSummaries:
    @pytest.mark.asyncio
    async def test_merges_shared_and_user_skills(self) -> None:
        shared_skill = _make_skill("alpha", source="shared")
        user_skill = _make_skill("beta", source="mongodb")

        shared = _StubSharedLoader({"alpha": shared_skill})
        user_loader = _make_user_loader_mock({"beta": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        summaries = await composite.list_all_summaries(
            user_id="user-1", allowed_skills=set()
        )

        names = [s.name for s in summaries]
        assert "alpha" in names
        assert "beta" in names

    @pytest.mark.asyncio
    async def test_user_skill_overrides_shared_on_name_collision(self) -> None:
        shared_skill = _make_skill("alpha", content="shared version", source="shared")
        user_skill = _make_skill("alpha", content="user version", source="mongodb")

        shared = _StubSharedLoader({"alpha": shared_skill})
        user_loader = _make_user_loader_mock({"alpha": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        summaries = await composite.list_all_summaries(
            user_id="user-1", allowed_skills=set()
        )

        assert len(summaries) == 1
        assert summaries[0].name == "alpha"
        assert summaries[0].metadata.get("source") == "mongodb"


class TestGetSkillDetailsForUser:
    @pytest.mark.asyncio
    async def test_returns_user_skill_when_exists(self) -> None:
        user_skill = _make_skill("my-skill", content="user version", source="mongodb")
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock({"my-skill": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        detail = await composite.get_skill_details_for_user(
            user_id="user-1", skill_name="my-skill"
        )

        assert detail.content == "user version"

    @pytest.mark.asyncio
    async def test_falls_back_to_shared_when_user_skill_missing(self) -> None:
        shared_skill = _make_skill("shared-skill", content="shared version")
        shared = _StubSharedLoader({"shared-skill": shared_skill})
        user_loader = _make_user_loader_mock({})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        detail = await composite.get_skill_details_for_user(
            user_id="user-1", skill_name="shared-skill"
        )

        assert detail.content == "shared version"

    @pytest.mark.asyncio
    async def test_raises_not_found_when_neither_has_skill(self) -> None:
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock({})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        with pytest.raises(SkillNotFoundError):
            await composite.get_skill_details_for_user(
                user_id="user-1", skill_name="nonexistent"
            )


class TestGetTools:
    def test_includes_shared_tools_plus_save_and_delete(self) -> None:
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock()
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        tools = composite.get_tools()
        tool_names = [t.name for t in tools]

        assert "save_skill" in tool_names
        assert "delete_skill" in tool_names


class TestGetInstructionsForUser:
    @pytest.mark.asyncio
    async def test_includes_user_skills_in_instructions(self) -> None:
        shared_skill = _make_skill("alpha")
        user_skill = _make_skill("beta", source="mongodb")

        shared = _StubSharedLoader({"alpha": shared_skill})
        user_loader = _make_user_loader_mock({"beta": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        instructions = await composite.get_instructions_for_user(user_id="user-1")

        assert "<available_skills>" in instructions
        assert "alpha" in instructions
        assert "beta" in instructions
        assert "save_skill" in instructions
