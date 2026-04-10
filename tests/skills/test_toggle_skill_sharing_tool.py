from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.loaders.user_skill_store import (
    UserSkillStore,
)
from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
)
from langchain_ai_skills_framework.tools.toggle_skill_sharing_tool import (
    ToggleSkillSharingTool,
)


def _make_doc(
    skill_name: str = "test-skill", shared: bool = True
) -> MongoSkillDocument:
    return MongoSkillDocument(
        user_id="user-1",
        skill_name=skill_name,
        description="A test",
        content="# Test\nContent",
        shared=shared,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_loader_mock(shared: bool = True) -> UserSkillStore:
    loader = AsyncMock(spec=UserSkillStore)
    loader.set_skill_shared.return_value = _make_doc(shared=shared)
    return loader


def _make_runtime(user_id: str = "user-1") -> MagicMock:
    """Create a mock ToolRuntime with the given user_id in context."""
    runtime = MagicMock()
    runtime.context = {"user_id": user_id}
    return runtime


class TestToggleSkillSharingTool:
    @pytest.mark.asyncio
    async def test_shares_skill_successfully(self) -> None:
        loader = _make_loader_mock(shared=True)
        tool = ToggleSkillSharingTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(
            skill_name="test-skill", shared=True, runtime=_make_runtime("user-1")
        )

        assert "shared" in result
        loader.set_skill_shared.assert_awaited_once_with(  # type: ignore[attr-defined]
            user_id="user-1", skill_name="test-skill", shared=True
        )

    @pytest.mark.asyncio
    async def test_makes_skill_private(self) -> None:
        loader = _make_loader_mock(shared=False)
        tool = ToggleSkillSharingTool(mongo_skill_loader=loader)

        result, _ = await tool._arun(
            skill_name="test-skill", shared=False, runtime=_make_runtime("user-1")
        )

        assert "private" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        tool = ToggleSkillSharingTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(skill_name="test", shared=True, runtime=_make_runtime(""))

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        tool = ToggleSkillSharingTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="skill_name must be a non-empty"):
            await tool._arun(skill_name="  ", shared=True, runtime=_make_runtime())

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = ToggleSkillSharingTool()

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(skill_name="test", shared=True, runtime=_make_runtime())

    @pytest.mark.asyncio
    async def test_wraps_unexpected_exception(self) -> None:
        loader = AsyncMock(spec=UserSkillStore)
        loader.set_skill_shared.side_effect = RuntimeError("db down")
        tool = ToggleSkillSharingTool(mongo_skill_loader=loader)

        with pytest.raises(ToolException, match="Unable to update sharing"):
            await tool._arun(skill_name="test", shared=True, runtime=_make_runtime())

    def test_sync_run_raises(self) -> None:
        tool = ToggleSkillSharingTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(skill_name="test", shared=True)

    def test_get_friendly_name(self) -> None:
        name = ToggleSkillSharingTool.get_friendly_name(
            tool_input={"skill_name": "my-skill"}
        )
        assert name == "Toggle Skill Sharing: my-skill"

    def test_get_friendly_name_empty(self) -> None:
        name = ToggleSkillSharingTool.get_friendly_name(tool_input={})
        assert name == "Toggle Skill Sharing"
