from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.loaders.user_skill_store import (
    UserSkillStore,
)
from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
)
from langchain_ai_skills_framework.tools.save_skill_tool import SaveSkillTool


def _make_loader_mock() -> UserSkillStore:
    loader = AsyncMock(spec=UserSkillStore)
    loader.save_skill.return_value = MongoSkillDocument(
        user_id="user-1",
        skill_name="test-skill",
        description="A test",
        content="# Test\nContent",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    return loader


class TestSaveSkillTool:
    @pytest.mark.asyncio
    async def test_saves_skill_successfully(self) -> None:
        loader = _make_loader_mock()
        tool = SaveSkillTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(
            skill_name="test-skill",
            content="# Test\nContent",
            user_id="user-1",
        )

        assert "saved successfully" in result
        loader.save_skill.assert_awaited_once_with(  # type: ignore[attr-defined]
            user_id="user-1",
            skill_name="test-skill",
            content="# Test\nContent",
        )

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(skill_name="test", content="content", user_id="")

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="skill_name must be a non-empty"):
            await tool._arun(skill_name="", content="content", user_id="user-1")

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="content must be a non-empty"):
            await tool._arun(skill_name="test", content="", user_id="user-1")

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = SaveSkillTool()

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(skill_name="test", content="content", user_id="user-1")

    def test_sync_run_raises(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(skill_name="test", content="content", user_id="user-1")

    def test_get_friendly_name(self) -> None:
        name = SaveSkillTool.get_friendly_name(tool_input={"skill_name": "my-skill"})
        assert name == "Save Skill: my-skill"

    def test_get_friendly_name_empty(self) -> None:
        name = SaveSkillTool.get_friendly_name(tool_input={})
        assert name == "Save Skill"
