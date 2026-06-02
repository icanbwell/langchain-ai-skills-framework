from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.langchain.tools.delete_skill_tool import DeleteSkillTool
from tests.skills.langchain.conftest import make_runtime


def _make_loader_mock(deleted: bool = True) -> AsyncMock:
    loader = AsyncMock(spec=PluginSkillStore)
    loader.delete_skill.return_value = deleted
    return loader


class TestDeleteSkillTool:
    @pytest.mark.asyncio
    async def test_deletes_skill_successfully(self) -> None:
        loader = _make_loader_mock(deleted=True)
        tool = DeleteSkillTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(
            plugin_name="test-plugin", skill_name="test-skill", runtime=make_runtime("user-1")
        )

        assert "deleted successfully" in result
        loader.delete_skill.assert_awaited_once_with(
            user_id="user-1", plugin_name="test-plugin", skill_name="test-skill"
        )

    @pytest.mark.asyncio
    async def test_reports_not_found(self) -> None:
        loader = _make_loader_mock(deleted=False)
        tool = DeleteSkillTool(mongo_skill_loader=loader)

        result, _ = await tool._arun(plugin_name="test-plugin", skill_name="nope", runtime=make_runtime("user-1"))

        assert "not found" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        tool = DeleteSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", runtime=make_runtime(""))

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        tool = DeleteSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="skill_name must be a non-empty"):
            await tool._arun(plugin_name="test-plugin", skill_name="  ", runtime=make_runtime())

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = DeleteSkillTool()

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", runtime=make_runtime())

    def test_sync_run_raises(self) -> None:
        tool = DeleteSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(plugin_name="test-plugin", skill_name="test", runtime=make_runtime())
