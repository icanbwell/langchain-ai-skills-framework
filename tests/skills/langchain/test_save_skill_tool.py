from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.langchain.tools.save_skill_tool import SaveSkillTool
from tests.skills.langchain.conftest import make_runtime

VALID_SKILL_CONTENT = "---\nname: test-skill\ndescription: A test skill\n---\n# Test\nContent"


def _make_loader_mock() -> AsyncMock:
    loader = AsyncMock(spec=PluginSkillStore)
    loader.save_skill.return_value = MongoPluginSkillDocument(
        plugin_name="test-plugin",
        author="user-1",
        skill_name="test-skill",
        path="test-plugin/skills/test-skill/SKILL.md",
        description="A test",
        content=VALID_SKILL_CONTENT,
        modified_by="user-1",
        date_created=datetime.now(timezone.utc),
        date_modified=datetime.now(timezone.utc),
    )
    return loader


class TestSaveSkillTool:
    @pytest.mark.asyncio
    async def test_saves_skill_successfully(self) -> None:
        loader = _make_loader_mock()
        tool = SaveSkillTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(
            plugin_name="test-plugin",
            skill_name="test-skill",
            content=VALID_SKILL_CONTENT,
            runtime=make_runtime("user-1"),
        )

        assert "saved successfully" in result
        loader.save_skill.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            content=VALID_SKILL_CONTENT,
            modified_by="user-1",
            folder=None,
            path=None,
            state=None,
        )

    @pytest.mark.asyncio
    async def test_saves_skill_with_folder(self) -> None:
        loader = _make_loader_mock()
        tool = SaveSkillTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(
            plugin_name="test-plugin",
            skill_name="test-skill",
            content=VALID_SKILL_CONTENT,
            folder="sub/dir",
            runtime=make_runtime("user-1"),
        )

        assert "saved successfully" in result
        loader.save_skill.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            content=VALID_SKILL_CONTENT,
            modified_by="user-1",
            folder="sub/dir",
            path=None,
            state=None,
        )

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", content="content", runtime=make_runtime(""))

    @pytest.mark.asyncio
    async def test_rejects_missing_user_id_in_context(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())
        runtime = MagicMock()
        runtime.context = {}

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", content="content", runtime=runtime)

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name_without_frontmatter(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        result, artifact = await tool._arun(
            plugin_name="test-plugin", skill_name="", content="content", runtime=make_runtime()
        )

        assert "Skill validation failed" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="content must be a non-empty"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", content="", runtime=make_runtime())

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_frontmatter(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        result, artifact = await tool._arun(
            plugin_name="test-plugin",
            skill_name="test",
            content="# No frontmatter here",
            runtime=make_runtime(),
        )

        assert "Skill validation failed" in result

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_metadata(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        result, artifact = await tool._arun(
            plugin_name="test-plugin",
            skill_name="test",
            content="---\ninvalid_field: true\n---\n# Missing required fields",
            runtime=make_runtime(),
        )

        assert "Skill validation failed" in result

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = SaveSkillTool()

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", content="content", runtime=make_runtime())

    def test_sync_run_raises(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(plugin_name="test-plugin", skill_name="test", content="content", runtime=make_runtime())

    def test_get_friendly_name(self) -> None:
        name = SaveSkillTool.get_friendly_name(tool_input={"skill_name": "my-skill"})
        assert name == "Save Skill: my-skill"

    def test_get_friendly_name_empty(self) -> None:
        name = SaveSkillTool.get_friendly_name(tool_input={})
        assert name == "Save Skill"
