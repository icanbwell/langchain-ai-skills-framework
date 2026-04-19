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
from langchain_ai_skills_framework.tools.save_skill_tool import SaveSkillTool

VALID_SKILL_CONTENT = "---\nname: test-skill\ndescription: A test skill\n---\n# Test\nContent"


def _make_loader_mock() -> AsyncMock:
    loader = AsyncMock(spec=UserSkillStore)
    loader.save_skill.return_value = MongoSkillDocument(
        user_id="user-1",
        skill_name="test-skill",
        description="A test",
        content=VALID_SKILL_CONTENT,
        modified_by="user-1",
        date_created=datetime.now(timezone.utc),
        date_modified=datetime.now(timezone.utc),
    )
    return loader


def _make_runtime(user_id: str = "user-1") -> MagicMock:
    """Create a mock ToolRuntime with the given user_id in context."""
    runtime = MagicMock()
    runtime.context = {"user_id": user_id}
    return runtime


class TestSaveSkillTool:
    @pytest.mark.asyncio
    async def test_saves_skill_successfully(self) -> None:
        loader = _make_loader_mock()
        tool = SaveSkillTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(
            skill_name="test-skill",
            content=VALID_SKILL_CONTENT,
            runtime=_make_runtime("user-1"),
        )

        assert "saved successfully" in result
        loader.save_skill.assert_awaited_once_with(
            user_id="user-1",
            skill_name="test-skill",
            content=VALID_SKILL_CONTENT,
            modified_by="user-1",
        )

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(skill_name="test", content="content", runtime=_make_runtime(""))

    @pytest.mark.asyncio
    async def test_rejects_missing_user_id_in_context(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())
        runtime = MagicMock()
        runtime.context = {}

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(skill_name="test", content="content", runtime=runtime)

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="skill_name must be a non-empty"):
            await tool._arun(skill_name="", content="content", runtime=_make_runtime())

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="content must be a non-empty"):
            await tool._arun(skill_name="test", content="", runtime=_make_runtime())

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_frontmatter(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        result, artifact = await tool._arun(
            skill_name="test",
            content="# No frontmatter here",
            runtime=_make_runtime(),
        )

        assert "Skill validation failed" in result

    @pytest.mark.asyncio
    async def test_returns_error_for_invalid_metadata(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        result, artifact = await tool._arun(
            skill_name="test",
            content="---\ninvalid_field: true\n---\n# Missing required fields",
            runtime=_make_runtime(),
        )

        assert "Skill validation failed" in result

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = SaveSkillTool()

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(skill_name="test", content="content", runtime=_make_runtime())

    def test_sync_run_raises(self) -> None:
        tool = SaveSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(skill_name="test", content="content", runtime=_make_runtime())

    def test_get_friendly_name(self) -> None:
        name = SaveSkillTool.get_friendly_name(tool_input={"skill_name": "my-skill"})
        assert name == "Save Skill: my-skill"

    def test_get_friendly_name_empty(self) -> None:
        name = SaveSkillTool.get_friendly_name(tool_input={})
        assert name == "Save Skill"
