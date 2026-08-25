from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.langchain.tools.publish_skill_tool import (
    PublishSkillTool,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from tests.skills.langchain.conftest import make_runtime


def _make_doc(skill_name: str = "test-skill", state: str = "in_review") -> MongoPluginSkillDocument:
    return MongoPluginSkillDocument(
        plugin_name="test-plugin",
        author="user-1",
        skill_name=skill_name,
        path=f"test-plugin/skills/{skill_name}/SKILL.md",
        description="A test",
        content="# Test\nContent",
        state=state,
        modified_by="user-1",
        date_created=datetime.now(UTC),
        date_modified=datetime.now(UTC),
    )


def _make_loader_mock(state: str = "in_review") -> AsyncMock:
    from langchain_ai_skills_framework.models.skills_model import (
        SkillDetails,
        SkillSummary,
    )

    loader = AsyncMock(spec=PluginSkillStore)
    loader.skill_exists.return_value = True
    loader.set_skill_state.return_value = _make_doc(state=state)

    # Mock get_skill_details to return staging state for validation
    summary = SkillSummary(
        name="test-skill",
        description="A test",
        plugin_name="test-plugin",
        state="staging",
    )
    loader.get_skill_details.return_value = SkillDetails(
        summary=summary,
        content="# Test\nContent",
    )
    return loader


def _make_publisher_mock() -> MagicMock:
    publisher = MagicMock(spec=GitHubMarketplacePublisher)
    publisher.use_branch = True
    publisher.publish_skill = AsyncMock(return_value="https://github.com/org/repo/pull/1")
    publisher.unpublish_skill = AsyncMock(return_value="https://github.com/org/repo/pull/2")
    return publisher


class TestPublishSkillTool:
    @pytest.mark.asyncio
    async def test_publishes_skill_successfully(self) -> None:
        loader = _make_loader_mock(state="in_review")
        publisher = _make_publisher_mock()
        tool = PublishSkillTool(mongo_skill_loader=loader, marketplace_publisher=publisher)

        result, _artifact = await tool._arun(
            plugin_name="test-plugin", skill_name="test-skill", published=True, runtime=make_runtime("user-1")
        )

        assert "submitted for review" in result
        loader.set_skill_state.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            state="in_review",
            published_branch="skill-publish/test-plugin/test-skill",
        )

    @pytest.mark.asyncio
    async def test_unpublishes_skill(self) -> None:
        loader = _make_loader_mock(state="draft")
        tool = PublishSkillTool(mongo_skill_loader=loader)

        result, _ = await tool._arun(
            plugin_name="test-plugin", skill_name="test-skill", published=False, runtime=make_runtime("user-1")
        )

        assert "unpublished" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        tool = PublishSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="user_id is required"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", published=True, runtime=make_runtime(""))

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        tool = PublishSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(ToolException, match="skill_name must be a non-empty"):
            await tool._arun(plugin_name="test-plugin", skill_name="  ", published=True, runtime=make_runtime())

    @pytest.mark.asyncio
    async def test_publishes_locally_when_publisher_not_configured(self) -> None:
        loader = _make_loader_mock(state="in_review")
        tool = PublishSkillTool(mongo_skill_loader=loader)

        result, _artifact = await tool._arun(
            plugin_name="test-plugin", skill_name="test-skill", published=True, runtime=make_runtime()
        )

        assert "submitted for review" in result
        loader.set_skill_state.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            state="in_review",
            published_branch=None,
        )

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = PublishSkillTool(marketplace_publisher=_make_publisher_mock())

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", published=True, runtime=make_runtime())

    @pytest.mark.asyncio
    async def test_wraps_unexpected_exception(self) -> None:
        from langchain_ai_skills_framework.models.skills_model import (
            SkillDetails,
            SkillSummary,
        )

        loader = AsyncMock(spec=PluginSkillStore)
        loader.skill_exists.return_value = True

        # Mock get_skill_details to return staging state
        summary = SkillSummary(
            name="test",
            description="A test",
            plugin_name="test-plugin",
            state="staging",
        )
        loader.get_skill_details.return_value = SkillDetails(
            summary=summary,
            content="# Test\nContent",
        )

        loader.set_skill_state.side_effect = RuntimeError("db down")
        tool = PublishSkillTool(mongo_skill_loader=loader, marketplace_publisher=_make_publisher_mock())

        with pytest.raises(ToolException, match="Unable to update publishing"):
            await tool._arun(plugin_name="test-plugin", skill_name="test", published=True, runtime=make_runtime())

    def test_sync_run_raises(self) -> None:
        tool = PublishSkillTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(plugin_name="test-plugin", skill_name="test", published=True, runtime=make_runtime())

    def test_get_friendly_name(self) -> None:
        name = PublishSkillTool.get_friendly_name(tool_input={"skill_name": "my-skill"})
        assert name == "Publish Skill: my-skill"

    def test_get_friendly_name_empty(self) -> None:
        name = PublishSkillTool.get_friendly_name(tool_input={})
        assert name == "Publish Skill"
