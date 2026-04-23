from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
)
from langchain_ai_skills_framework.langchain.tools.list_plugins_tool import ListPluginsTool


def _make_loader_mock(
    plugins: list[MongoPluginDefinitionDocument] | None = None,
) -> AsyncMock:
    loader = AsyncMock(spec=PluginSkillStore)
    loader.list_plugins.return_value = plugins or []
    return loader


def _make_runtime(user_id: str = "user-1") -> MagicMock:
    runtime = MagicMock()
    runtime.context = {"user_id": user_id}
    return runtime


def _make_plugin(
    name: str = "test-plugin",
    description: str = "A test plugin",
    skills: list[str] | None = None,
) -> MongoPluginDefinitionDocument:
    return MongoPluginDefinitionDocument(
        plugin_name=name,
        description=description,
        skills=skills or [],
    )


class TestListPluginsTool:
    @pytest.mark.asyncio
    async def test_lists_plugins_successfully(self) -> None:
        plugins = [
            _make_plugin(name="beta-plugin", description="Beta", skills=["b-skill"]),
            _make_plugin(name="alpha-plugin", description="Alpha", skills=["a-skill"]),
        ]
        loader = _make_loader_mock(plugins)
        tool = ListPluginsTool(mongo_skill_loader=loader)

        result, artifact = await tool._arun(runtime=_make_runtime())

        assert "alpha-plugin" in result
        assert "beta-plugin" in result
        # Sorted alphabetically — alpha before beta
        assert result.index("alpha-plugin") < result.index("beta-plugin")
        loader.list_plugins.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_plugins(self) -> None:
        loader = _make_loader_mock([])
        tool = ListPluginsTool(mongo_skill_loader=loader)

        result, _ = await tool._arun(runtime=_make_runtime())

        assert "<available_plugins>" in result
        assert "<plugin>" not in result

    @pytest.mark.asyncio
    async def test_includes_skills_in_output(self) -> None:
        plugins = [_make_plugin(skills=["skill-a", "skill-b"])]
        loader = _make_loader_mock(plugins)
        tool = ListPluginsTool(mongo_skill_loader=loader)

        result, _ = await tool._arun(runtime=_make_runtime())

        assert "skill-a" in result
        assert "skill-b" in result

    @pytest.mark.asyncio
    async def test_rejects_when_loader_not_configured(self) -> None:
        tool = ListPluginsTool()

        with pytest.raises(ToolException, match="mongo_skill_loader is not configured"):
            await tool._arun(runtime=_make_runtime())

    def test_sync_run_raises(self) -> None:
        tool = ListPluginsTool(mongo_skill_loader=_make_loader_mock())

        with pytest.raises(NotImplementedError):
            tool._run(runtime=_make_runtime())

    @pytest.mark.asyncio
    async def test_includes_description_in_output(self) -> None:
        plugins = [_make_plugin(description="Does amazing things")]
        loader = _make_loader_mock(plugins)
        tool = ListPluginsTool(mongo_skill_loader=loader)

        result, _ = await tool._arun(runtime=_make_runtime())

        assert "Does amazing things" in result
