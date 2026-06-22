from __future__ import annotations

from types import MappingProxyType
from typing import Any, Sequence
from unittest.mock import AsyncMock


from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import MyScriptExecutor
from langchain_ai_skills_framework.langchain.tools.tool_factory import build_skill_tools
from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)


class _StubSharedLoader(SkillLoaderProtocol):
    def __init__(self) -> None:
        pass

    def list_skill_summaries(self, *, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return []

    async def list_all_summaries(
        self, *, user_id: str, allowed_skills: set[str], include_staging: bool = False
    ) -> Sequence[SkillSummary]:
        return []

    def get_skill_details(self, *, skill_name: str, plugin_name: str | None = None) -> SkillDetails:
        raise SkillNotFoundError(f"'{skill_name}' not found")

    async def get_skill_details_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> SkillDetails:
        return self.get_skill_details(skill_name=skill_name)

    def refresh(self) -> None:
        pass

    async def refresh_async(self) -> None:
        pass

    async def get_instructions(self) -> str:
        return ""

    def read_skill_resource(self, *, skill_name: str, resource_name: str, plugin_name: str | None = None) -> str:
        raise NotImplementedError

    async def read_skill_resource_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str, resource_name: str
    ) -> str:
        raise NotImplementedError

    async def run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        raise NotImplementedError

    def list_skill_script_names(self, *, skill_name: str, plugin_name: str | None = None) -> Sequence[str]:
        return []

    async def list_skill_script_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        return []

    def list_skill_resource_names(self, *, skill_name: str, plugin_name: str | None = None) -> Sequence[str]:
        return []

    async def list_skill_resource_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        return []

    async def get_plugin_mcp_configs(self) -> Sequence[PluginMcpServerEntry]:
        return []

    async def list_plugin_definitions(self) -> Sequence[PluginDefinition]:
        return []


def _make_user_loader_mock() -> PluginSkillStore:
    loader = AsyncMock(spec=PluginSkillStore)
    empty_snapshot = SkillSnapshot(
        details_by_name=MappingProxyType({}),
        ordered_summaries=(),
    )
    loader.load_snapshot.return_value = empty_snapshot
    loader.load_shared_snapshot.return_value = empty_snapshot
    loader.get_skill_usage_counts.return_value = {}
    return loader


class TestBuildSkillTools:
    def test_includes_expected_tools(self) -> None:
        shared = _StubSharedLoader()
        user_loader = _make_user_loader_mock()
        composite = CompositeSkillLoader(
            shared_loader=shared, user_loader=user_loader, script_executor=MyScriptExecutor()
        )

        tools = build_skill_tools(skill_loader=composite, user_skill_store=user_loader)
        tool_names = [t.name for t in tools]

        assert "save_skill" in tool_names
        assert "delete_skill" in tool_names
        assert "list_plugins" in tool_names
        assert "publish_skill" in tool_names
        assert "list_skills" in tool_names
        assert "load_skill" in tool_names
