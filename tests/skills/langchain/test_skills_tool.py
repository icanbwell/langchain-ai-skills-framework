from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Any

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary
from langchain_ai_skills_framework.langchain.tools.load_skill_tool import LoadSkillTool
from tests.skills.langchain.conftest import make_runtime


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(self, details_by_name: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details_by_name)

    def list_skill_summaries(self, *, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        del allowed_skills
        return [detail.summary for detail in self._details.values()]

    async def list_all_summaries(
        self, *, user_id: str, allowed_skills: set[str], include_staging: bool = False
    ) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills=allowed_skills)

    def get_skill_details(self, *, skill_name: str, plugin_name: str | None = None) -> SkillDetails:
        try:
            return self._details[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError from exc

    async def get_skill_details_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> SkillDetails:
        return self.get_skill_details(skill_name=skill_name)

    def refresh(self) -> None:
        return None

    async def refresh_async(self) -> None:
        return None

    async def get_instructions(self) -> str:  # pragma: no cover
        return ""

    def read_skill_resource(self, *, skill_name: str, resource_name: str, plugin_name: str | None = None) -> str:
        raise NotImplementedError()

    async def read_skill_resource_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str, resource_name: str
    ) -> str:
        return self.read_skill_resource(skill_name=skill_name, resource_name=resource_name)

    async def run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError()

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name=skill_name, script_name=script_name, arguments=arguments)

    def list_skill_script_names(self, *, skill_name: str, plugin_name: str | None = None) -> Sequence[str]:
        return []

    async def list_skill_script_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        return self.list_skill_script_names(skill_name=skill_name)

    def list_skill_resource_names(self, *, skill_name: str, plugin_name: str | None = None) -> Sequence[str]:
        return []

    async def list_skill_resource_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name=skill_name)

    async def get_plugin_mcp_configs(self) -> Sequence[PluginMcpServerEntry]:
        return []

    async def list_plugin_definitions(self) -> Sequence[PluginDefinition]:
        return []


def _make_skill(name: str, *, content: str = "Skill content") -> SkillDetails:
    source_path = Path(f"/skills/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path,
    )
    return SkillDetails(summary=summary, content=content, source_path=source_path)


@pytest.mark.asyncio
async def test_load_skill_tool_returns_availability_for_empty_name() -> None:
    details = _make_skill("alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    with pytest.raises(ToolException, match="No skill name provided"):
        await tool._arun(plugin_name="test-plugin", skill_name="", runtime=make_runtime())


@pytest.mark.asyncio
async def test_load_skill_tool_returns_availability_when_missing() -> None:
    details_alpha = _make_skill("alpha")
    details_beta = _make_skill("beta")
    loader = _StubSkillLoader({"beta": details_beta, "alpha": details_alpha})
    tool = LoadSkillTool(skill_loader=loader)

    result, _ = await tool._arun(plugin_name="test-plugin", skill_name="gamma", runtime=make_runtime())

    assert "Skill 'gamma' not found" in result
    assert "Available skills: alpha, beta" in result


@pytest.mark.asyncio
async def test_load_skill_tool_returns_none_configured_when_no_skills_exist() -> None:
    tool = LoadSkillTool(skill_loader=_StubSkillLoader({}))

    result, _ = await tool._arun(plugin_name="test-plugin", skill_name="alpha", runtime=make_runtime())

    assert "Skill 'alpha' not found" in result
    assert "Available skills: None configured" in result


@pytest.mark.asyncio
async def test_load_skill_tool_returns_skill_content() -> None:
    details = _make_skill("alpha", content="Body for alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    message, _ = await tool._arun(plugin_name="test-plugin", skill_name=" alpha ", runtime=make_runtime())

    assert message == "Body for alpha"


def test_sync_run_raises() -> None:
    tool = LoadSkillTool(skill_loader=_StubSkillLoader({}))

    with pytest.raises(NotImplementedError):
        tool._run(plugin_name="test-plugin", skill_name="test", runtime=make_runtime())


def test_get_friendly_name_casts_skill_name_to_string() -> None:
    friendly_name = LoadSkillTool.get_friendly_name(tool_input={"skill_name": None})

    assert friendly_name == "None"
