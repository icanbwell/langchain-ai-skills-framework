from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from skillkit import ScriptNotFoundError

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)
from langchain_ai_skills_framework.langchain.tools.run_skill_script_tool import (
    RunSkillScriptTool,
)
from tests.skills.langchain.conftest import make_runtime


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(
        self,
        details_by_name: Mapping[str, SkillDetails],
        script_names_by_skill: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._details = dict(details_by_name)
        self._script_names_by_skill = dict(script_names_by_skill or {})
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def list_skill_summaries(self, *, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        del allowed_skills
        return [detail.summary for detail in self._details.values()]

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
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

    async def run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        self.calls.append((skill_name, script_name, arguments))
        if skill_name not in self._details:
            raise SkillNotFoundError
        return MyScriptExecutionResult(
            stdout="script output",
            stderr=None,
            exit_code=0,
            execution_time_ms=1.0,
            success=True,
        )

    async def read_skill_resource_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str, resource_name: str
    ) -> str:
        return self.read_skill_resource(skill_name=skill_name, resource_name=resource_name)

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
        return self._script_names_by_skill.get(skill_name, [])

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


class _FailingScriptLoader(_StubSkillLoader):
    async def run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        self.calls.append((skill_name, script_name, arguments))
        return MyScriptExecutionResult(
            stdout=None,
            stderr="boom",
            exit_code=2,
            execution_time_ms=1.0,
            success=False,
        )


class _ScriptNotFoundLoader(_StubSkillLoader):
    async def run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        self.calls.append((skill_name, script_name, arguments))
        raise ScriptNotFoundError(f"Script '{script_name}' not found")


def _make_skill(name: str) -> SkillDetails:
    source_path = Path(f"/skills/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path,
    )
    return SkillDetails(
        summary=summary,
        content=f"Body for {name}",
        source_path=source_path,
    )


@pytest.mark.asyncio
async def test_arun_executes_script_with_named_arguments() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = RunSkillScriptTool(skill_loader=loader)

    message, output = await tool._arun(
        plugin_name="test-plugin",
        skill_name="alpha",
        script_name="analyze.py",
        arguments={"threshold": 0.5},
        runtime=make_runtime(),
    )

    assert message == "script output"
    assert output == "script output"
    assert loader.calls == [("alpha", "analyze.py", {"threshold": 0.5})]


@pytest.mark.asyncio
async def test_arun_returns_not_found_message_when_skill_missing() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = RunSkillScriptTool(skill_loader=loader)

    result = await tool._arun(
        plugin_name="test-plugin",
        skill_name="missing",
        script_name="analyze.py",
        arguments=None,
        runtime=make_runtime(),
    )
    assert "Skill 'missing' not found." in result[0]


@pytest.mark.asyncio
async def test_arun_returns_error_output_when_script_fails() -> None:
    loader = _FailingScriptLoader({"alpha": _make_skill("alpha")})
    tool = RunSkillScriptTool(skill_loader=loader)

    message, artifact = await tool._arun(
        plugin_name="test-plugin",
        skill_name="alpha",
        script_name="analyze.py",
        arguments=None,
        runtime=make_runtime(),
    )
    assert message == "boom"
    assert artifact == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skill_name", "script_name", "arguments", "message"),
    [
        ("", "analyze.py", None, "No skill name provided."),
        ("alpha", "", None, "No script name provided."),
    ],
)
async def test_arun_validates_parameters_raises(
    skill_name: Any,
    script_name: Any,
    arguments: Any,
    message: str,
) -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = RunSkillScriptTool(skill_loader=loader)

    with pytest.raises(ToolException, match=message):
        await tool._arun(
            plugin_name="test-plugin",
            skill_name=skill_name,
            script_name=script_name,
            arguments=arguments,
            runtime=make_runtime(),
        )


@pytest.mark.asyncio
async def test_arun_script_not_found_lists_available_scripts() -> None:
    loader = _ScriptNotFoundLoader(
        {"alpha": _make_skill("alpha")},
        script_names_by_skill={"alpha": ["analyze.py", "process.py"]},
    )
    tool = RunSkillScriptTool(skill_loader=loader)

    result, artifact = await tool._arun(
        plugin_name="test-plugin",
        skill_name="alpha",
        script_name="missing.py",
        arguments=None,
        runtime=make_runtime(),
    )
    assert "Script 'missing.py' not found in skill 'alpha'" in result
    assert "analyze.py" in result
    assert "process.py" in result
    assert "Available scripts:" in result


@pytest.mark.asyncio
async def test_arun_script_not_found_no_scripts_shows_none() -> None:
    loader = _ScriptNotFoundLoader({"alpha": _make_skill("alpha")})
    tool = RunSkillScriptTool(skill_loader=loader)

    result, artifact = await tool._arun(
        plugin_name="test-plugin",
        skill_name="alpha",
        script_name="missing.py",
        arguments=None,
        runtime=make_runtime(),
    )
    assert "Script 'missing.py' not found in skill 'alpha'" in result
    assert "Available scripts: none" in result


def test_sync_run_raises() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = RunSkillScriptTool(skill_loader=loader)

    with pytest.raises(NotImplementedError):
        tool._run(plugin_name="test-plugin", skill_name="alpha", script_name="analyze.py", runtime=make_runtime())


def test_get_friendly_name_casts_inputs_to_string() -> None:
    friendly_name = RunSkillScriptTool.get_friendly_name(tool_input={"skill_name": None, "script_name": 123})

    assert friendly_name == "None (123)"
