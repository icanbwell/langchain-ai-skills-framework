from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool
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
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary
from langchain_ai_skills_framework.tools.read_skill_resource_tool import (
    ReadSkillResourceTool,
)


def _make_runtime(user_id: str = "user-1") -> MagicMock:
    """Create a mock ToolRuntime with the given user_id in context."""
    runtime = MagicMock()
    runtime.context = {"user_id": user_id}
    return runtime


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(
        self,
        details_by_name: Mapping[str, SkillDetails],
        resource_names_by_skill: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._details = dict(details_by_name)
        self._resource_names_by_skill = dict(resource_names_by_skill or {})
        self.calls: list[tuple[str, str]] = []

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        del allowed_skills
        return [detail.summary for detail in self._details.values()]

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills)

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        try:
            return self._details[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError from exc

    async def get_skill_details_for_user(self, *, user_id: str, skill_name: str) -> SkillDetails:
        return self.get_skill_details(skill_name)

    def refresh(self) -> None:
        return None

    async def get_instructions(self) -> str:  # pragma: no cover
        return ""

    def get_tools(self) -> list[BaseTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        self.calls.append((skill_name, resource_name))
        if skill_name not in self._details:
            raise SkillNotFoundError
        return f"{skill_name}:{resource_name}"

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError()

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return []

    async def list_skill_script_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_script_names(skill_name)

    async def read_skill_resource_for_user(self, *, user_id: str, skill_name: str, resource_name: str) -> str:
        return self.read_skill_resource(skill_name, resource_name)

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name, script_name, arguments)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return self._resource_names_by_skill.get(skill_name, [])

    async def list_skill_resource_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name)


class _ResourceNotFoundLoader(_StubSkillLoader):
    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        self.calls.append((skill_name, resource_name))
        raise SkillNotFoundError(f"Resource '{resource_name}' not found")


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
async def test_run_reads_named_resource() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    message, artifact = await tool._arun(skill_name="alpha", resource_name="FORMS.md", runtime=_make_runtime())

    assert message == "alpha:FORMS.md"
    assert artifact == "alpha:FORMS.md"
    assert loader.calls == [("alpha", "FORMS.md")]


@pytest.mark.asyncio
async def test_run_returns_not_found_message_for_missing_skill() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    message, artifact = await tool._arun(skill_name="missing", resource_name="FORMS.md", runtime=_make_runtime())

    assert "not found" in message
    assert "missing" in message
    assert "Available skills:" in message


@pytest.mark.asyncio
async def test_run_raises_tool_exception_for_empty_skill_name() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    with pytest.raises(ToolException, match="No skill name provided"):
        await tool._arun(skill_name=" ", resource_name="FORMS.md", runtime=_make_runtime())


@pytest.mark.asyncio
async def test_arun_validates_empty_skill_name_raises() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    with pytest.raises(ToolException, match="No skill name provided."):
        await tool._arun(
            skill_name="",
            resource_name="FORMS.md",
            runtime=_make_runtime(),
        )


@pytest.mark.asyncio
async def test_arun_validates_empty_resource_name_returns_error() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    result, artifact = await tool._arun(
        skill_name="alpha",
        resource_name="",
        runtime=_make_runtime(),
    )
    assert result == "No resource name provided."
    assert artifact == ""


@pytest.mark.asyncio
async def test_run_resource_not_found_lists_available_resources() -> None:
    loader = _ResourceNotFoundLoader(
        {"alpha": _make_skill("alpha")},
        resource_names_by_skill={"alpha": ["FORMS.md", "REFERENCE.md"]},
    )
    tool = ReadSkillResourceTool(skill_loader=loader)

    message, artifact = await tool._arun(skill_name="alpha", resource_name="MISSING.md", runtime=_make_runtime())
    assert "Resource 'MISSING.md' not found in skill 'alpha'" in message
    assert "FORMS.md" in message
    assert "REFERENCE.md" in message
    assert "Available resources:" in message


@pytest.mark.asyncio
async def test_run_resource_not_found_no_resources_shows_none() -> None:
    loader = _ResourceNotFoundLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    message, artifact = await tool._arun(skill_name="alpha", resource_name="MISSING.md", runtime=_make_runtime())
    assert "Resource 'MISSING.md' not found in skill 'alpha'" in message
    assert "Available resources: none" in message


def test_sync_run_raises() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    with pytest.raises(NotImplementedError):
        tool._run(skill_name="alpha", resource_name="FORMS.md", runtime=_make_runtime())


def test_get_friendly_name_casts_inputs_to_string() -> None:
    friendly_name = ReadSkillResourceTool.get_friendly_name(tool_input={"skill_name": None, "resource_name": 123})

    assert friendly_name == "None 123"
