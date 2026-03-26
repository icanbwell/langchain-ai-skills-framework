from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

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


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(self, details_by_name: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details_by_name)
        self.calls: list[tuple[str, str]] = []

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        del allowed_skills
        return [detail.summary for detail in self._details.values()]

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        try:
            return self._details[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError from exc

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


def test_run_uses_second_positional_arg_for_resource_name() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    message = tool._run("alpha", "FORMS.md")

    assert message == "alpha:FORMS.md"
    assert loader.calls == [("alpha", "FORMS.md")]


def test_run_raises_tool_exception_for_missing_skill() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    with pytest.raises(ToolException, match="Skill 'missing' not found"):
        tool._run("missing", "FORMS.md")


def test_run_raises_tool_exception_for_empty_skill_name() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = ReadSkillResourceTool(skill_loader=loader)

    with pytest.raises(ToolException, match="No skill name provided"):
        tool._run(" ", "FORMS.md")
