from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Any

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
from langchain_ai_skills_framework.tools.load_skill_tool import LoadSkillTool


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(self, details_by_name: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details_by_name)

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
        raise NotImplementedError()

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError()


def _make_skill(name: str, *, content: str = "Skill content") -> SkillDetails:
    source_path = Path(f"/skills/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path,
    )
    return SkillDetails(summary=summary, content=content, source_path=source_path)


def test_load_skill_tool_returns_availability_for_empty_name() -> None:
    details = _make_skill("alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    with pytest.raises(ToolException, match="No skill name provided"):
        tool._load_skill("")


def test_load_skill_tool_returns_availability_when_missing() -> None:
    details_alpha = _make_skill("alpha")
    details_beta = _make_skill("beta")
    loader = _StubSkillLoader({"beta": details_beta, "alpha": details_alpha})
    tool = LoadSkillTool(skill_loader=loader)

    with pytest.raises(ToolException, match="Skill 'gamma' not found"):
        tool._load_skill("gamma")


def test_load_skill_tool_returns_none_configured_when_no_skills_exist() -> None:
    tool = LoadSkillTool(skill_loader=_StubSkillLoader({}))

    with pytest.raises(ToolException, match="Available skills: None configured"):
        tool._load_skill("alpha")


def test_load_skill_tool_returns_skill_content() -> None:
    details = _make_skill("alpha", content="Body for alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    message = tool._load_skill(" alpha ")

    assert message == "Body for alpha"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skill_name", "message"),
    [
        (123, "Skill name must be a string."),
        ("", "No skill name provided."),
    ],
)
async def test_arun_validates_skill_name(skill_name: Any, message: str) -> None:
    details = _make_skill("alpha", content="Body for alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    with pytest.raises(ToolException, match=message):
        await tool._arun(skill_name=skill_name)


def test_get_friendly_name_casts_skill_name_to_string() -> None:
    friendly_name = LoadSkillTool.get_friendly_name(tool_input={"skill_name": None})

    assert friendly_name == "None"
