from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
from langchain_core.tools import StructuredTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)
from langchain_ai_skills_framework.tools.run_inline_skill_script_tool import (
    RunInlineSkillScriptTool,
)


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(self, details_by_name: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details_by_name)
        self.calls: list[tuple[str, str, str, dict[str, Any] | None]] = []

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

    def get_tools(self) -> list[StructuredTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        raise NotImplementedError()

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        del skill_name, script_name, arguments
        raise NotImplementedError()

    async def run_inline_skill_script(
        self,
        skill_name: str,
        script_name: str,
        script: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        self.calls.append((skill_name, script_name, script, arguments))
        if skill_name not in self._details:
            raise SkillNotFoundError
        return MyScriptExecutionResult(
            stdout="script output",
            stderr=None,
            exit_code=0,
            execution_time_ms=1.0,
            success=True,
        )


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


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (("alpha", "analyze.py", "print('ok')"), {}, "print('ok')"),
        (("alpha", "analyze.py"), {}, ""),
        (
            ("alpha", "analyze.py", "print('ok')"),
            {"script": "print('better')"},
            "print('better')",
        ),
    ],
)
def test_resolve_script_prefers_kwargs_and_uses_third_positional_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], expected: str
) -> None:
    assert (
        RunInlineSkillScriptTool._resolve_script(args=args, kwargs=kwargs) == expected
    )


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (
            ("alpha", "analyze.py", "print('ok')", {"threshold": 1}),
            {},
            {"threshold": 1},
        ),
        (("alpha", "analyze.py", "print('ok')"), {}, None),
        (
            ("alpha", "analyze.py", "print('ok')", {"threshold": 1}),
            {"arguments": {"threshold": 2}},
            {"threshold": 2},
        ),
    ],
)
def test_resolve_arguments_prefers_kwargs_and_uses_fourth_positional_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    assert (
        RunInlineSkillScriptTool._resolve_arguments(args=args, kwargs=kwargs)
        == expected
    )


def test_run_uses_positional_mapping_for_script_and_arguments() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = RunInlineSkillScriptTool(skill_loader=loader)

    message = tool._run(
        "alpha",
        "analyze.py",
        "print('ok')",
        {"threshold": 0.5},
        config={},
    )

    assert message == "script output"
    assert loader.calls == [("alpha", "analyze.py", "print('ok')", {"threshold": 0.5})]


@pytest.mark.asyncio
async def test_arun_uses_positional_mapping_for_script_and_arguments() -> None:
    loader = _StubSkillLoader({"alpha": _make_skill("alpha")})
    tool = RunInlineSkillScriptTool(skill_loader=loader)

    message = await tool._arun(
        "alpha",
        "analyze.py",
        "print('ok')",
        {"threshold": 0.5},
        config={},
    )

    assert message == "script output"
    assert loader.calls == [("alpha", "analyze.py", "print('ok')", {"threshold": 0.5})]
