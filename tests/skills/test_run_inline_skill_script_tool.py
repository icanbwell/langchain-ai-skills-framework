from __future__ import annotations

from typing import Any

import pytest

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.tools.run_python_script_tool import (
    RunPythonScriptTool,
)


class _StubExecutor:
    calls: list[tuple[str, str, dict[str, Any], int]] = []

    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
        use_uv: bool = True,
    ) -> MyScriptExecutionResult:
        del use_uv
        _StubExecutor.calls.append((script_name, script, arguments, timeout))
        return MyScriptExecutionResult(
            stdout="script output",
            stderr=None,
            exit_code=0,
            execution_time_ms=1.0,
            success=True,
        )


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (("print('ok')",), {}, "print('ok')"),
        ((), {}, ""),
        (
            ("print('ok')",),
            {"script": "print('better')"},
            "print('better')",
        ),
    ],
)
def test_resolve_script_prefers_kwargs_and_uses_first_positional_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], expected: str
) -> None:
    assert RunPythonScriptTool._resolve_script(args=args, kwargs=kwargs) == expected


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (
            ("print('ok')", {"threshold": 1}),
            {},
            {"threshold": 1},
        ),
        (("print('ok')",), {}, None),
        (
            ("print('ok')", {"threshold": 1}),
            {"arguments": {"threshold": 2}},
            {"threshold": 2},
        ),
    ],
)
def test_resolve_arguments_prefers_kwargs_and_uses_second_positional_arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], expected: dict[str, Any] | None
) -> None:
    assert RunPythonScriptTool._resolve_arguments(args=args, kwargs=kwargs) == expected


def test_run_uses_positional_mapping_for_script_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubExecutor.calls = []
    monkeypatch.setattr(
        "langchain_ai_skills_framework.tools.run_python_script_tool.MyScriptExecutor",
        _StubExecutor,
    )
    tool = RunPythonScriptTool()

    message = tool._run(
        "print('ok')",
        {"MixedCase": 0.5},
        config={},
    )

    assert message == "script output"
    assert _StubExecutor.calls == [
        ("inline_script.py", "print('ok')", {"mixedcase": 0.5}, 30)
    ]


@pytest.mark.asyncio
async def test_arun_uses_positional_mapping_for_script_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubExecutor.calls = []
    tool = RunPythonScriptTool()
    monkeypatch.setattr(
        "langchain_ai_skills_framework.tools.run_python_script_tool.MyScriptExecutor",
        _StubExecutor,
    )

    message = await tool._arun(
        "print('ok')",
        {"MixedCase": 0.5},
        config={},
    )

    assert message == "script output"
    assert _StubExecutor.calls == [
        ("inline_script.py", "print('ok')", {"mixedcase": 0.5}, 30)
    ]
