from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.tools.run_python_script_tool import (
    RunPythonScriptOutput,
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


class _FailingExecutor:
    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
        use_uv: bool = True,
    ) -> MyScriptExecutionResult:
        del script_name, script, arguments, timeout, use_uv
        return MyScriptExecutionResult(
            stdout=None,
            stderr="inline boom",
            exit_code=3,
            execution_time_ms=1.0,
            success=False,
        )


def test_run_returns_summary_and_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubExecutor.calls = []
    monkeypatch.setattr(
        "langchain_ai_skills_framework.tools.run_python_script_tool.MyScriptExecutor",
        _StubExecutor,
    )
    tool = RunPythonScriptTool()

    message, output = tool._run(
        "print('ok')",
        {"MixedCase": 0.5},
    )

    assert message == "script output"
    assert output == RunPythonScriptOutput(
        success=True,
        stdout="script output",
        stderr=None,
        exit_code=0,
        error_message=None,
    )
    assert _StubExecutor.calls == [
        ("inline_script.py", "print('ok')", {"mixedcase": 0.5}, 30)
    ]


@pytest.mark.asyncio
async def test_arun_returns_summary_and_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _StubExecutor.calls = []
    tool = RunPythonScriptTool()
    monkeypatch.setattr(
        "langchain_ai_skills_framework.tools.run_python_script_tool.MyScriptExecutor",
        _StubExecutor,
    )

    message, output = await tool._arun(
        "print('ok')",
        {"MixedCase": 0.5},
    )

    assert message == "script output"
    assert output == RunPythonScriptOutput(
        success=True,
        stdout="script output",
        stderr=None,
        exit_code=0,
        error_message=None,
    )
    assert _StubExecutor.calls == [
        ("inline_script.py", "print('ok')", {"mixedcase": 0.5}, 30)
    ]


@pytest.mark.asyncio
async def test_arun_raises_tool_exception_when_inline_script_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = RunPythonScriptTool()
    monkeypatch.setattr(
        "langchain_ai_skills_framework.tools.run_python_script_tool.MyScriptExecutor",
        _FailingExecutor,
    )

    with pytest.raises(ToolException, match="Inline script failed"):
        await tool._arun("print('fail')", None)
