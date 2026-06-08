from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import ToolException

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.langchain.tools.run_python_script_tool import (
    RunPythonScriptTool,
)
from tests.skills.langchain.conftest import make_runtime


class _StubExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any], int]] = []

    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
    ) -> MyScriptExecutionResult:
        self.calls.append((script_name, script, arguments, timeout))
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
    ) -> MyScriptExecutionResult:
        del script_name, script, arguments, timeout
        return MyScriptExecutionResult(
            stdout=None,
            stderr="inline boom",
            exit_code=3,
            execution_time_ms=1.0,
            success=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("script_name", ["inline_script.py", "custom_script.py"])
async def test_arun_passes_script_name_and_returns_output(*, script_name: str) -> None:
    stub = _StubExecutor()
    tool = RunPythonScriptTool(script_executor=stub)

    message, output = await tool._arun(
        script="print('ok')",
        script_name=script_name,
        arguments={"MixedCase": 0.5},
        runtime=make_runtime(),
    )

    assert message == "script output"
    assert output == "script output"
    assert stub.calls == [(script_name, "print('ok')", {"mixedcase": 0.5}, 30)]


@pytest.mark.asyncio
async def test_arun_raises_tool_exception_for_blank_script_name() -> None:
    stub = _StubExecutor()
    tool = RunPythonScriptTool(script_executor=stub)

    with pytest.raises(ToolException, match="script_name must be a non-empty string"):
        await tool._arun(
            script="print('ok')",
            script_name="   ",
            arguments=None,
            runtime=make_runtime(),
        )


@pytest.mark.asyncio
async def test_arun_returns_error_output_when_script_fails() -> None:
    tool = RunPythonScriptTool(script_executor=_FailingExecutor())

    result = await tool._arun(
        script="print('fail')",
        script_name="inline_script.py",
        arguments=None,
        runtime=make_runtime(),
    )
    assert result[0] == "inline boom"
    assert result[1] == ""
