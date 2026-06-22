from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)


@runtime_checkable
class ScriptExecutorProtocol(Protocol):
    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
    ) -> MyScriptExecutionResult: ...
