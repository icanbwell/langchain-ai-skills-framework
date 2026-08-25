from __future__ import annotations

from typing import Any, Protocol

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)


class ScriptExecutorProtocol(Protocol):
    """Runs inline skill script content and returns the result.

    A script's own failure (non-zero exit, an exception raised inside the
    script) is reported through ``MyScriptExecutionResult.success``/
    ``stderr``, not by raising. Raising is reserved for the executor
    failing to run the script at all (infra unreachable, malformed
    arguments) — callers such as ``CompositeSkillLoader`` let those
    propagate as-is.
    """

    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
    ) -> MyScriptExecutionResult: ...
