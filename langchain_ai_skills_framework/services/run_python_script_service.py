from __future__ import annotations

import logging
from typing import Any

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.script_executor_protocol import (
    ScriptExecutorProtocol,
)
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class RunPythonScriptService:
    """Execute inline Python script content."""

    def __init__(self, *, script_executor: ScriptExecutorProtocol) -> None:
        self._script_executor = script_executor

    async def execute(
        self,
        *,
        script: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> tuple[str, str]:
        """Run the inline script and return ``(content, artifact)``.

        Raises ``SkillOperationError`` on failure.
        """
        logger.debug(
            "RunPythonScriptService: Running inline script script_name=%s argument_keys=%s timeout=%s",
            script_name,
            sorted((arguments or {}).keys()),
            timeout,
        )

        try:
            result = await self._run_inline_script(
                script=script,
                script_name=script_name,
                arguments=arguments,
                timeout=timeout,
            )

            logger.debug(
                "RunPythonScriptService: Output from Python script with arguments %s\n%s",
                arguments,
                result,
            )

            if result.success:
                return (
                    result.stdout or "No output",
                    result.stdout or "",
                )
            else:
                return (
                    result.stderr or result.stdout or "No output",
                    result.stdout or "",
                )
        except SkillOperationError:
            raise
        except Exception as exc:
            logger.exception(
                "RunPythonScriptService: Error running inline Python script",
            )
            raise SkillOperationError(f"Error running inline Python script: {exc}") from exc

    async def _run_inline_script(
        self,
        *,
        script: str,
        script_name: str,
        arguments: dict[str, Any] | None,
        timeout: int = 30,
    ) -> MyScriptExecutionResult:
        resolved_script_name = script_name.strip()
        if not resolved_script_name:
            raise SkillOperationError("script_name must be a non-empty string")

        normalized_arguments = {k.lower(): v for k, v in (arguments or {}).items()}

        result: MyScriptExecutionResult = await self._script_executor.execute_inline_script(
            script_name=resolved_script_name,
            script=script,
            arguments=normalized_arguments,
            timeout=timeout,
        )
        return result
