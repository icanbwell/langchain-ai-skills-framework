from __future__ import annotations
import asyncio
import logging
from typing import Any, Type, Literal
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field
from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import MyScriptExecutor
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class RunPythonScriptInput(BaseModel):
    """Input schema for the run_python_script tool."""

    model_config = ConfigDict(extra="forbid")

    script: str = Field(
        description=(
            """Python script content to execute.
            The full script body is executed in the configured runtime context."""
        ),
    )
    script_name: str = Field(
        description=(
            "Name to identify the script being executed (e.g., 'data_processing.py')."
        ),
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            """Optional dictionary of arguments to pass to the script when executing.
            The keys and values should match what the script expects."""
        ),
    )
    timeout: int = Field(
        description="Timeout for the script execution in seconds.", default=30
    )


class RunPythonScriptOutput(BaseModel):
    """Structured output schema for the run_python_script tool."""

    model_config = ConfigDict(extra="allow")

    success: bool = Field(description="Whether the script executed successfully")
    stdout: str | None = Field(
        default=None, description="Standard output from the script"
    )
    stderr: str | None = Field(
        default=None, description="Standard error from the script"
    )
    exit_code: int | None = Field(
        default=None, description="Exit code from the script execution"
    )
    error_message: str | None = Field(
        default=None, description="Human-readable error message if failed"
    )


class RunPythonScriptTool(BaseTool):  # Changed from StructuredTool
    """LangChain tool that executes inline Python script content within a runtime context."""

    name: str = "run_python_script"
    description: str = """Execute inline script content within a skill's execution context.
        This tool runs Python script content provided at runtime and executes it
        using the selected runtime context and configuration.

        When to use this:
        - When you need to execute generated script content in the configured context
        - When a workflow provides script text directly instead of a script file

        Important:
        - The script content is executed as provided
        - Script supports PEP 723 inline metadata to define dependencies
        - Keep script content minimal and task-specific
        - Review skill instructions before running scripts
        - Scripts may modify external state (files, databases, APIs)
        - Execution errors are included in the output
        """
    args_schema: Type[BaseModel] = RunPythonScriptInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def _run(
        self,
        *,
        script: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        """Synchronous execution with named parameters."""
        return asyncio.run(
            self._arun(
                script=script,
                arguments=arguments,
                timeout=timeout,
                script_name=script_name,
            )
        )

    async def _arun(
        self,
        *,
        script: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        """Async execution with named parameters."""
        if not isinstance(script_name, str):
            raise ToolException("Script name must be a string.")
        if arguments is not None and not isinstance(arguments, dict):
            raise ToolException("Arguments must be a dict.")
        if not isinstance(timeout, int):
            raise ToolException("Timeout must be an int.")
        if not isinstance(script_name, str):
            raise ToolException("Script name must be a string.")

        logger.debug(
            "RunPythonScriptTool: Running inline script script_name=%s argument_keys=%s timeout=%s",
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

            if not result.success:
                raise ToolException(
                    f"Inline script failed with exit code {result.exit_code}. "
                    f"Error: {result.stderr or 'Unknown error'}"
                )

            # Create structured script_result
            script_result = RunPythonScriptOutput(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                error_message=None if result.success else result.stderr,
            )

            logger.debug(
                "RunPythonScriptTool: Output from Python script with arguments %s\n%s",
                arguments,
                script_result,
            )

            if script_result.success:
                return (
                    script_result.stdout or "No script_result",
                    script_result.stdout or "",
                )
            else:
                return (
                    script_result.stderr or script_result.stdout or "No script_result",
                    script_result.stdout or "",
                )

        except ToolException:
            raise
        except Exception as exc:
            logger.exception(
                "RunPythonScriptTool: Error running inline Python script",
            )
            raise ToolException(f"Error running inline Python script: {exc}") from exc

    async def _run_inline_script(
        self,
        *,
        script: str,
        script_name: str,
        arguments: dict[str, Any] | None,
        timeout: int = 30,
    ) -> MyScriptExecutionResult:
        """Execute the script using MyScriptExecutor."""
        resolved_script_name = script_name.strip()
        if not resolved_script_name:
            raise ToolException("script_name must be a non-empty string")

        normalized_arguments = {k.lower(): v for k, v in (arguments or {}).items()}

        executor = MyScriptExecutor()
        result: MyScriptExecutionResult = await executor.execute_inline_script(
            script_name=resolved_script_name,
            script=script,
            arguments=normalized_arguments,
            timeout=timeout,
        )
        return result

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        """Get the friendly name of the skill."""
        script_name: str = str(tool_input.get("script_name") if tool_input else "")
        return f"Python Script ({script_name})"
