from __future__ import annotations
import asyncio
import logging
from typing import Any, Type, Literal
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool  # Changed from StructuredTool
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
    arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            """Optional dictionary of arguments to pass to the script when executing.
            The keys and values should match what the script expects."""
        ),
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
    response_format: Literal['content', 'content_and_artifact'] = "content_and_artifact"
    _inline_script_name: str = "inline_script.py"

    def _run(
        self,
        script: str,
        arguments: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, RunPythonScriptOutput]:
        """Synchronous execution with named parameters."""
        return asyncio.run(
            self._arun(
                script=script,
                arguments=arguments,
                run_manager=run_manager,
            )
        )

    async def _arun(
        self,
        script: str,
        arguments: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, RunPythonScriptOutput]:
        """Async execution with named parameters."""
        logger.debug(
            "RunPythonScriptTool: Running Python script with script=%s and arguments %s",
            script,
            arguments,
        )

        try:
            result = await self._run_inline_skill_script(
                script=script,
                arguments=arguments,
            )

            # Create structured output
            output = RunPythonScriptOutput(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                error_message=None if result.success else result.stderr,
            )

            logger.debug(
                "RunPythonScriptTool: Output from Python script with arguments %s\n%s",
                arguments,
                output,
            )

            # Create human-readable summary
            if output.success:
                summary = f"Script executed successfully.\nOutput:\n{output.stdout}"
            else:
                summary = f"Script failed with exit code {output.exit_code}.\nError:\n{output.stderr}"

            return summary, output

        except Exception as exc:
            logger.exception(
                "RunPythonScriptTool: Error running Python script: %s",
                script,
            )
            output = RunPythonScriptOutput(
                success=False,
                stdout=None,
                stderr=str(exc),
                exit_code=-1,
                error_message=f"Error running script: {exc}",
            )
            return f"Error running script: {exc}", output

    async def _run_inline_skill_script(
        self,
        script: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        """Execute the script using MyScriptExecutor."""
        normalized_arguments = {k.lower(): v for k, v in (arguments or {}).items()}

        executor = MyScriptExecutor()
        result: MyScriptExecutionResult = await executor.execute_inline_script(
            script_name=self._inline_script_name,
            script=script,
            arguments=normalized_arguments,
            timeout=30,
        )
        return result
