from __future__ import annotations

import asyncio
from typing import Any, Type, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.services.run_python_script_service import RunPythonScriptService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class RunPythonScriptInput(BaseModel):
    """Input schema for the run_python_script tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    script: str = Field(
        description=(
            """Python script content to execute.
            The full script body is executed in the configured runtime context."""
        ),
    )
    script_name: str = Field(
        description=("Name to identify the script being executed (e.g., 'data_processing.py')."),
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            """Optional dictionary of arguments to pass to the script when executing.
            The keys and values should match what the script expects."""
        ),
    )
    timeout: int = Field(description="Timeout for the script execution in seconds.", default=30)
    runtime: ToolRuntime = Field(exclude=True)


class RunPythonScriptTool(BaseTool):
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
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        return asyncio.run(
            self._arun(
                script=script,
                arguments=arguments,
                timeout=timeout,
                script_name=script_name,
                runtime=runtime,
            )
        )

    async def _arun(
        self,
        *,
        script: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        service = RunPythonScriptService()
        try:
            return await service.execute(
                script=script,
                script_name=script_name,
                arguments=arguments,
                timeout=timeout,
            )
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = tool_input.get("skill_name") if tool_input else None
        return f"{skill_name}"
