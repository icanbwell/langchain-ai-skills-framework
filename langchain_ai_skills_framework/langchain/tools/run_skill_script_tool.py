from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.services.run_skill_script_service import RunSkillScriptService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.text_humanizer import Humanizer


class RunSkillScriptInput(BaseModel):
    """Input schema for the run_skill_script tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str = Field(
        description="Name of the plugin containing the skill.",
    )
    skill_name: str = Field(
        description="Name of the skill containing the script.",
    )
    script_name: str = Field(
        description=(
            """Exact name of the script as listed in the skill.
            Usually includes .py extension: "analyze.py", "process.py"
            Must match exactly - do not infer or guess."""
        ),
    )
    arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            """Optional dictionary of arguments to pass to the script when executing.
            The keys and values should match what the script expects.
            For example, if the script is designed to take parameters like {"input_file": "data.csv", "threshold": 0.5}, you would provide those here."""
        ),
    )
    timeout: int = Field(description="Timeout for the script execution in seconds.", default=30)
    runtime: ToolRuntime


class RunSkillScriptTool(BaseTool):
    """LangChain tool that executes skill scripts."""

    name: str = "run_skill_script"
    description: str = """Execute a skill script that performs actions or computations.

        Scripts are executable programs provided by skills that can perform actions
        (API calls, file operations), process data (transformations, analysis), or
        generate outputs (reports, visualizations).

        When to use this:
        - When a skill's instructions tell you to run a specific script
        - To perform automated tasks that the skill provides
        - For data processing, API interactions, or file operations

        Important:
        - Get script names from the skill's documentation first
        - Use exact script names - do not modify or guess
        - Check the script's parameter schema for required arguments
        - Review skill instructions before running scripts
        - Scripts may modify external state (files, databases, APIs)
        - Execution errors are included in the output
        """
    args_schema: type[BaseModel] = RunSkillScriptInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    skill_loader: SkillLoaderProtocol

    def _run(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    async def _arun(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = RunSkillScriptService(skill_loader=self.skill_loader)
        try:
            return await service.execute(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
                arguments=arguments,
                timeout=timeout,
            )
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name: str = str(tool_input.get("skill_name") if tool_input else "")
        script_name: str = str(tool_input.get("script_name") if tool_input else "")
        return f"{Humanizer.humanize_tool_name(key=skill_name)} ({script_name})"
