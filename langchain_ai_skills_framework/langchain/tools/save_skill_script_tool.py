from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Type, override

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.services.save_skill_script_service import SaveSkillScriptService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class SaveSkillScriptInput(BaseModel):
    """Input schema for the save_skill_script tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str = Field(
        description="Name of the plugin containing the skill.",
    )
    skill_name: str = Field(
        description="Name of the skill to save the script for (e.g., 'my-custom-skill').",
    )
    script_name: str = Field(
        description="Name of the script file (e.g., 'analyze.py', 'process.py').",
    )
    content: str = Field(
        description="Full content of the script file (Python source code).",
    )
    runtime: ToolRuntime


class SaveSkillScriptTool(BaseTool):
    """LangChain tool that saves a script to a MongoDB-stored skill."""

    name: str = "save_skill_script"
    description: str = (
        "Create or update a script file for a skill. Scripts are executable programs that "
        "perform actions (API calls, file operations), process data, or generate "
        "outputs. If a script with the same name already exists, its content will be replaced. "
        "The script will be associated with the specified skill for the current user."
    )
    args_schema: Type[BaseModel] = SaveSkillScriptInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[PluginSkillStore] = None

    @override
    def _run(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    @override
    async def _arun(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = SaveSkillScriptService(mongo_skill_loader=self.mongo_skill_loader)
        try:
            result = await service.execute(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
                content=content,
            )
            return result.message, result.message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        script_name = str(tool_input.get("script_name", "")) if tool_input else ""
        return f"Save Script: {script_name} ({skill_name})" if script_name else "Save Script"
