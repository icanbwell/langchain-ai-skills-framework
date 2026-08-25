from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.services.load_skill_service import LoadSkillService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.text_humanizer import Humanizer


class LoadSkillInput(BaseModel):
    """Input schema for the load_skill tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str = Field(
        description="Name of the plugin containing the skill.",
    )
    skill_name: str = Field(
        description="Name of the skill to load (e.g., 'sales_analytics').",
    )
    runtime: ToolRuntime


class LoadSkillTool(BaseTool):
    """LangChain tool that loads full skill definitions for the agent."""

    name: str = "load_skill"
    description: str = (
        "Load the full content of a skill into the agent's context for detailed"
        " handling instructions, policies, and guidelines."
    )
    args_schema: type[BaseModel] = LoadSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    skill_loader: SkillLoaderProtocol
    user_skill_store: PluginSkillStore | None = None

    def _run(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    async def _arun(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = LoadSkillService(
            skill_loader=self.skill_loader,
            user_skill_store=self.user_skill_store,
        )
        try:
            content = await service.execute(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)
            return content, content
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name: str = str(tool_input.get("skill_name") if tool_input else "")
        return f"{Humanizer.humanize_tool_name(key=skill_name)}"
