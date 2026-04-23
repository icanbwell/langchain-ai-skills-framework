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
from langchain_ai_skills_framework.services.delete_skill_service import DeleteSkillService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class DeleteSkillInput(BaseModel):
    """Input schema for the delete_skill tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str = Field(
        description="Name of the plugin containing the skill.",
    )
    skill_name: str = Field(
        description="Name of the skill to delete.",
    )
    runtime: ToolRuntime


class DeleteSkillTool(BaseTool):
    """LangChain tool that deletes a user-saved skill from MongoDB."""

    name: str = "delete_skill"
    description: str = (
        "Delete a previously saved skill for the current user. This only affects the current user's skills."
    )
    args_schema: Type[BaseModel] = DeleteSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[PluginSkillStore] = None

    @override
    def _run(
        self,
        *,
        plugin_name: str,
        skill_name: str,
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
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = DeleteSkillService(mongo_skill_loader=self.mongo_skill_loader)
        try:
            message = await service.execute(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)
            return message, message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        return f"Delete Skill: {skill_name}" if skill_name else "Delete Skill"
