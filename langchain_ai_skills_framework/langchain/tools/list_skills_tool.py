from __future__ import annotations

from typing import Any, Literal, override

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
from langchain_ai_skills_framework.services.list_skills_service import ListSkillsService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class ListSkillsInput(BaseModel):
    """Input schema for the list_skills tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str | None = Field(
        default=None,
        description="Optional plugin name to filter skills by. If not provided, lists skills from all plugins.",
    )
    folder: str | None = Field(
        default=None,
        description="Optional folder path to filter skills by. If not provided, lists skills from all folders.",
    )
    runtime: ToolRuntime


class ListSkillsTool(BaseTool):
    """LangChain tool that lists available skills for the current user."""

    name: str = "list_skills"
    description: str = (
        "List all skills available to the current user, including both shared skills "
        "and the user's own saved skills. Returns skill names and descriptions."
    )
    args_schema: type[BaseModel] = ListSkillsInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    skill_loader: SkillLoaderProtocol

    @override
    def _run(
        self,
        *,
        plugin_name: str | None = None,
        folder: str | None = None,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    @override
    async def _arun(
        self,
        *,
        plugin_name: str | None = None,
        folder: str | None = None,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = ListSkillsService(skill_loader=self.skill_loader)
        try:
            skills = await service.execute(user_id=user_id, plugin_name=plugin_name, folder=folder)
            message = ListSkillsService.format_as_text(skills)
            return message, message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        return "List Skills"
