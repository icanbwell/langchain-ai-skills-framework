from __future__ import annotations

from typing import Any, Literal, override

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, ConfigDict

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.list_plugins_service import ListPluginsService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class ListPluginsInput(BaseModel):
    """Input schema for the list_plugins tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    runtime: ToolRuntime


class ListPluginsTool(BaseTool):
    """LangChain tool that lists registered plugins."""

    name: str = "list_plugins"
    description: str = (
        "List all registered plugins. Returns plugin names, descriptions, and the skills each plugin provides."
    )
    args_schema: type[BaseModel] = ListPluginsInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: PluginSkillStore | None = None

    @override
    def _run(
        self,
        *,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    @override
    async def _arun(
        self,
        *,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        service = ListPluginsService(mongo_skill_loader=self.mongo_skill_loader)
        try:
            plugins = await service.execute()
            message = ListPluginsService.format_as_text(plugins)
            return message, message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        return "List Plugins"
