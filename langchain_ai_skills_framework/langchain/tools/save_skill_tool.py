from __future__ import annotations

from typing import Any, Literal, override

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
from langchain_ai_skills_framework.services.save_skill_service import SaveSkillService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class SaveSkillInput(BaseModel):
    """Input schema for the save_skill tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str = Field(
        description="Name of the plugin to save the skill to.",
    )
    skill_name: str = Field(
        description="Name of the skill to save (e.g., 'my-custom-skill').",
    )
    content: str = Field(
        description=(
            "Full content of the skill in SKILL.md format. May include YAML frontmatter with description and metadata."
        ),
    )
    folder: str | None = Field(
        default=None,
        description="Optional subfolder path within the plugin to organize skills.",
    )
    runtime: ToolRuntime


class SaveSkillTool(BaseTool):
    """LangChain tool that saves a skill to MongoDB for the current user."""

    name: str = "save_skill"
    description: str = (
        "Create a new skill or update an existing one for the current user. "
        "If a skill with the same name already exists, its content will be replaced. "
        "The skill will only be available to the user who saved it unless shared. "
        "Content should follow the SKILL.md format with optional YAML frontmatter."
    )
    args_schema: type[BaseModel] = SaveSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: PluginSkillStore | None = None

    @override
    def _run(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        content: str,
        folder: str | None = None,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    @override
    async def _arun(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        content: str,
        folder: str | None = None,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = SaveSkillService(mongo_skill_loader=self.mongo_skill_loader)
        try:
            result = await service.execute(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                content=content,
                folder=folder,
            )
            return result.message, result.message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        return f"Save Skill: {skill_name}" if skill_name else "Save Skill"
