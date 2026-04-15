from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Type, override

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.loaders.user_skill_store import (
    UserSkillStore,
)
from langchain_ai_skills_framework.services.save_skill_resource_service import SaveSkillResourceService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError


class SaveSkillResourceInput(BaseModel):
    """Input schema for the save_skill_resource tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    skill_name: str = Field(
        description="Name of the skill to save the resource for (e.g., 'my-custom-skill').",
    )
    resource_name: str = Field(
        description="Name of the resource file (e.g., 'FORMS.md', 'REFERENCE.md').",
    )
    content: str = Field(
        description="Full content of the resource file.",
    )
    runtime: ToolRuntime


class SaveSkillResourceTool(BaseTool):
    """LangChain tool that saves a resource to a MongoDB-stored skill."""

    name: str = "save_skill_resource"
    description: str = (
        "Create or update a resource file for a skill. Resources are supplementary files "
        "like documentation, templates, or schemas that support skill execution. "
        "If a resource with the same name already exists, its content will be replaced. "
        "The resource will be associated with the specified skill for the current user."
    )
    args_schema: Type[BaseModel] = SaveSkillResourceInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[UserSkillStore] = None

    @override
    def _run(
        self,
        *,
        skill_name: str,
        resource_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    @override
    async def _arun(
        self,
        *,
        skill_name: str,
        resource_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = SaveSkillResourceService(mongo_skill_loader=self.mongo_skill_loader)
        try:
            message = await service.execute(
                user_id=user_id,
                skill_name=skill_name,
                resource_name=resource_name,
                content=content,
            )
            return message, message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        resource_name = str(tool_input.get("resource_name", "")) if tool_input else ""
        return f"Save Resource: {resource_name} ({skill_name})" if resource_name else "Save Resource"
