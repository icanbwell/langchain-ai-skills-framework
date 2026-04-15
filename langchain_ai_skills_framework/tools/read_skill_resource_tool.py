from __future__ import annotations

from typing import Type, Literal, Tuple, Any

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
from langchain_ai_skills_framework.services.read_skill_resource_service import ReadSkillResourceService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.text_humanizer import Humanizer


class ReadSkillResourceInput(BaseModel):
    """Input schema for the read_skill_resource tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    skill_name: str = Field(
        description="Name of the skill containing the resource.",
    )
    resource_name: str = Field(
        description=(
            """Exact name of the resource as listed in the skill.
            Examples: "FORMS.md", "REFERENCE.md", "get_schema"
            Must match exactly - do not infer or guess."""
        ),
    )
    runtime: ToolRuntime


class ReadSkillResourceTool(BaseTool):
    """LangChain tool that reads supplementary resources from skills."""

    name: str = "read_skill_resource"
    description: str = """Access supplementary documentation, templates, or data from a skill.

        Resources are additional files that support skill execution. They can be static
        content (markdown docs, templates, schemas) or dynamic callables (functions that
        generate content based on parameters).

        When to use this:
        - When a skill's instructions reference a specific resource
        - To access form templates, reference documentation, or data schemas
        - When you need supplementary information beyond the skill instructions"""
    args_schema: Type[BaseModel] = ReadSkillResourceInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    skill_loader: SkillLoaderProtocol

    def _run(
        self,
        *,
        skill_name: str,
        resource_name: str,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    async def _arun(
        self,
        *,
        skill_name: str,
        resource_name: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = ReadSkillResourceService(skill_loader=self.skill_loader)
        try:
            return await service.execute(
                user_id=user_id,
                skill_name=skill_name,
                resource_name=resource_name,
            )
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name: str = str(tool_input.get("skill_name") if tool_input else "")
        resource_name: str = str(tool_input.get("resource_name") if tool_input else "")
        return f"{Humanizer.humanize_tool_name(key=skill_name)} {Humanizer.humanize_tool_name(key=resource_name)}"
