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
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.services.publish_skill_service import PublishSkillService


class PublishSkillInput(BaseModel):
    """Input schema for the publish_skill tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    plugin_name: str = Field(
        description="Name of the plugin containing the skill.",
    )
    skill_name: Optional[str] = Field(
        default=None,
        description="Name of the skill to publish or unpublish. If omitted, extracted from content frontmatter.",
    )
    content: Optional[str] = Field(
        default=None,
        description="Skill content (SKILL.md format). Used to extract skill_name from frontmatter when skill_name is not provided.",
    )
    published: bool = Field(
        description="True to publish the skill to the marketplace, False to unpublish it.",
    )
    branch_name: Optional[str] = Field(
        default=None,
        description="Optional branch name for the PR. Defaults to 'skill-publish/{plugin}/{skill}'.",
    )
    runtime: ToolRuntime


class PublishSkillTool(BaseTool):
    """LangChain tool that publishes or unpublishes a skill in the marketplace."""

    name: str = "publish_skill"
    description: str = "Publish or unpublish a saved skill to the marketplace. The skill must already exist."
    args_schema: Type[BaseModel] = PublishSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[PluginSkillStore] = None
    marketplace_publisher: Optional[GitHubMarketplacePublisher] = None

    @override
    def _run(
        self,
        *,
        plugin_name: str,
        skill_name: Optional[str] = None,
        content: Optional[str] = None,
        published: bool,
        branch_name: Optional[str] = None,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError("Synchronous execution is not supported. Use the asynchronous method instead.")

    @override
    async def _arun(
        self,
        *,
        plugin_name: str,
        skill_name: Optional[str] = None,
        content: Optional[str] = None,
        published: bool,
        branch_name: Optional[str] = None,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = (ctx.get("user_id", "") or "").strip()

        service = PublishSkillService(
            mongo_skill_loader=self.mongo_skill_loader,
            marketplace_publisher=self.marketplace_publisher,
        )
        try:
            message = await service.execute(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                content=content,
                published=published,
                branch_name=branch_name,
            )
            return message, message
        except SkillOperationError as exc:
            raise ToolException(str(exc)) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        return f"Publish Skill: {skill_name}" if skill_name else "Publish Skill"
