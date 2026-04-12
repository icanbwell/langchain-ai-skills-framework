from __future__ import annotations

import logging
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
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class SaveSkillInput(BaseModel):
    """Input schema for the save_skill tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    skill_name: str = Field(
        description="Name of the skill to save (e.g., 'my-custom-skill').",
    )
    content: str = Field(
        description=(
            "Full content of the skill in SKILL.md format. "
            "May include YAML frontmatter with description and metadata."
        ),
    )
    runtime: ToolRuntime


class SaveSkillTool(BaseTool):
    """LangChain tool that saves a skill to MongoDB for the current user."""

    name: str = "save_skill"
    description: str = (
        "Save a new skill or update an existing one for the current user. "
        "The skill will only be available to the user who saved it. "
        "Content should follow the SKILL.md format with optional YAML frontmatter."
    )
    args_schema: Type[BaseModel] = SaveSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[UserSkillStore] = None

    @override
    def _run(
        self,
        *,
        skill_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError(
            "Synchronous execution is not supported. Use the asynchronous method instead."
        )

    @override
    async def _arun(
        self,
        *,
        skill_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = ctx.get("user_id", "")
        stripped_user_id = user_id.strip() if user_id else ""
        if not stripped_user_id:
            raise ToolException("user_id is required for save_skill")
        if not skill_name or not skill_name.strip():
            raise ToolException("skill_name must be a non-empty string.")
        if not content or not content.strip():
            raise ToolException("content must be a non-empty string.")
        if self.mongo_skill_loader is None:
            raise ToolException("mongo_skill_loader is not configured.")

        try:
            doc = await self.mongo_skill_loader.save_skill(
                user_id=stripped_user_id,
                skill_name=skill_name,
                content=content,
            )
            message = f"Skill '{doc.skill_name}' saved successfully."
            logger.info("SaveSkillTool: %s (user=%s)", message, user_id)
            return message, message
        except Exception as exc:
            logger.exception(
                "SaveSkillTool failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise ToolException(
                f"Unable to save skill '{skill_name}' due to an internal error."
            ) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        return f"Save Skill: {skill_name}" if skill_name else "Save Skill"
