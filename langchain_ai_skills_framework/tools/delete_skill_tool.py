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


class DeleteSkillInput(BaseModel):
    """Input schema for the delete_skill tool."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        description="Name of the skill to delete.",
    )


class DeleteSkillTool(BaseTool):
    """LangChain tool that deletes a user-saved skill from MongoDB."""

    name: str = "delete_skill"
    description: str = (
        "Delete a previously saved skill for the current user. "
        "This only affects the current user's skills."
    )
    args_schema: Type[BaseModel] = DeleteSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[UserSkillStore] = None

    @override
    def _run(
        self,
        *,
        skill_name: str,
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
        runtime: ToolRuntime[dict[str, Any], Any],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        user_id = (runtime.context or {}).get("user_id", "") if runtime else ""
        stripped_user_id = user_id.strip() if user_id else ""
        if not stripped_user_id:
            raise ToolException("user_id is required for delete_skill")
        if not skill_name or not skill_name.strip():
            raise ToolException("skill_name must be a non-empty string.")
        if self.mongo_skill_loader is None:
            raise ToolException("mongo_skill_loader is not configured.")

        try:
            deleted = await self.mongo_skill_loader.delete_skill(
                user_id=stripped_user_id,
                skill_name=skill_name,
            )
            if deleted:
                message = f"Skill '{skill_name}' deleted successfully."
            else:
                message = f"Skill '{skill_name}' not found — nothing to delete."
            logger.info("DeleteSkillTool: %s (user=%s)", message, user_id)
            return message, message
        except Exception as exc:
            logger.exception(
                "DeleteSkillTool failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise ToolException(
                f"Unable to delete skill '{skill_name}' due to an internal error."
            ) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        return f"Delete Skill: {skill_name}" if skill_name else "Delete Skill"
