from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, Optional, Tuple, Type, override

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, InjectedToolArg, ToolException
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.loaders.user_skill_store import (
    UserSkillStore,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class ToggleSkillSharingInput(BaseModel):
    """Input schema for the toggle_skill_sharing tool."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        description="Name of the skill to toggle sharing for.",
    )
    shared: bool = Field(
        description="True to share the skill with all users, False to make it private.",
    )


class ToggleSkillSharingTool(BaseTool):
    """LangChain tool that toggles a skill between shared and private."""

    name: str = "toggle_skill_sharing"
    description: str = (
        "Toggle whether a saved skill is shared with all users or private to the owner. "
        "The skill must already exist."
    )
    args_schema: Type[BaseModel] = ToggleSkillSharingInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[UserSkillStore] = None

    @override
    def _run(
        self,
        *,
        skill_name: str,
        shared: bool,
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
        shared: bool,
        user_id: Annotated[str, InjectedToolArg],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        stripped_user_id = user_id.strip() if user_id else ""
        if not stripped_user_id:
            raise ToolException("user_id is required for toggle_skill_sharing")
        if not skill_name or not skill_name.strip():
            raise ToolException("skill_name must be a non-empty string.")
        if self.mongo_skill_loader is None:
            raise ToolException("mongo_skill_loader is not configured.")

        try:
            doc = await self.mongo_skill_loader.set_skill_shared(
                user_id=stripped_user_id,
                skill_name=skill_name,
                shared=shared,
            )
            state = "shared" if doc.shared else "private"
            message = f"Skill '{doc.skill_name}' is now {state}."
            logger.info(
                "ToggleSkillSharingTool: %s (user=%s)", message, stripped_user_id
            )
            return message, message
        except Exception as exc:
            logger.exception(
                "ToggleSkillSharingTool failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise ToolException(
                f"Unable to update sharing for skill '{skill_name}' "
                f"due to an internal error."
            ) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        return (
            f"Toggle Skill Sharing: {skill_name}"
            if skill_name
            else "Toggle Skill Sharing"
        )
