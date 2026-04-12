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


class SaveSkillScriptInput(BaseModel):
    """Input schema for the save_skill_script tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    skill_name: str = Field(
        description="Name of the skill to save the script for (e.g., 'my-custom-skill').",
    )
    script_name: str = Field(
        description="Name of the script file (e.g., 'analyze.py', 'process.py').",
    )
    content: str = Field(
        description="Full content of the script file (Python source code).",
    )
    runtime: ToolRuntime


class SaveSkillScriptTool(BaseTool):
    """LangChain tool that saves a script to a MongoDB-stored skill."""

    name: str = "save_skill_script"
    description: str = (
        "Save a script file for a skill. Scripts are executable programs that "
        "perform actions (API calls, file operations), process data, or generate "
        "outputs. The script will be associated with the specified skill for the current user."
    )
    args_schema: Type[BaseModel] = SaveSkillScriptInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    mongo_skill_loader: Optional[UserSkillStore] = None

    @override
    def _run(
        self,
        *,
        skill_name: str,
        script_name: str,
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
        script_name: str,
        content: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = ctx.get("user_id", "")
        stripped_user_id = user_id.strip() if user_id else ""
        if not stripped_user_id:
            raise ToolException("user_id is required for save_skill_script")
        if not skill_name or not skill_name.strip():
            raise ToolException("skill_name must be a non-empty string.")
        if not script_name or not script_name.strip():
            raise ToolException("script_name must be a non-empty string.")
        if not content or not content.strip():
            raise ToolException("content must be a non-empty string.")
        if self.mongo_skill_loader is None:
            raise ToolException("mongo_skill_loader is not configured.")

        try:
            doc = await self.mongo_skill_loader.save_script(
                user_id=stripped_user_id,
                skill_name=skill_name,
                script_name=script_name,
                content=content,
            )
            message = f"Script '{doc.script_name}' saved for skill '{doc.skill_name}'."
            logger.info("SaveSkillScriptTool: %s (user=%s)", message, user_id)
            return message, message
        except Exception as exc:
            logger.exception(
                "SaveSkillScriptTool failed for skill_name=%s script_name=%s user=%s",
                skill_name,
                script_name,
                user_id,
            )
            raise ToolException(
                f"Unable to save script '{script_name}' for skill '{skill_name}' due to an internal error."
            ) from exc

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        skill_name = str(tool_input.get("skill_name", "")) if tool_input else ""
        script_name = str(tool_input.get("script_name", "")) if tool_input else ""
        return (
            f"Save Script: {script_name} ({skill_name})"
            if script_name
            else "Save Script"
        )
