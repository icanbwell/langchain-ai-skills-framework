from __future__ import annotations

import logging
from typing import Type, Literal, Tuple, Any
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langgraph.prebuilt.tool_node import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS
from langchain_ai_skills_framework.utilities.text_humanizer import Humanizer

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class LoadSkillInput(BaseModel):
    """Input schema for the load_skill tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    skill_name: str = Field(
        description="Name of the skill to load (e.g., 'sales_analytics').",
    )
    runtime: ToolRuntime


class LoadSkillTool(BaseTool):
    """LangChain tool that loads full skill definitions for the agent."""

    name: str = "load_skill"
    description: str = (
        "Load the full content of a skill into the agent's context for detailed"
        " handling instructions, policies, and guidelines."
    )
    args_schema: Type[BaseModel] = LoadSkillInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    skill_loader: SkillLoaderProtocol

    def _run(
        self,
        *,
        skill_name: str,
        runtime: ToolRuntime,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError(
            "Synchronous execution is not supported. Use the asynchronous method instead."
        )

    async def _arun(
        self,
        *,
        skill_name: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        """Asynchronously load a skill by name."""
        if not isinstance(skill_name, str):
            raise ToolException("Skill name must be a string.")

        normalized_name = skill_name.strip()
        if not normalized_name:
            raise ToolException(
                await self._format_availability_message(
                    self.skill_loader, normalized_name, runtime=runtime
                )
            )

        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = ctx.get("user_id", "")
        stripped_user_id = user_id.strip() if user_id else ""

        skill = await self._load_skill(
            normalized_name, user_id=stripped_user_id, runtime=runtime
        )
        logger.debug("LoadSkillTool (async): loaded skill_name=%s", normalized_name)
        return skill, skill

    async def _load_skill(
        self, skill_name: str, *, user_id: str, runtime: ToolRuntime
    ) -> str:
        """Load skill content checking user skills first, then shared."""
        normalized_name = skill_name.strip()

        if not normalized_name:
            raise ToolException(
                await self._format_availability_message(
                    self.skill_loader, normalized_name, runtime=runtime
                )
            )

        try:
            if user_id:
                skill = await self.skill_loader.get_skill_details_for_user(
                    user_id=user_id, skill_name=normalized_name
                )
            else:
                skill = self.skill_loader.get_skill_details(skill_name=normalized_name)
            return f"{skill.content}"
        except SkillNotFoundError:
            return await self._format_availability_message(
                self.skill_loader, normalized_name, runtime=runtime
            )
        except Exception as exc:
            logger.exception("LoadSkillTool failed for skill_name=%s", normalized_name)
            raise ToolException(
                f"Unable to load skill '{normalized_name}' due to an internal error."
            ) from exc

    @staticmethod
    async def _format_availability_message(
        loader: SkillLoaderProtocol,
        normalized_name: str,
        *,
        runtime: ToolRuntime,
    ) -> str:
        """Format a message showing available skills."""
        ctx: dict[str, Any] = runtime.context or {} if runtime else {}
        user_id = ctx.get("user_id", "")
        stripped_user_id = user_id.strip() if user_id else ""

        if stripped_user_id:
            summaries = await loader.list_all_summaries(
                user_id=stripped_user_id, allowed_skills=set()
            )
            available_names = sorted(s.name for s in summaries)
        else:
            available_names = sorted(
                summary.name
                for summary in loader.list_skill_summaries(allowed_skills=set())
            )
        available = ", ".join(available_names)

        availability_message = (
            f"Skill '{normalized_name}' not found."
            if normalized_name
            else "No skill name provided."
        )

        return (
            f"{availability_message} Available skills: {available or 'None configured'}"
        )

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        """Get the friendly name of the skill."""
        skill_name: str = str(tool_input.get("skill_name") if tool_input else "")
        return f"{Humanizer.humanize_tool_name(key=skill_name)}"
