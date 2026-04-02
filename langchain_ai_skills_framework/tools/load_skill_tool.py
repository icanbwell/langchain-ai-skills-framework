from __future__ import annotations

import asyncio
import logging
from typing import Type, Literal, Tuple, Any
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
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

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        description="Name of the skill to load (e.g., 'sales_analytics').",
    )


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
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        """Synchronously load a skill by name."""
        return asyncio.run(
            self._arun(
                skill_name=skill_name,
            )
        )

    async def _arun(
        self,
        *,
        skill_name: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        """Asynchronously load a skill by name."""
        if not isinstance(skill_name, str):
            raise ToolException("Skill name must be a string.")

        normalized_name = skill_name.strip()
        if not normalized_name:
            raise ToolException(
                self._format_availability_message(self.skill_loader, normalized_name)
            )

        skill = self._load_skill(normalized_name)
        logger.debug("LoadSkillTool (async): loaded skill_name=%s", normalized_name)
        return skill, skill

    def _load_skill(self, skill_name: str) -> str:
        """Load skill content and raise when a skill cannot be resolved."""
        normalized_name = skill_name.strip()

        if not normalized_name:
            raise ToolException(
                self._format_availability_message(self.skill_loader, normalized_name)
            )

        try:
            skill = self.skill_loader.get_skill_details(skill_name=normalized_name)
            return f"{skill.content}"
        except SkillNotFoundError:
            return self._format_availability_message(self.skill_loader, normalized_name)
        except Exception as exc:
            logger.exception("LoadSkillTool failed for skill_name=%s", normalized_name)
            raise ToolException(
                f"Unable to load skill '{normalized_name}' due to an internal error."
            ) from exc

    @staticmethod
    def _format_availability_message(
        loader: SkillLoaderProtocol, normalized_name: str
    ) -> str:
        """Format a message showing available skills."""
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
