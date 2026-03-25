from __future__ import annotations
import logging
from typing import Type
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

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
    skill_loader: SkillLoaderProtocol

    def _run(
        self,
        skill_name: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Synchronously load a skill by name."""
        skill = self._load_skill(skill_name)
        logger.debug(f"LoadSkillTool (sync): {skill_name}\n{skill}")
        return skill

    async def _arun(
        self,
        skill_name: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Asynchronously load a skill by name."""
        skill = self._load_skill(skill_name)
        logger.debug(f"LoadSkillTool: {skill_name}\n{skill}")
        return skill

    def _load_skill(self, skill_name: str) -> str:
        """Load skill content or return availability message."""
        normalized_name = skill_name.strip()

        if not normalized_name:
            return self._format_availability_message(self.skill_loader, normalized_name)

        try:
            skill = self.skill_loader.get_skill_details(skill_name=normalized_name)
            return f"Loaded skill: {skill.name}\n\n{skill.content}"
        except SkillNotFoundError:
            return self._format_availability_message(self.skill_loader, normalized_name)

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
