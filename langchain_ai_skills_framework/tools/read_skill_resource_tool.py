from __future__ import annotations

import logging
from typing import Type

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

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class ReadSkillResourceInput(BaseModel):
    """Input schema for the read_skill_resource tool."""

    model_config = ConfigDict(extra="forbid")

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
    skill_loader: SkillLoaderProtocol

    def _run(
        self,
        skill_name: str,
        resource_name: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Synchronously load a skill resource."""
        return self._load_skill_resource(
            skill_name=skill_name, resource_name=resource_name
        )

    async def _arun(
        self,
        skill_name: str,
        resource_name: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        """Asynchronously load a skill resource."""
        resource = self._load_skill_resource(
            skill_name=skill_name, resource_name=resource_name
        )
        logger.debug(
            "ReadSkillResourceTool: Loaded resource_name=%s from skill_name=%s",
            resource_name,
            skill_name,
        )
        return resource

    def _load_skill_resource(self, *, skill_name: str, resource_name: str) -> str:
        """Load resource content and raise when a skill cannot be resolved."""
        normalized_name = skill_name.strip()

        if not normalized_name:
            raise ToolException(
                self._format_availability_message(self.skill_loader, normalized_name)
            )

        try:
            resource = self.skill_loader.read_skill_resource(
                skill_name=normalized_name, resource_name=resource_name
            )
            return resource
        except SkillNotFoundError as exc:
            raise ToolException(
                self._format_availability_message(self.skill_loader, normalized_name)
            ) from exc
        except Exception as exc:
            logger.exception(
                "ReadSkillResourceTool failed for skill_name=%s resource_name=%s",
                normalized_name,
                resource_name,
            )
            raise ToolException(
                f"Unable to read resource '{resource_name}' from skill '{normalized_name}' due to an internal error."
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
