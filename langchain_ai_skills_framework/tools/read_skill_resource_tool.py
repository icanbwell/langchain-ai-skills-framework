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
    runtime: ToolRuntime = Field(exclude=True)


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
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError(
            "Synchronous execution is not supported. Use the asynchronous method instead."
        )

    async def _arun(
        self,
        *,
        skill_name: str,
        resource_name: str,
        runtime: ToolRuntime,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Tuple[str, str]:
        """Asynchronously load a skill resource."""
        if not isinstance(skill_name, str):
            raise ToolException("Skill name must be a string.")

        normalized_name = skill_name.strip()
        if not normalized_name:
            raise ToolException(
                await self._format_availability_message(
                    self.skill_loader, normalized_name, runtime=runtime
                )
            )

        if not isinstance(resource_name, str):
            return "Resource name must be a string.", ""

        normalized_resource_name = resource_name.strip()
        if not normalized_resource_name:
            return "No resource name provided.", ""

        resource = await self._load_skill_resource(
            skill_name=normalized_name,
            resource_name=normalized_resource_name,
            runtime=runtime,
        )
        logger.debug(
            "ReadSkillResourceTool: Loaded resource_name=%s from skill_name=%s",
            normalized_resource_name,
            normalized_name,
        )
        return resource, resource

    async def _load_skill_resource(
        self, *, skill_name: str, resource_name: str, runtime: ToolRuntime
    ) -> str:
        """Load resource content and raise when a skill cannot be resolved."""
        normalized_name = skill_name.strip()

        if not normalized_name:
            raise ToolException(
                await self._format_availability_message(
                    self.skill_loader, normalized_name, runtime=runtime
                )
            )

        try:
            resource = self.skill_loader.read_skill_resource(
                skill_name=normalized_name, resource_name=resource_name
            )
            return resource
        except SkillNotFoundError:
            return await self._format_availability_message(
                self.skill_loader,
                normalized_name,
                resource_name=resource_name,
                runtime=runtime,
            )
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
    async def _format_availability_message(
        loader: SkillLoaderProtocol,
        normalized_name: str,
        resource_name: str | None = None,
        *,
        runtime: ToolRuntime,
    ) -> str:
        """Format a message showing available skills or resources."""
        if resource_name and normalized_name:
            try:
                loader.get_skill_details(normalized_name)
            except SkillNotFoundError:
                pass
            else:
                resource_names = loader.list_skill_resource_names(normalized_name)
                available_resources = ", ".join(resource_names)
                return (
                    f"Resource '{resource_name}' not found in skill '{normalized_name}'. "
                    f"Available resources: {available_resources or 'none'}"
                )

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

        if normalized_name:
            availability_message = f"Skill '{normalized_name}' not found."
        else:
            availability_message = "No skill name provided."

        return (
            f"{availability_message} Available skills: {available or 'None configured'}"
        )

    @staticmethod
    def get_friendly_name(*, tool_input: dict[str, Any]) -> str:
        """Get the friendly name of the skill."""
        skill_name: str = str(tool_input.get("skill_name") if tool_input else "")
        resource_name: str = str(tool_input.get("resource_name") if tool_input else "")
        return f"{Humanizer.humanize_tool_name(key=skill_name)} {Humanizer.humanize_tool_name(key=resource_name)}"
