from __future__ import annotations

from typing import Any, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)


class ReadSkillResourceInput(BaseModel):
    """Input schema for the load_skill tool."""

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


class ReadSkillResourceTool(StructuredTool):
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
        *args: Any,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        skill_name = self._resolve_skill_name(args=args, kwargs=kwargs)
        resource_name = self._resolve_resource_name(args=args, kwargs=kwargs)
        return self._load_skill_resource(
            skill_name=skill_name, resource_name=resource_name
        )

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        skill_name = self._resolve_skill_name(args=args, kwargs=kwargs)
        resource_name = self._resolve_resource_name(args=args, kwargs=kwargs)
        return self._load_skill_resource(
            skill_name=skill_name, resource_name=resource_name
        )

    @staticmethod
    def _resolve_skill_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_skill_name = kwargs.get("skill_name", args[0] if args else "")
        return raw_skill_name if isinstance(raw_skill_name, str) else ""

    @staticmethod
    def _resolve_resource_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_resource_name = kwargs.get("resource_name", args[0] if args else "")
        return raw_resource_name if isinstance(raw_resource_name, str) else ""

    def _load_skill_resource(self, *, skill_name: str, resource_name: str) -> str:
        normalized_name = skill_name.strip()
        if not normalized_name:
            return self._format_availability_message(self.skill_loader, normalized_name)

        try:
            resource = self.skill_loader.read_skill_resource(
                skill_name=normalized_name, resource_name=resource_name
            )
            return resource
        except SkillNotFoundError:
            return self._format_availability_message(self.skill_loader, normalized_name)

    @staticmethod
    def _format_availability_message(
        loader: SkillLoaderProtocol, normalized_name: str
    ) -> str:
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
