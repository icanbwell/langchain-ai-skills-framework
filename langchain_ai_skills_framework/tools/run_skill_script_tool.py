from __future__ import annotations

from typing import Any, Type

import anyio
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)


class RunSkillScriptInput(BaseModel):
    """Input schema for the load_skill tool."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        description="Name of the skill containing the resource.",
    )

    script_name: str = Field(
        description=(
            """Exact name of the script as listed in the skill.
                    Usually includes .py extension: "analyze.py", "process.py"
                    Must match exactly - do not infer or guess."""
        ),
    )

    arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            """Optional dictionary of arguments to pass to the script when executing.
                    The keys and values should match what the script expects.
                    For example, if the script is designed to take parameters like {"input_file": "data.csv", "threshold": 0.5}, you would provide those here."""
        ),
    )


class RunSkillScriptTool(StructuredTool):
    """LangChain tool that loads full skill definitions for the agent."""

    name: str = "run_skill_script"
    description: str = """Execute a skill script that performs actions or computations.

            Scripts are executable programs provided by skills that can perform actions
            (API calls, file operations), process data (transformations, analysis), or
            generate outputs (reports, visualizations).

            When to use this:
            - When a skill's instructions tell you to run a specific script
            - To perform automated tasks that the skill provides
            - For data processing, API interactions, or file operations

            Important:
            - Get script names from the skill's documentation first
            - Use exact script names - do not modify or guess
            - Check the script's parameter schema for required arguments
            - Review skill instructions before running scripts
            - Scripts may modify external state (files, databases, APIs)
            - Execution errors are included in the output
            """
    args_schema: Type[BaseModel] = RunSkillScriptInput
    skill_loader: SkillLoaderProtocol

    def _run(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str | None:
        skill_name = self._resolve_skill_name(args=args, kwargs=kwargs)
        script_name = self._resolve_script_name(args=args, kwargs=kwargs)
        arguments = self._resolve_arguments(args=args, kwargs=kwargs)
        return anyio.run(
            lambda: self._run_skill_script(
                skill_name=skill_name, script_name=script_name, arguments=arguments
            )
        )

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str | None:
        skill_name = self._resolve_skill_name(args=args, kwargs=kwargs)
        script_name = self._resolve_script_name(args=args, kwargs=kwargs)
        arguments = self._resolve_arguments(args=args, kwargs=kwargs)
        return await self._run_skill_script(
            skill_name=skill_name, script_name=script_name, arguments=arguments
        )

    @staticmethod
    def _resolve_skill_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_skill_name = kwargs.get("skill_name", args[0] if args else "")
        return raw_skill_name if isinstance(raw_skill_name, str) else ""

    @staticmethod
    def _resolve_script_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_script_name = kwargs.get("script_name", args[1] if len(args) > 1 else "")
        return raw_script_name if isinstance(raw_script_name, str) else ""

    @staticmethod
    def _resolve_arguments(
        *, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        arguments = kwargs.get("arguments", args[2] if len(args) > 2 else None)
        return arguments

    async def _run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> str | None:
        normalized_name = skill_name.strip()
        if not normalized_name:
            return self._format_availability_message(self.skill_loader, normalized_name)

        try:
            result: MyScriptExecutionResult = await self.skill_loader.run_skill_script(
                skill_name=normalized_name, script_name=script_name, arguments=arguments
            )
            if result.success:
                return result.stdout  # Script output
            else:
                return f"Error: {result.stderr} Exit code: {result.exit_code}"
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
