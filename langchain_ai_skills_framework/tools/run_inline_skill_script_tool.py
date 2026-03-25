from __future__ import annotations

import asyncio
import logging
from typing import Any, Type

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
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class RunInlineSkillScriptInput(BaseModel):
    """Input schema for the run_inline_skill_script tool."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(
        description="Name of the skill used as the execution context.",
    )

    script_name: str = Field(
        description=(
            """Name used for the inline script execution context.
                    Usually includes .py extension: \"analyze.py\", \"process.py\".
                    Must match exactly when referenced by skill instructions."""
        ),
    )

    script: str = Field(
        description=(
            """Python script content to execute.
                    The full script body is executed in the skill base directory context."""
        ),
    )

    arguments: dict[str, Any] | None = Field(
        default=None,
        description=(
            """Optional dictionary of arguments to pass to the script when executing.
                    The keys and values should match what the script expects."""
        ),
    )


class RunInlineSkillScriptTool(StructuredTool):
    """LangChain tool that executes inline Python script content within a skill context."""

    name: str = "run_inline_skill_script"
    description: str = """Execute inline script content within a skill's execution context.

            This tool runs Python script content provided at runtime and executes it
            using the selected skill's base directory and runtime configuration.

            When to use this:
            - When you need to execute generated script content for a known skill
            - When a workflow provides script text directly instead of a script file

            Important:
            - The script content is executed as provided
            - Keep script content minimal and task-specific
            - Review skill instructions before running scripts
            - Scripts may modify external state (files, databases, APIs)
            - Execution errors are included in the output
            """
    args_schema: Type[BaseModel] = RunInlineSkillScriptInput
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
        script = self._resolve_script(args=args, kwargs=kwargs)
        arguments = self._resolve_arguments(args=args, kwargs=kwargs)
        return asyncio.run(
            self._run_inline_skill_script(
                skill_name=skill_name,
                script_name=script_name,
                script=script,
                arguments=arguments,
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
        script = self._resolve_script(args=args, kwargs=kwargs)
        arguments = self._resolve_arguments(args=args, kwargs=kwargs)
        logger.debug(
            "RunInlineSkillScriptTool: Running inline script %s in %s with script_length=%d and arguments %s",
            script_name,
            skill_name,
            len(script),
            arguments,
        )
        try:
            script_result = await self._run_inline_skill_script(
                skill_name=skill_name,
                script_name=script_name,
                script=script,
                arguments=arguments,
            )
            logger.debug(
                "RunInlineSkillScriptTool: Output from inline script %s in %s with arguments %s\n%s",
                script_name,
                skill_name,
                arguments,
                script_result,
            )
            return script_result
        except Exception as exc:
            logger.exception(
                "RunInlineSkillScriptTool: Error running inline script %s in %s",
                script_name,
                skill_name,
            )
            return f"Error running script: {exc}"

    @staticmethod
    def _resolve_skill_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_skill_name = kwargs.get("skill_name", args[0] if args else "")
        return raw_skill_name if isinstance(raw_skill_name, str) else ""

    @staticmethod
    def _resolve_script_name(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_script_name = kwargs.get("script_name", args[1] if len(args) > 1 else "")
        return raw_script_name if isinstance(raw_script_name, str) else ""

    @staticmethod
    def _resolve_script(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        raw_script = kwargs.get("script", args[2] if len(args) > 2 else "")
        return raw_script if isinstance(raw_script, str) else ""

    @staticmethod
    def _resolve_arguments(
        *, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        arguments = kwargs.get("arguments", args[3] if len(args) > 3 else None)
        return arguments

    async def _run_inline_skill_script(
        self,
        *,
        skill_name: str,
        script_name: str,
        script: str,
        arguments: dict[str, Any] | None,
    ) -> str | None:
        normalized_name = skill_name.strip()
        if not normalized_name:
            return self._format_availability_message(self.skill_loader, normalized_name)

        try:
            result: MyScriptExecutionResult = (
                await self.skill_loader.run_inline_skill_script(
                    skill_name=normalized_name,
                    script_name=script_name,
                    script=script,
                    arguments=arguments,
                )
            )
            if result.success:
                return result.stdout
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
