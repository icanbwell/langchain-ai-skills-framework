from __future__ import annotations
import asyncio
import logging
from typing import Any, Type
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
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


class RunSkillScriptInput(BaseModel):
    """Input schema for the run_skill_script tool."""

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


class RunSkillScriptTool(BaseTool):
    """LangChain tool that executes skill scripts."""

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
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str | None:
        """Synchronously execute a skill script."""
        return asyncio.run(
            self._run_skill_script(
                skill_name=skill_name,
                script_name=script_name,
                arguments=arguments,
            )
        )

    async def _arun(
        self,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str | None:
        """Asynchronously execute a skill script."""
        logger.debug(
            "RunSkillScriptTool: Running script_name=%s skill_name=%s argument_keys=%s",
            script_name,
            skill_name,
            sorted((arguments or {}).keys()),
        )
        try:
            script_result = await self._run_skill_script(
                skill_name=skill_name, script_name=script_name, arguments=arguments
            )
            logger.debug(
                "RunSkillScriptTool: Script completed script_name=%s skill_name=%s",
                script_name,
                skill_name,
            )
            return script_result
        except ToolException:
            raise
        except Exception as exc:
            logger.exception(
                "RunSkillScriptTool unexpected failure script_name=%s skill_name=%s",
                script_name,
                skill_name,
            )
            raise ToolException(
                f"Unable to run script '{script_name}' in skill '{skill_name}' due to an internal error."
            ) from exc

    async def _run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> str | None:
        """Execute the skill script and return results."""
        normalized_name = skill_name.strip()

        if not normalized_name:
            raise ToolException(
                self._format_availability_message(self.skill_loader, normalized_name)
            )

        try:
            result: MyScriptExecutionResult = await self.skill_loader.run_skill_script(
                skill_name=normalized_name, script_name=script_name, arguments=arguments
            )

            if result.success:
                return result.stdout  # Script output
            raise ToolException(
                f"Script '{script_name}' failed in skill '{normalized_name}'. "
                f"Exit code: {result.exit_code}. Error: {result.stderr or 'Unknown error'}"
            )
        except ToolException:
            raise
        except SkillNotFoundError as exc:
            raise ToolException(
                self._format_availability_message(self.skill_loader, normalized_name)
            ) from exc
        except Exception as exc:
            logger.exception(
                "RunSkillScriptTool failed for skill_name=%s script_name=%s",
                normalized_name,
                script_name,
            )
            raise ToolException(
                f"Unable to run script '{script_name}' in skill '{normalized_name}' due to an internal error."
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
