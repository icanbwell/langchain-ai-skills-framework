from __future__ import annotations

import logging
from typing import Any

from skillkit import ScriptNotFoundError

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.services.availability_helpers import (
    format_script_availability,
    format_skill_availability,
)
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class RunSkillScriptService:
    """Execute a skill script and return a (content, artifact) tuple."""

    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self._loader = skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> tuple[str, str]:
        """Run the script and return ``(content, artifact)``.

        ``plugin_name`` is optional. When omitted the loader resolves the skill
        by ``(user_id, skill_name)`` alone — the right behavior for LLM-driven
        callers that don't reliably know the owning plugin.

        Returns availability messages on not-found (soft errors).
        Raises ``SkillOperationError`` on unexpected failures.
        """
        normalized_name = skill_name.strip()
        if not normalized_name:
            raise SkillOperationError(
                await format_skill_availability(loader=self._loader, normalized_name=normalized_name, user_id=user_id)
            )

        normalized_script_name = script_name.strip()
        if not normalized_script_name:
            raise SkillOperationError("No script name provided.")

        logger.debug(
            "RunSkillScriptService: Running script_name=%s skill_name=%s argument_keys=%s",
            normalized_script_name,
            normalized_name,
            sorted((arguments or {}).keys()),
        )

        try:
            script_result = await self._run_skill_script(
                skill_name=normalized_name,
                script_name=normalized_script_name,
                arguments=arguments,
                plugin_name=plugin_name,
                user_id=user_id,
            )
            logger.debug(
                "RunSkillScriptService: Script completed script_name=%s skill_name=%s",
                normalized_script_name,
                normalized_name,
            )
            if script_result.success:
                return (
                    script_result.stdout or "No output",
                    script_result.stdout or "",
                )
            else:
                return (
                    script_result.stderr or script_result.stdout or "No output",
                    script_result.stdout or "",
                )
        except ScriptNotFoundError:
            return (
                await format_script_availability(
                    loader=self._loader,
                    skill_name=normalized_name,
                    script_name=normalized_script_name,
                    user_id=user_id,
                    plugin_name=plugin_name,
                ),
                "",
            )
        except SkillNotFoundError:
            return (
                await format_skill_availability(loader=self._loader, normalized_name=normalized_name, user_id=user_id),
                "",
            )
        except SkillOperationError:
            raise
        except Exception as exc:
            logger.exception(
                "RunSkillScriptService unexpected failure script_name=%s skill_name=%s",
                normalized_script_name,
                normalized_name,
            )
            raise SkillOperationError(
                f"Unable to run script '{normalized_script_name}' in skill '{normalized_name}' due to an internal error."
            ) from exc

    async def _run_skill_script(
        self,
        *,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
        plugin_name: str | None,
        user_id: str,
    ) -> MyScriptExecutionResult:
        try:
            if user_id:
                result: MyScriptExecutionResult = await self._loader.run_skill_script_for_user(
                    user_id=user_id,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                    script_name=script_name,
                    arguments=arguments,
                )
            else:
                result = await self._loader.run_skill_script(
                    skill_name=skill_name,
                    script_name=script_name,
                    arguments=arguments,
                    plugin_name=plugin_name,
                )
            return result
        except (SkillOperationError, SkillNotFoundError, ScriptNotFoundError):
            raise
        except Exception as exc:
            logger.exception(
                "RunSkillScriptService failed for skill_name=%s script_name=%s",
                skill_name,
                script_name,
            )
            raise SkillOperationError(
                f"Unable to run script '{script_name}' in skill '{skill_name}' due to an internal error."
            ) from exc
