from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class SaveSkillScriptService:
    """Save or update a script attached to a skill."""

    def __init__(self, *, mongo_skill_loader: UserSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        content: str,
    ) -> str:
        if not user_id:
            raise SkillOperationError("user_id is required for save_skill_script")
        if not skill_name or not skill_name.strip():
            raise SkillOperationError("skill_name must be a non-empty string.")
        if not script_name or not script_name.strip():
            raise SkillOperationError("script_name must be a non-empty string.")
        if not content or not content.strip():
            raise SkillOperationError("content must be a non-empty string.")
        if self._store is None:
            raise SkillOperationError("mongo_skill_loader is not configured.")

        try:
            doc = await self._store.save_script(
                user_id=user_id,
                skill_name=skill_name,
                script_name=script_name,
                content=content,
                modified_by=user_id,
            )
            message = f"Script '{doc.script_name}' saved for skill '{doc.skill_name}'."
            logger.info("SaveSkillScriptService: %s (user=%s)", message, user_id)
            return message
        except Exception as exc:
            logger.exception(
                "SaveSkillScriptService failed for skill_name=%s script_name=%s user=%s",
                skill_name,
                script_name,
                user_id,
            )
            raise SkillOperationError(
                f"Unable to save script '{script_name}' for skill '{skill_name}' due to an internal error."
            ) from exc
