from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class ToggleSkillSharingService:
    """Toggle a skill between shared and private."""

    def __init__(self, *, mongo_skill_loader: UserSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(self, *, user_id: str, skill_name: str, shared: bool) -> str:
        if not user_id:
            raise SkillOperationError("user_id is required for toggle_skill_sharing")
        if not skill_name or not skill_name.strip():
            raise SkillOperationError("skill_name must be a non-empty string.")
        if self._store is None:
            raise SkillOperationError("mongo_skill_loader is not configured.")

        try:
            doc = await self._store.set_skill_shared(
                user_id=user_id,
                skill_name=skill_name,
                shared=shared,
            )
            state = "shared" if doc.shared else "private"
            message = f"Skill '{doc.skill_name}' is now {state}."
            logger.info("ToggleSkillSharingService: %s (user=%s)", message, user_id)
            return message
        except Exception as exc:
            logger.exception(
                "ToggleSkillSharingService failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise SkillOperationError(
                f"Unable to update sharing for skill '{skill_name}' due to an internal error."
            ) from exc
