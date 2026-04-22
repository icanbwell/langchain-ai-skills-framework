from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class DeleteSkillService:
    """Delete a user-saved skill from the store."""

    def __init__(self, *, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(self, *, user_id: str, plugin_name: str, skill_name: str) -> str:
        if not user_id:
            raise SkillOperationError("user_id is required for delete_skill")
        if not skill_name or not skill_name.strip():
            raise SkillOperationError("skill_name must be a non-empty string.")
        if self._store is None:
            raise SkillOperationError("mongo_skill_loader is not configured.")

        try:
            deleted = await self._store.delete_skill(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            if deleted:
                message = f"Skill '{skill_name}' deleted successfully."
            else:
                message = f"Skill '{skill_name}' not found — nothing to delete."
            logger.info("DeleteSkillService: %s (user=%s)", message, user_id)
            return message
        except Exception as exc:
            logger.exception(
                "DeleteSkillService failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise SkillOperationError(f"Unable to delete skill '{skill_name}' due to an internal error.") from exc
