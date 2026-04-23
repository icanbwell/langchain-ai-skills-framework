from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class SaveSkillResourceService:
    """Save or update a resource attached to a skill."""

    def __init__(self, *, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        content: str,
    ) -> str:
        if not user_id:
            raise SkillOperationError("user_id is required for save_skill_resource")
        if not skill_name or not skill_name.strip():
            raise SkillOperationError("skill_name must be a non-empty string.")
        if not resource_name or not resource_name.strip():
            raise SkillOperationError("resource_name must be a non-empty string.")
        if not content or not content.strip():
            raise SkillOperationError("content must be a non-empty string.")
        if self._store is None:
            raise SkillOperationError("mongo_skill_loader is not configured.")

        try:
            doc = await self._store.save_resource(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=resource_name,
                content=content,
                modified_by=user_id,
            )
            message = f"Resource '{doc.resource_name}' saved for skill '{doc.skill_name}'."
            logger.info("SaveSkillResourceService: %s (user=%s)", message, user_id)
            return message
        except Exception as exc:
            logger.exception(
                "SaveSkillResourceService failed for skill_name=%s resource_name=%s user=%s",
                skill_name,
                resource_name,
                user_id,
            )
            raise SkillOperationError(
                f"Unable to save resource '{resource_name}' for skill '{skill_name}' due to an internal error."
            ) from exc
