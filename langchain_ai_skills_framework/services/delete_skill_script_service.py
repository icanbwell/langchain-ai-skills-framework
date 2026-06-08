from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_non_empty,
    require_store,
    require_user_id,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class DeleteSkillScriptService:
    """Delete a script attached to a skill."""

    def __init__(self, *, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> str:
        require_user_id(user_id=user_id, operation="delete_skill_script")
        require_non_empty(value=skill_name, label="skill_name")
        require_non_empty(value=script_name, label="script_name")
        store = require_store(store=self._store)

        try:
            deleted = await store.delete_script(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
            )
            if not deleted:
                raise SkillOperationError(f"Script '{script_name}' not found for skill '{skill_name}'.")
            message = f"Script '{script_name}' deleted successfully from skill '{skill_name}'."
            logger.info("DeleteSkillScriptService: %s (user=%s)", message, user_id)
            return message
        except SkillOperationError:
            raise
        except Exception as exc:
            logger.exception(
                "DeleteSkillScriptService failed for skill_name=%s script_name=%s user=%s",
                skill_name,
                script_name,
                user_id,
            )
            raise SkillOperationError(
                f"Unable to delete script '{script_name}' from skill '{skill_name}' due to an internal error."
            ) from exc
