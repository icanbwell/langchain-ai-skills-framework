from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.mutation_result import MutationResult
from langchain_ai_skills_framework.services.post_save_script_hook import PostSaveScriptHook
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_non_empty,
    require_store,
    require_user_id,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class SaveSkillScriptService:
    """Save or update a script attached to a skill."""

    def __init__(
        self,
        *,
        mongo_skill_loader: PluginSkillStore | None,
        post_save_hook: PostSaveScriptHook | None = None,
    ) -> None:
        self._store = mongo_skill_loader
        self._post_save_hook = post_save_hook

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        content: str,
        folder: str | None = None,
        path: str | None = None,
    ) -> MutationResult:
        require_user_id(user_id=user_id, operation="save_skill_script")
        require_non_empty(value=skill_name, label="skill_name")
        require_non_empty(value=script_name, label="script_name")
        require_non_empty(value=content, label="content")
        store = require_store(store=self._store)

        try:
            doc = await store.save_script(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
                content=content,
                modified_by=user_id,
                folder=folder,
                path=path,
            )
            message = f"Script '{doc.script_name}' saved for skill '{doc.skill_name}'."
            logger.info("SaveSkillScriptService: %s (user=%s)", message, user_id)

            if self._post_save_hook is not None:
                try:
                    await self._post_save_hook.on_script_saved(
                        user_id=user_id,
                        plugin_name=plugin_name,
                        skill_name=skill_name,
                        script_name=script_name,
                    )
                except Exception:
                    logger.warning(
                        "Post-save hook failed for script_name=%s (non-fatal)",
                        script_name,
                        exc_info=True,
                    )

            return MutationResult(ok=True, message=message)
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
