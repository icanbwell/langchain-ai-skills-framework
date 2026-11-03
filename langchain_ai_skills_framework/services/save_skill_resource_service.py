from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.mutation_result import MutationResult
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_non_empty,
    require_store,
    require_user_id,
    resolve_owner_and_actor,
)
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
        folder: str | None = None,
        path: str | None = None,
        author: str | None = None,
        modified_by: str | None = None,
    ) -> MutationResult:
        """Execute the operation.

        ``author`` is whose skill is acted on and ``modified_by`` is the acting user recorded in
        history. Each defaults to ``user_id`` only when omitted; an explicitly blank value raises
        ``SkillOperationError``. This service performs no authorization: a caller passing an
        ``author`` or ``modified_by`` different from ``user_id`` must first verify that user may do so.
        """
        require_user_id(user_id=user_id, operation="save_skill_resource")
        who = resolve_owner_and_actor(user_id=user_id, author=author, modified_by=modified_by)
        owner, actor = who.owner, who.actor
        require_non_empty(value=skill_name, label="skill_name")
        require_non_empty(value=resource_name, label="resource_name")
        require_non_empty(value=content, label="content")
        store = require_store(store=self._store)

        try:
            doc = await store.save_resource(
                author=owner,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=resource_name,
                content=content,
                modified_by=actor,
                folder=folder,
                path=path,
            )
            message = f"Resource '{doc.resource_name}' saved for skill '{doc.skill_name}'."
            logger.info("SaveSkillResourceService: %s (user=%s)", message, user_id)
            return MutationResult(ok=True, message=message)
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
