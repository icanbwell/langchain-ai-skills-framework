from __future__ import annotations

import logging
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class SaveSkillResourceResult:
    """Outcome of a save_skill_resource operation.

    Mirrors :class:`SaveSkillResult` so all skill-mutation services
    expose a uniform ``(ok, message)`` shape. This service has no soft
    failure today — every non-raising path is ``ok=True`` — but the
    structured shape keeps callers symmetric with the other save/delete
    services and leaves room for future validation paths.
    """

    ok: bool
    message: str


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
    ) -> SaveSkillResourceResult:
        require_user_id(user_id=user_id, operation="save_skill_resource")
        require_non_empty(value=skill_name, label="skill_name")
        require_non_empty(value=resource_name, label="resource_name")
        require_non_empty(value=content, label="content")
        store = require_store(store=self._store)

        try:
            doc = await store.save_resource(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=resource_name,
                content=content,
                modified_by=user_id,
                folder=folder,
                path=path,
            )
            message = f"Resource '{doc.resource_name}' saved for skill '{doc.skill_name}'."
            logger.info("SaveSkillResourceService: %s (user=%s)", message, user_id)
            return SaveSkillResourceResult(ok=True, message=message)
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
