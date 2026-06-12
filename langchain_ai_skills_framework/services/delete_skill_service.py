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
class DeleteSkillResult:
    """Outcome of a delete_skill operation.

    ``ok=True`` means a row was actually removed. ``ok=False`` means the
    skill was not found — the operation is still idempotent (callers can
    return HTTP 200) but the distinction is available to callers that
    want to surface it differently (e.g. HTTP 404).
    """

    ok: bool
    message: str


class DeleteSkillService:
    """Delete a user-saved skill from the store."""

    def __init__(self, *, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(self, *, user_id: str, plugin_name: str, skill_name: str) -> DeleteSkillResult:
        require_user_id(user_id=user_id, operation="delete_skill")
        require_non_empty(value=skill_name, label="skill_name")
        store = require_store(store=self._store)

        try:
            deleted = await store.delete_skill(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            if deleted:
                message = f"Skill '{skill_name}' deleted successfully."
            else:
                message = f"Skill '{skill_name}' not found — nothing to delete."
            logger.info("DeleteSkillService: %s (user=%s)", message, user_id)
            return DeleteSkillResult(ok=deleted, message=message)
        except Exception as exc:
            logger.exception(
                "DeleteSkillService failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise SkillOperationError(f"Unable to delete skill '{skill_name}' due to an internal error.") from exc
