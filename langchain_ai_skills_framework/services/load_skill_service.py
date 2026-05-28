from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.availability_helpers import (
    format_skill_availability,
)
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class LoadSkillService:
    """Load a skill's full content, optionally recording usage."""

    def __init__(
        self,
        *,
        skill_loader: SkillLoaderProtocol,
        user_skill_store: PluginSkillStore | None = None,
    ) -> None:
        self._loader = skill_loader
        self._user_skill_store = user_skill_store

    async def execute(self, *, user_id: str, plugin_name: str, skill_name: str) -> str:
        """Return the skill content string.

        On not-found, returns an availability message (soft error).
        Raises ``SkillOperationError`` on unexpected failures.
        """
        normalized_name = skill_name.strip()
        if not normalized_name:
            raise SkillOperationError(
                await format_skill_availability(loader=self._loader, normalized_name=normalized_name, user_id=user_id)
            )

        content = await self._load_skill(normalized_name, plugin_name=plugin_name, user_id=user_id)
        logger.debug("LoadSkillService: loaded skill_name=%s", normalized_name)

        if self._user_skill_store and user_id:
            try:
                await self._user_skill_store.record_skill_usage(
                    plugin_name=plugin_name, skill_name=normalized_name, user_id=user_id
                )
            except Exception:
                logger.debug(
                    "Failed to record skill usage for skill_name=%s",
                    normalized_name,
                    exc_info=True,
                )

        return content

    async def _load_skill(self, skill_name: str, *, plugin_name: str, user_id: str) -> str:
        normalized_name = skill_name.strip()
        if not normalized_name:
            raise SkillOperationError(
                await format_skill_availability(loader=self._loader, normalized_name=normalized_name, user_id=user_id)
            )

        try:
            if user_id:
                skill = await self._loader.get_skill_details_for_user(
                    user_id=user_id, plugin_name=plugin_name, skill_name=normalized_name
                )
            else:
                skill = self._loader.get_skill_details(skill_name=normalized_name, plugin_name=plugin_name)
            author = skill.summary.metadata.get("user_id") if skill.summary.metadata else None
            if author:
                return f"Author: {author}\n\n{skill.content}"
            return f"{skill.content}"
        except SkillNotFoundError:
            return await format_skill_availability(
                loader=self._loader, normalized_name=normalized_name, user_id=user_id
            )
        except Exception as exc:
            logger.exception("LoadSkillService failed for skill_name=%s", normalized_name)
            raise SkillOperationError(f"Unable to load skill '{normalized_name}' due to an internal error.") from exc
