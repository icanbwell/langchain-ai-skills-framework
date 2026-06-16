from __future__ import annotations

import logging
from dataclasses import dataclass

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

    async def execute(self, *, user_id: str, plugin_name: str | None = None, skill_name: str) -> str:
        """Return the skill content string.

        ``plugin_name`` is optional. When omitted the loader resolves the skill
        by ``(user_id, skill_name)`` alone, which is the right behavior for
        LLM-driven callers that don't reliably know the owning plugin.

        On not-found, returns an availability message (soft error).
        Raises ``SkillOperationError`` on unexpected failures.
        """
        normalized_name = skill_name.strip()
        if not normalized_name:
            raise SkillOperationError(
                await format_skill_availability(loader=self._loader, normalized_name=normalized_name, user_id=user_id)
            )

        loaded = await self._load_skill(normalized_name, plugin_name=plugin_name, user_id=user_id)
        logger.debug("LoadSkillService: loaded skill_name=%s", normalized_name)

        if self._user_skill_store and user_id and loaded.plugin_name:
            try:
                await self._user_skill_store.record_skill_usage(
                    plugin_name=loaded.plugin_name,
                    skill_name=normalized_name,
                    user_id=user_id,
                )
            except Exception:
                logger.debug(
                    "Failed to record skill usage for skill_name=%s",
                    normalized_name,
                    exc_info=True,
                )

        return loaded.content

    async def _load_skill(self, skill_name: str, *, plugin_name: str | None, user_id: str) -> _LoadedSkill:
        try:
            if user_id:
                skill = await self._loader.get_skill_details_for_user(
                    user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
                )
            else:
                skill = self._loader.get_skill_details(skill_name=skill_name, plugin_name=plugin_name)
            author = skill.summary.metadata.get("user_id") if skill.summary.metadata else None
            content = f"Author: {author}\n\n{skill.content}" if author else skill.content
            return _LoadedSkill(content=content, plugin_name=skill.summary.plugin_name)
        except SkillNotFoundError:
            availability = await format_skill_availability(
                loader=self._loader, normalized_name=skill_name, user_id=user_id
            )
            return _LoadedSkill(content=availability, plugin_name=None)
        except Exception as exc:
            logger.exception("LoadSkillService failed for skill_name=%s", skill_name)
            raise SkillOperationError(f"Unable to load skill '{skill_name}' due to an internal error.") from exc


@dataclass(frozen=True, slots=True)
class _LoadedSkill:
    """Internal carrier so ``execute`` can record usage with the resolved plugin."""

    content: str
    plugin_name: str | None
