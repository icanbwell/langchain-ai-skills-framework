from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """Lightweight summary returned by :class:`ListSkillsService`."""

    name: str
    description: str
    plugin_name: str | None = None
    folder: str | None = None
    state: str = "published"


class ListSkillsService:
    """List skills available to a user."""

    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self._loader = skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        folder: str | None = None,
        include_testing: bool = False,
    ) -> Sequence[SkillInfo]:
        """Return available skills as a sequence of ``SkillInfo``."""
        if user_id:
            summaries = await self._loader.list_all_summaries(
                user_id=user_id, allowed_skills=set(), include_testing=include_testing
            )
        else:
            summaries = self._loader.list_skill_summaries(allowed_skills=set())

        if plugin_name:
            summaries = [s for s in summaries if s.plugin_name == plugin_name]

        if folder is not None:
            summaries = [s for s in summaries if s.folder == folder]

        results = sorted(
            (
                SkillInfo(
                    name=s.name, description=s.description, plugin_name=s.plugin_name, folder=s.folder, state=s.state
                )
                for s in summaries
            ),
            key=lambda s: s.name,
        )
        logger.debug("ListSkillsService: found %d skills for user_id=%s", len(results), user_id)
        return results

    @staticmethod
    def format_as_text(skills: Sequence[SkillInfo]) -> str:
        """Format the skill list as an XML string."""
        if not skills:
            return "<available_skills>\n</available_skills>"
        skill_elements = []
        for s in skills:
            plugin_tag = f"<plugin_name>{s.plugin_name}</plugin_name>\n" if s.plugin_name else ""
            folder_tag = f"<folder>{s.folder}</folder>\n" if s.folder else ""
            skill_elements.append(
                f"<skill>\n{plugin_tag}{folder_tag}<name>{s.name}</name>\n"
                f"<description>{s.description}</description>\n</skill>"
            )
        return "<available_skills>\n" + "\n".join(skill_elements) + "\n</available_skills>"
