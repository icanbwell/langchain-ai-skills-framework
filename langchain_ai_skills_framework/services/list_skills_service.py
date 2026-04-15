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


class ListSkillsService:
    """List skills available to a user."""

    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self._loader = skill_loader

    async def execute(self, *, user_id: str) -> Sequence[SkillInfo]:
        """Return available skills as a sequence of ``SkillInfo``."""
        if user_id:
            summaries = await self._loader.list_all_summaries(user_id=user_id, allowed_skills=set())
        else:
            summaries = self._loader.list_skill_summaries(allowed_skills=set())

        results = sorted(
            (SkillInfo(name=s.name, description=s.description) for s in summaries),
            key=lambda s: s.name,
        )
        logger.debug("ListSkillsService: found %d skills for user_id=%s", len(results), user_id)
        return results

    @staticmethod
    def format_as_text(skills: Sequence[SkillInfo]) -> str:
        """Format the skill list as a human-readable string."""
        if not skills:
            return "No skills available."
        lines = [f"- **{s.name}**: {s.description}" for s in skills]
        return f"Available skills ({len(skills)}):\n" + "\n".join(lines)
