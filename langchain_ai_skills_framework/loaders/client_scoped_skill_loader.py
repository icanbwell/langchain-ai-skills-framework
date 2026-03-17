from __future__ import annotations

from typing import Sequence

from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillLoaderProtocol,
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.skills_model import SkillSummary, SkillDetails


class ClientScopedSkillLoader(SkillLoaderProtocol):
    """Skill loader wrapper that enforces client-specific allowlists."""

    def __init__(
        self, *, base_loader: SkillLoaderProtocol, allowed_skills: set[str]
    ) -> None:
        if base_loader is None:
            raise ValueError("base_loader must not be None")
        if not isinstance(base_loader, SkillLoaderProtocol):
            raise TypeError(
                f"base_loader must be SkillLoaderProtocol, got {type(base_loader)}"
            )
        self._base_loader = base_loader
        self._allowed_skills = {
            skill.strip().lower() for skill in allowed_skills if skill
        }

    def list_skill_summaries(self) -> Sequence[SkillSummary]:
        summaries = self._base_loader.list_skill_summaries()
        if not self._allowed_skills:
            return summaries
        return tuple(
            summary
            for summary in summaries
            if summary.name.strip().lower() in self._allowed_skills
        )

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        normalized = skill_name.strip().lower()
        if self._allowed_skills and normalized not in self._allowed_skills:
            raise SkillNotFoundError(f"Skill '{skill_name}' not allowed")
        return self._base_loader.get_skill_details(skill_name)

    def refresh(self) -> None:
        self._base_loader.refresh()
