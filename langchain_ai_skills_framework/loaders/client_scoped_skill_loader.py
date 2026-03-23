from __future__ import annotations

from typing import Sequence

from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillLoaderProtocol,
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.skills_model import SkillSummary, SkillDetails


class ClientScopedSkillLoader(SkillLoaderProtocol):
    """Skill loader wrapper that enforces client-specific allowlists.

    A wildcard token ("*") in ``allowed_skills`` allows access to all skills.
    """

    def __init__(self, *, base_loader: SkillLoaderProtocol) -> None:
        if base_loader is None:
            raise ValueError("base_loader must not be None")
        if not isinstance(base_loader, SkillLoaderProtocol):
            raise TypeError(
                f"base_loader must be SkillLoaderProtocol, got {type(base_loader)}"
            )
        self._base_loader = base_loader
        self._allow_all_skills = "*" in self._allowed_skills

    def list_skill_summaries(
        self, *, allowed_skills: set[str]
    ) -> Sequence[SkillSummary]:
        summaries = self._base_loader.list_skill_summaries(
            allowed_skills=allowed_skills
        )
        if not allowed_skills or self._allow_all_skills:
            return summaries
        allowed_skill_names = self._resolve_allowed_skill_names(summaries)
        return tuple(
            summary
            for summary in summaries
            if self._normalize_skill_token(summary.name) in allowed_skill_names
        )

    def get_skill_details(
            self, *, skill_name: str
    ) -> SkillDetails:
        result: SkillDetails = self._base_loader.get_skill_details(
            skill_name=skill_name,
        )
        return result

    def refresh(self) -> None:
        self._base_loader.refresh()

    async def get_instructions(self) -> str:
        return await self._base_loader.get_instructions()

    def _resolve_allowed_skill_names(
        self, *, summaries: Sequence[SkillSummary], allowed_skills: set[str]
    ) -> set[str]:
        if not allowed_skills:
            return set()
        allowed_skill_names: set[str] = set()
        for summary in summaries:
            normalized_name = self._normalize_skill_token(summary.name)
            if normalized_name in allowed_skills:
                allowed_skill_names.add(normalized_name)
                continue
            group_name = self._extract_group_name(summary)
            if group_name and group_name in allowed_skills:
                allowed_skill_names.add(normalized_name)
        return allowed_skill_names

    @classmethod
    def _extract_group_name(cls, summary: SkillSummary) -> str | None:
        skill_dir = summary.source_path.parent
        group_dir = skill_dir.parent if skill_dir else None
        if group_dir is None or not group_dir.name:
            return None
        return cls._normalize_skill_token(group_dir.name)

    @staticmethod
    def _normalize_skill_token(value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        while "--" in normalized:
            normalized = normalized.replace("--", "-")
        return normalized.strip("-")
