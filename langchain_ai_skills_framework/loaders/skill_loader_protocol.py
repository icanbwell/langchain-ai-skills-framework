from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary


@runtime_checkable
class SkillLoaderProtocol(Protocol):
    def list_skill_summaries(self) -> Sequence[SkillSummary]: ...

    def get_skill_details(self, skill_name: str) -> SkillDetails: ...

    def refresh(self) -> None: ...
