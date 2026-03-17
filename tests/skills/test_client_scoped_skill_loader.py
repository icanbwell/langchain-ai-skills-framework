from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest

from langchain_ai_skills_framework.loaders.client_scoped_skill_loader import (
    ClientScopedSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_loader import SkillNotFoundError
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary


class _StubSkillLoader:
    def __init__(self, details_by_name: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details_by_name)

    def list_skill_summaries(self) -> tuple[SkillSummary, ...]:
        return tuple(detail.summary for detail in self._details.values())

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        try:
            return self._details[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError from exc

    def refresh(self) -> None:
        return None


def _make_skill(name: str, *, content: str = "Skill content") -> SkillDetails:
    source_path = Path(f"/skills/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path,
    )
    return SkillDetails(summary=summary, content=content, source_path=source_path)


def test_client_scoped_skill_loader_allows_all_when_empty() -> None:
    details = _make_skill("alpha")
    loader = ClientScopedSkillLoader(
        base_loader=_StubSkillLoader({"alpha": details}),
        allowed_skills=set(),
    )

    summaries = loader.list_skill_summaries()

    assert len(summaries) == 1
    assert summaries[0].name == "alpha"


def test_client_scoped_skill_loader_filters_summaries() -> None:
    details = _make_skill("alpha")
    loader = ClientScopedSkillLoader(
        base_loader=_StubSkillLoader({"alpha": details}),
        allowed_skills={"beta"},
    )

    summaries = loader.list_skill_summaries()

    assert summaries == ()


def test_client_scoped_skill_loader_rejects_disallowed_skill() -> None:
    details = _make_skill("alpha")
    loader = ClientScopedSkillLoader(
        base_loader=_StubSkillLoader({"alpha": details}),
        allowed_skills={"beta"},
    )

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_details("alpha")


def test_client_scoped_skill_loader_allows_allowed_skill() -> None:
    details = _make_skill("alpha", content="Body")
    loader = ClientScopedSkillLoader(
        base_loader=_StubSkillLoader({"alpha": details}),
        allowed_skills={"alpha"},
    )

    result = loader.get_skill_details("alpha")

    assert result.content == "Body"
