from __future__ import annotations

from pathlib import Path
from typing import Mapping

from aiskills.skills.loaders.skill_loader import SkillNotFoundError
from aiskills.skills.models.skills_model import SkillDetails, SkillSummary
from aiskills.skills.tools.skills_tool import LoadSkillTool


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


def test_load_skill_tool_returns_availability_for_empty_name() -> None:
    details = _make_skill("alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    message = tool._load_skill("")

    assert "No skill name provided." in message
    assert "alpha" in message


def test_load_skill_tool_returns_availability_when_missing() -> None:
    details = _make_skill("alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    message = tool._load_skill("beta")

    assert "Skill 'beta' not found." in message
    assert "alpha" in message


def test_load_skill_tool_returns_skill_content() -> None:
    details = _make_skill("alpha", content="Body for alpha")
    loader = _StubSkillLoader({"alpha": details})
    tool = LoadSkillTool(skill_loader=loader)

    message = tool._load_skill(" alpha ")

    assert message.startswith("Loaded skill: alpha")
    assert "Body for alpha" in message
