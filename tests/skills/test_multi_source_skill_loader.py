from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest
from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.multi_source_skill_loader import (
    MultiSourceSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)


def _make_skill(name: str, *, description: str = "", source: str = "primary") -> SkillDetails:
    desc = description or f"Description for {name}"
    source_path = Path(f"/{source}/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=desc,
        source_path=source_path,
        metadata={"source": source},
    )
    return SkillDetails(summary=summary, content=f"Content for {name}", source_path=source_path)


class StubLoader(SkillLoaderProtocol):
    """Minimal in-memory loader for testing."""

    def __init__(self, skills: dict[str, SkillDetails]) -> None:
        self._skills = skills
        self.refresh_count = 0

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return [d.summary for d in self._skills.values()]

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills)

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        try:
            return self._skills[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError(f"'{skill_name}' not found") from exc

    async def get_skill_details_for_user(self, *, user_id: str, skill_name: str) -> SkillDetails:
        return self.get_skill_details(skill_name)

    def refresh(self) -> None:
        self.refresh_count += 1

    async def get_instructions(self) -> str:
        return "<available_skills>primary-only</available_skills>"

    def get_tools(self) -> list[BaseTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        raise SkillNotFoundError("no resources in stub")

    async def read_skill_resource_for_user(self, *, user_id: str, skill_name: str, resource_name: str) -> str:
        return self.read_skill_resource(skill_name, resource_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        raise SkillNotFoundError("no scripts in stub")

    async def run_skill_script_for_user(
        self, *, user_id: str, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name, script_name, arguments)

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return []

    async def list_skill_script_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_script_names(skill_name)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return []

    async def list_skill_resource_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name)

    def get_plugin_mcp_configs(self) -> Sequence[PluginMcpServerEntry]:
        return []


class TestMultiSourceSkillLoaderInit:
    def test_rejects_empty_loaders(self) -> None:
        with pytest.raises(ValueError, match="At least one loader"):
            MultiSourceSkillLoader(loaders=[])

    def test_primary_loader_is_first(self) -> None:
        loader_a = StubLoader({})
        loader_b = StubLoader({})
        multi = MultiSourceSkillLoader(loaders=[loader_a, loader_b])
        assert multi.primary_loader is loader_a


class TestListSkillSummaries:
    def test_merges_summaries_from_all_loaders(self) -> None:
        primary = StubLoader({"alpha": _make_skill("alpha", source="primary")})
        secondary = StubLoader({"beta": _make_skill("beta", source="marketplace")})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        summaries = multi.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "alpha" in names
        assert "beta" in names

    def test_primary_wins_on_name_collision(self) -> None:
        primary_skill = _make_skill("shared-name", description="primary version", source="primary")
        secondary_skill = _make_skill("shared-name", description="marketplace version", source="marketplace")
        primary = StubLoader({"shared-name": primary_skill})
        secondary = StubLoader({"shared-name": secondary_skill})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        summaries = multi.list_skill_summaries(allowed_skills=set())

        assert len(summaries) == 1
        assert summaries[0].description == "primary version"

    def test_results_are_sorted_by_name(self) -> None:
        primary = StubLoader({"zebra": _make_skill("zebra")})
        secondary = StubLoader({"apple": _make_skill("apple")})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        summaries = multi.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert names == sorted(names)


class TestGetSkillDetails:
    def test_returns_from_primary_first(self) -> None:
        skill = _make_skill("my-skill", source="primary")
        primary = StubLoader({"my-skill": skill})
        secondary = StubLoader({})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        details = multi.get_skill_details("my-skill")
        assert details.content == "Content for my-skill"

    def test_falls_back_to_secondary(self) -> None:
        skill = _make_skill("marketplace-skill", source="marketplace")
        primary = StubLoader({})
        secondary = StubLoader({"marketplace-skill": skill})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        details = multi.get_skill_details("marketplace-skill")
        assert details.content == "Content for marketplace-skill"

    def test_raises_not_found_when_no_loader_has_skill(self) -> None:
        primary = StubLoader({})
        secondary = StubLoader({})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        with pytest.raises(SkillNotFoundError):
            multi.get_skill_details("nonexistent")


class TestGetInstructions:
    @pytest.mark.asyncio
    async def test_includes_skills_from_all_loaders(self) -> None:
        primary = StubLoader({"alpha": _make_skill("alpha", source="primary")})
        secondary = StubLoader({"beta": _make_skill("beta", source="marketplace")})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        instructions = await multi.get_instructions()

        assert "<available_skills>" in instructions
        assert "alpha" in instructions
        assert "beta" in instructions

    @pytest.mark.asyncio
    async def test_does_not_delegate_to_primary_only(self) -> None:
        """Regression: previously delegated to primary loader, hiding marketplace skills."""
        primary = StubLoader({"alpha": _make_skill("alpha")})
        secondary = StubLoader({"beta": _make_skill("beta", source="marketplace")})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        instructions = await multi.get_instructions()

        # Should NOT just return the primary loader's instructions
        assert "primary-only" not in instructions
        # Should include marketplace skills
        assert "beta" in instructions

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_skills(self) -> None:
        primary = StubLoader({})
        secondary = StubLoader({})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        instructions = await multi.get_instructions()
        assert instructions == ""


class TestRefresh:
    def test_refresh_delegates_to_all_loaders(self) -> None:
        primary = StubLoader({})
        secondary = StubLoader({})
        multi = MultiSourceSkillLoader(loaders=[primary, secondary])

        multi.refresh()

        assert primary.refresh_count == 1
        assert secondary.refresh_count == 1
