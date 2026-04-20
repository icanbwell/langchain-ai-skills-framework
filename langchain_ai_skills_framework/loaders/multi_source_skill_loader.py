from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)

logger = logging.getLogger(__name__)


class MultiSourceSkillLoader(SkillLoaderProtocol):
    """Merges multiple SkillLoaderProtocol implementations into a single view.

    The first loader in the list has highest precedence on name collisions.
    Typically: [SkillkitDirectoryLoader, MarketplaceDirectoryLoader].
    """

    def __init__(self, *, loaders: Sequence[SkillLoaderProtocol]) -> None:
        if not loaders:
            raise ValueError("At least one loader must be provided")
        self._loaders: tuple[SkillLoaderProtocol, ...] = tuple(loaders)

    @property
    def primary_loader(self) -> SkillLoaderProtocol:
        """The first (highest-precedence) loader."""
        return self._loaders[0]

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        seen: set[str] = set()
        merged: list[SkillSummary] = []
        for loader in self._loaders:
            for summary in loader.list_skill_summaries(allowed_skills):
                if summary.name not in seen:
                    seen.add(summary.name)
                    merged.append(summary)
        return sorted(merged, key=lambda s: s.name)

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills)

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        last_exc: SkillNotFoundError | None = None
        for loader in self._loaders:
            try:
                return loader.get_skill_details(skill_name)
            except SkillNotFoundError as exc:
                last_exc = exc
        raise last_exc or SkillNotFoundError(f"Skill '{skill_name}' not found")

    async def get_skill_details_for_user(self, *, user_id: str, skill_name: str) -> SkillDetails:
        return self.get_skill_details(skill_name)

    def refresh(self) -> None:
        for loader in self._loaders:
            loader.refresh()

    async def get_instructions(self) -> str:
        # Delegate to the primary loader for the instructions template
        return await self._loaders[0].get_instructions()

    def get_tools(self) -> list[BaseTool]:
        # Tools come from the primary loader only to avoid duplicates
        return self._loaders[0].get_tools()

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        last_exc: Exception | None = None
        for loader in self._loaders:
            try:
                return loader.read_skill_resource(skill_name, resource_name)
            except SkillNotFoundError as exc:
                last_exc = exc
        raise last_exc or SkillNotFoundError(f"Resource '{resource_name}' not found for skill '{skill_name}'")

    async def read_skill_resource_for_user(self, *, user_id: str, skill_name: str, resource_name: str) -> str:
        return self.read_skill_resource(skill_name, resource_name)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        names: set[str] = set()
        for loader in self._loaders:
            names.update(loader.list_skill_resource_names(skill_name))
        return sorted(names)

    async def list_skill_resource_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name)

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        names: set[str] = set()
        for loader in self._loaders:
            names.update(loader.list_skill_script_names(skill_name))
        return sorted(names)

    async def list_skill_script_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]:
        return self.list_skill_script_names(skill_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        last_exc: Exception | None = None
        for loader in self._loaders:
            try:
                return await loader.run_skill_script(skill_name, script_name, arguments)
            except SkillNotFoundError as exc:
                last_exc = exc
        raise last_exc or SkillNotFoundError(f"Script '{script_name}' not found for skill '{skill_name}'")

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name, script_name, arguments)
