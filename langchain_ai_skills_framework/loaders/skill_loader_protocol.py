from typing import Protocol, Sequence, runtime_checkable, Any

from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary


@runtime_checkable
class SkillLoaderProtocol(Protocol):
    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]: ...

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]: ...

    def get_skill_details(self, skill_name: str) -> SkillDetails: ...

    async def get_skill_details_for_user(self, *, user_id: str, skill_name: str) -> SkillDetails: ...

    def refresh(self) -> None: ...

    async def get_instructions(self) -> str: ...

    def get_tools(self) -> list[BaseTool]: ...

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str: ...

    async def read_skill_resource_for_user(self, *, user_id: str, skill_name: str, resource_name: str) -> str: ...

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult: ...

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult: ...

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]: ...

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]: ...

    async def list_skill_resource_names_for_user(self, *, user_id: str, skill_name: str) -> Sequence[str]: ...
