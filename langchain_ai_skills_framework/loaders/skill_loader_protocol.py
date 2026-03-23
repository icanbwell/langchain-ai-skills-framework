from typing import Protocol, Sequence, runtime_checkable

from langchain_core.tools import StructuredTool

from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary


@runtime_checkable
class SkillLoaderProtocol(Protocol):
    def list_skill_summaries(
        self, allowed_skills: set[str]
    ) -> Sequence[SkillSummary]: ...

    def get_skill_details(self, skill_name: str) -> SkillDetails: ...

    def refresh(self) -> None: ...

    async def get_instructions(self) -> str: ...

    def get_tools(self) -> list[StructuredTool]: ...
