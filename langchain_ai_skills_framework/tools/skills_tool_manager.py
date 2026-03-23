from langchain_core.tools import StructuredTool

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.tools.skills_tool_manager_protocol import (
    SkillsToolManagerProtocol,
)


class SkillsToolManager(SkillsToolManagerProtocol):
    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self.skill_loader = skill_loader

    def get_tools(self) -> list[StructuredTool]:
        return self.skill_loader.get_tools()
