from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.tools.skills_tool import LoadSkillTool


class SkillsToolManager:
    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self.skill_loader = skill_loader

    def get_tools(self) -> list[BaseTool]:
        return [
            LoadSkillTool(
                skill_loader=self.skill_loader,
            ),
        ]
