from typing import Protocol, runtime_checkable

from langchain_core.tools import BaseTool


@runtime_checkable
class SkillsToolManagerProtocol(Protocol):
    def get_tools(self) -> list[BaseTool]: ...
