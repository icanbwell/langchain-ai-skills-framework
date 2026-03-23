from typing import Protocol, runtime_checkable

from langchain_core.tools import StructuredTool


@runtime_checkable
class SkillsToolManagerProtocol(Protocol):
    def get_tools(self) -> list[StructuredTool]: ...
