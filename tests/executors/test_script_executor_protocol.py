from __future__ import annotations

from langchain_ai_skills_framework.executors.my_script_executor import MyScriptExecutor
from langchain_ai_skills_framework.executors.script_executor_protocol import (
    ScriptExecutorProtocol,
)


class TestScriptExecutorProtocol:
    def test_my_script_executor_satisfies_protocol(self) -> None:
        assert isinstance(MyScriptExecutor(), ScriptExecutorProtocol)
