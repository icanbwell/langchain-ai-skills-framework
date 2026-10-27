from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.services.run_skill_script_service import (
    RunSkillScriptService,
)


def _make_loader() -> AsyncMock:
    loader = AsyncMock(spec=SkillLoaderProtocol)
    loader.run_skill_script_for_user.return_value = MyScriptExecutionResult(
        stdout="done", stderr=None, exit_code=0, execution_time_ms=1.0, success=True
    )
    return loader


@pytest.mark.asyncio
async def test_non_default_timeout_is_forwarded_to_loader() -> None:
    loader = _make_loader()
    service = RunSkillScriptService(skill_loader=loader)

    content, _ = await service.execute(
        user_id="user-1", plugin_name="p", skill_name="s", script_name="run.py", arguments={"a": 1}, timeout=120
    )

    assert content == "done"
    loader.run_skill_script_for_user.assert_awaited_once_with(
        user_id="user-1", plugin_name="p", skill_name="s", script_name="run.py", arguments={"a": 1}, timeout=120
    )


@pytest.mark.asyncio
async def test_default_timeout_is_30() -> None:
    loader = _make_loader()
    service = RunSkillScriptService(skill_loader=loader)

    await service.execute(user_id="user-1", plugin_name="p", skill_name="s", script_name="run.py")

    assert loader.run_skill_script_for_user.await_args.kwargs["timeout"] == 30
