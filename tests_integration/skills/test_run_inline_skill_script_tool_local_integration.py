from __future__ import annotations

import json
from pathlib import Path
from typing import cast, override

import pytest

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
)
from langchain_ai_skills_framework.tools.run_inline_skill_script_tool import (
    RunInlineSkillScriptTool,
)


class _LocalSkillsEnvironmentVariables(LangchainAISkillsFrameworkEnvironmentVariables):
    @override
    @property
    def skills_directory(self) -> str:
        return str(Path(__file__).parent.joinpath("skills").absolute())


def _build_loader() -> SkillkitDirectoryLoader:
    return SkillkitDirectoryLoader(
        environment_variables=_LocalSkillsEnvironmentVariables(),
        github_skill_downloader=GithubSkillDownloader(),
    )


def _get_inline_script_tool(
    loader: SkillkitDirectoryLoader,
) -> RunInlineSkillScriptTool:
    for tool in loader.get_tools():
        if tool.name == "run_inline_skill_script":
            return cast(RunInlineSkillScriptTool, tool)
    raise AssertionError("run_inline_skill_script tool is not registered")


@pytest.mark.asyncio
async def test_run_inline_skill_script_tool_executes_inline_script_locally() -> None:
    loader = _build_loader()
    tool = _get_inline_script_tool(loader)

    inline_script = """
import json
import sys

args = json.loads(sys.stdin.read() or "{}")
result = {
    "received_keys": sorted(args.keys()),
    "mixedcase": args.get("mixedcase"),
}
print(json.dumps(result))
""".strip()

    response = await tool.ainvoke(
        {
            "script": inline_script,
            "arguments": {"MixedCase": "VALUE"},
        }
    )

    payload = json.loads(response)
    assert payload["received_keys"] == ["mixedcase"]
    assert payload["mixedcase"] == "VALUE"


@pytest.mark.asyncio
async def test_run_inline_skill_script_tool_rejects_legacy_name_fields() -> None:
    loader = _build_loader()
    tool = _get_inline_script_tool(loader)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        await tool.ainvoke(
            {
                "skill_name": "skill-with-references",
                "script_name": "inline_script.py",
                "script": "print('ok')",
                "arguments": None,
            }
        )
