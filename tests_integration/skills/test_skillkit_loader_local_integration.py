from __future__ import annotations

import traceback
from pathlib import Path
from typing import override

from langchain_core.tools import StructuredTool

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
)


class TestLangchainAISkillsFrameworkEnvironmentVariables(
    LangchainAISkillsFrameworkEnvironmentVariables
):
    @override
    @property
    def skills_directory(self) -> str:
        # Override to point to local test skills directory instead of GitHub for this test
        return str(Path(__file__).parent.joinpath("skills").absolute())


async def test_skillkit_loader_reads_skills_from_local_and_prints_parsed_summaries() -> (
    None
):
    environment_variables = TestLangchainAISkillsFrameworkEnvironmentVariables()
    loader = SkillkitDirectoryLoader(
        environment_variables=environment_variables,
        github_skill_downloader=GithubSkillDownloader(),
    )

    try:
        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) > 0

        print("========= Skills ===========")
        for summary in summaries:
            print(
                "- name={name}, description={description}, source_path={source_path}, "
                "allowed_tools={allowed_tools}".format(
                    name=summary.name,
                    description=summary.description,
                    source_path=summary.source_path,
                    allowed_tools=",".join(summary.allowed_tools),
                )
            )
        print("===========================================")
        details = loader.get_skill_details(skill_name=summaries[0].name)
        assert details.name == summaries[0].name
        assert details.source_path.name == "SKILL.md"

        # test instructions
        instructions: str = await loader.get_instructions()
        print("========= Instructions ===========")
        print(instructions)
        print("===========================================")
        assert "<available_skills>" in instructions

        # test tools
        tools: list[StructuredTool] = loader.get_tools()

        print(f"========= Tools [{len(tools)}]  ===========")
        for tool in tools:
            print(tool)
        print("===========================================")

        assert len(tools) > 0

        # test reference
        resource: str = loader.read_skill_resource(
            skill_name="skill-with-references", resource_name="REFERENCE.md"
        )
        print("========= Resource REFERENCE.md ===========")
        print(f"{resource}")
        print("===========================================")
        assert "This is a reference" in resource

        # test scripts
        script_result: MyScriptExecutionResult = await loader.run_skill_script(
            skill_name="skill-with-references", script_name="extract.py", arguments=None
        )
        print("========= Script EXTRACT.md STDOUT ===========")
        print(f"{script_result.stdout}")
        print("========= Script EXTRACT.md STDERR ===========")
        print(f"{script_result.stderr}")
        print("===========================================")
        assert script_result.stdout is not None
        assert "Hello" in script_result.stdout

    except Exception:
        traceback.print_exc()
        raise
