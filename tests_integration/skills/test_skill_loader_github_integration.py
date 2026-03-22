from __future__ import annotations

import os

import pytest

from langchain_ai_skills_framework.cache.skill_cache import SkillCache
from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader import SkillDirectoryLoader


def test_skill_loader_reads_skills_from_github_and_prints_parsed_summaries() -> None:
    skills_directory = os.environ.get("SKILLS_DIRECTORY", "").strip()
    github_token = (
        os.environ.get("SKILLS_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    ).strip()

    if not skills_directory:
        pytest.skip(
            "Set SKILLS_DIRECTORY to a github:// URI to run GitHub integration test"
        )
    if not skills_directory.startswith("github://"):
        pytest.skip("SKILLS_DIRECTORY must use a github:// URI")
    if not github_token:
        pytest.skip("Set SKILLS_GITHUB_TOKEN or GITHUB_TOKEN to run this test")

    environment_variables = LangchainAISkillsFrameworkEnvironmentVariables()
    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=environment_variables,
    )

    if not loader._path_exists(loader._skills_path):
        pytest.skip(f"Configured skills directory does not exist: {skills_directory}")
    if not loader._is_dir(loader._skills_path):
        pytest.skip(f"Configured skills directory is not a directory: {skills_directory}")

    summaries = loader.list_skill_summaries()
    assert len(summaries) > 0

    print("Parsed skills from GitHub:")
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

    details = loader.get_skill_details(summaries[0].name)
    assert details.name == summaries[0].name
    assert details.source_path.name == "SKILL.md"
