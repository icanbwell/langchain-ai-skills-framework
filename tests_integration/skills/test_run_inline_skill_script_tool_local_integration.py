from __future__ import annotations

from pathlib import Path
from typing import override


from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
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
