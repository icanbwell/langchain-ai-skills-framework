import os
from pathlib import Path

from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillLoaderEnvironmentVariables,
)


class LangchainAISkillsFrameworkEnvironmentVariables(
    EnvironmentVariables, SkillLoaderEnvironmentVariables
):
    @property
    def skills_github_token(self) -> str | None:
        """Optional token used for authenticated github:// skill loading.

        Supports fine-grained PATs and GitHub App installation tokens.
        """
        token = os.environ.get("SKILLS_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token is None or not token.strip():
            return None
        return token.strip()

    @property
    def skills_directory(self) -> str:
        """Return the absolute path to the Agent Skills directory."""

        configured = os.environ.get("SKILLS_DIRECTORY")
        if configured and configured.strip():
            return configured

        # Attempt to infer a sensible default based on the current package layout.
        # This file lives at: <repo_root>/langchain_ai_skills_framework/environment/environment_variables.py
        package_root = Path(__file__).resolve().parents[1]
        repo_root = package_root.parent

        candidate_dirs = [
            repo_root / "skills",
            repo_root / "skills" / "skills",
            package_root / "skills",
        ]

        for candidate in candidate_dirs:
            if candidate.is_dir():
                return str(candidate)

        raise RuntimeError(
            "SKILLS_DIRECTORY environment variable is not set and no default skills "
            "directory could be found. Please set SKILLS_DIRECTORY to the directory "
            "containing your Agent Skills."
        )

    @property
    def excluded_skills(self) -> set[str]:
        """List of skill names to skip when loading Agent Skills."""
        raw_value = os.environ.get("SKILLS_EXCLUDED")
        if not raw_value or not raw_value.strip():
            return set()
        return {item.strip() for item in raw_value.split(",") if item.strip()}

    @property
    def excluded_skill_groups(self) -> set[str]:
        """List of skill group names to skip when loading Agent Skills."""
        raw_value = os.environ.get("SKILL_GROUPS_EXCLUDED")
        if not raw_value or not raw_value.strip():
            return set()
        return {item.strip() for item in raw_value.split(",") if item.strip()}
