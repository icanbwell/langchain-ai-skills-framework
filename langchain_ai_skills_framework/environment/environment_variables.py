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
    def skills_directory(self) -> str:
        """Return the absolute path to the Agent Skills directory."""

        configured = os.environ.get("SKILLS_DIRECTORY")
        if configured and configured.strip():
            return configured

        # Compute repository root (three levels up from this file):
        # repo_root / baileyai / utilities / environment / baileyai_environment_variables.py
        repo_root = Path(__file__).resolve().parents[3]
        default_skills_dir = repo_root / "baileyai" / "skills" / "skills"
        if default_skills_dir.is_dir():
            return str(default_skills_dir)

        # Fallback to legacy Docker path for backward compatibility
        return "/usr/src/baileyai/baileyai/skills/skills"

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
