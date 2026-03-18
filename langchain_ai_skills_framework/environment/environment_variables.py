import os
from pathlib import Path

from simple_container.environment.environment_variables import EnvironmentVariables


class LangchainAISkillsFrameworkEnvironmentVariables(EnvironmentVariables):
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
