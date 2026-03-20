from langchain_ai_skills_framework.loaders.exceptions.skill_loader_error import (
    SkillLoaderError,
)


class SkillValidationError(SkillLoaderError):
    """Raised when a skill definition violates the Agent Skills specification."""
