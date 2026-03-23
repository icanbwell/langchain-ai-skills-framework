from langchain_ai_skills_framework.loaders.exceptions.skill_loader_error import (
    SkillLoaderError,
)


class SkillNotFoundError(SkillLoaderError):
    """Raised when a requested skill cannot be found."""
