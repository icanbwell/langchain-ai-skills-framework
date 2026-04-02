from pathlib import Path

from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.github_directory_downloader import (
    GithubDirectoryDownloader,
    GitLocation,
)

# Re-export so existing ``from github_skill_downloader import GitLocation`` keeps working.
__all__ = ["GithubSkillDownloader", "GitLocation"]


class GithubSkillDownloader(GithubDirectoryDownloader):
    """Backward-compatible subclass that raises SkillValidationError."""

    def download(  # type: ignore[override]
        self,
        *,
        skills_directory: str,
        github_token: str | None,
        cache_path: Path,
    ) -> Path:
        try:
            return super().download(
                source_uri=skills_directory,
                github_token=github_token,
                cache_path=cache_path,
            )
        except ValueError as exc:
            raise SkillValidationError(str(exc)) from exc

    @classmethod
    def parse_github_uri(cls, skills_directory: str) -> GitLocation:
        try:
            return GithubDirectoryDownloader.parse_github_uri(skills_directory)
        except ValueError as exc:
            raise SkillValidationError(str(exc)) from exc
