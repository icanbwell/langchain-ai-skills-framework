from typing import Protocol, runtime_checkable


@runtime_checkable
class SkillLoaderEnvironmentVariables(Protocol):
    """Environment contract for skill loading configuration."""

    @property
    def excluded_skills(self) -> set[str]: ...

    @property
    def excluded_skill_groups(self) -> set[str]: ...

    @property
    def skills_directory(self) -> str:
        """Base location where skills are discovered.

        Examples:
        - Local filesystem: "/opt/app/skills"
        - GitHub via fsspec: "github://skill-repo@main/skills"
        """
        ...

    @property
    def skills_github_token(self) -> str | None:
        """Optional token used for authenticated github:// skill loading.

        The value may be a fine-grained PAT or a short-lived GitHub App
        installation token.

        Expected environment variables:
        - SKILLS_GITHUB_TOKEN (preferred)
        - GITHUB_TOKEN (fallback)
        """
        ...
