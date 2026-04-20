from typing import Protocol, runtime_checkable


@runtime_checkable
class SkillLoaderEnvironmentVariables(Protocol):
    """Environment contract for skill loading configuration."""

    @property
    def skills_cache_timeout_seconds(self) -> int: ...

    @property
    def excluded_skills(self) -> set[str]: ...

    @property
    def excluded_skill_groups(self) -> set[str]: ...

    @property
    def skills_directory(self) -> str:
        """Base location where skills are discovered.

        Examples:
        - Local filesystem: "/opt/app/skills"
        - GitHub via pydantic-ai-skills: "github://my-org/private-skills/skills?ref=main"
        """
        ...

    @property
    def plugins_marketplace(self) -> str | None:
        """Optional github:// URI to a Claude plugin marketplace repository.

        When set, skills are also loaded from the marketplace structure
        (plugins/*/skills/) in addition to the primary skills_directory.

        Example:
        - "github://my-org/claude-plugin-marketplace/plugins?ref=main"
        """
        ...

    @property
    def skills_github_token(self) -> str | None:
        """Optional token used for authenticated GitSkillsRegistry loading.

        The value may be a fine-grained PAT or a short-lived GitHub App
        installation token.

        Expected environment variables:
        - SKILLS_GITHUB_TOKEN (preferred)
        - GITHUB_TOKEN (fallback)
        """
        ...
