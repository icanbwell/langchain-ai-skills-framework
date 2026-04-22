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
    def skills_directory(self) -> str | None:
        """Base location where skills are discovered.

        Returns None when no directory is configured, allowing the system
        to operate with marketplace-only or user-only skills.

        Examples:
        - Local filesystem: "/opt/app/skills"
        - GitHub via pydantic-ai-skills: "github://my-org/private-skills/skills?ref=main"
        - None: no primary skills directory configured
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
    def plugins_marketplace_include(self) -> set[str] | None:
        """Optional allowlist of plugin names to include from the marketplace.

        When set, only plugins whose directory name matches an entry in this
        set are loaded. When None/empty, all plugins are included (subject
        to the exclude list).

        Expected environment variable: PLUGINS_MARKETPLACE_INCLUDE (comma-separated)
        """
        ...

    @property
    def plugins_marketplace_exclude(self) -> set[str]:
        """Plugin names to exclude from the marketplace.

        Plugins whose directory name matches an entry in this set are skipped.
        Applied after the include list (if set).

        Expected environment variable: PLUGINS_MARKETPLACE_EXCLUDE (comma-separated)
        """
        ...

    @property
    def snapshot_cache_skills_collection(self) -> str | None:
        """Optional MongoDB collection for skillkit directory snapshots.

        When set, SkillkitDirectoryLoader stores its snapshot in this
        collection instead of the store's default collection.

        Expected environment variable: SNAPSHOT_CACHE_SKILLS_COLLECTION
        """
        ...

    @property
    def snapshot_cache_plugins_collection(self) -> str | None:
        """Optional MongoDB collection for marketplace plugin snapshots.

        When set, MarketplaceDirectoryLoader stores its snapshot in this
        collection instead of the store's default collection.

        Expected environment variable: SNAPSHOT_CACHE_PLUGINS_COLLECTION
        """
        ...

    @property
    def plugins_collection(self) -> str | None:
        """MongoDB collection for individual plugin definition documents.

        Each plugin discovered from the marketplace is written as a
        separate document to this collection (keyed by plugin name).

        Expected environment variable: PLUGINS_COLLECTION
        Default: "plugins"
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
