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
    def plugins_marketplace(self) -> str | None:
        """URI to a Claude plugin marketplace repository.

        Skills are loaded from the marketplace structure
        (plugins/*/skills/).

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
    def plugin_skills_collection(self) -> str | None:
        """MongoDB collection name for plugin-scoped skill documents.

        Expected environment variable: PLUGIN_SKILLS_COLLECTION
        Default: "plugin_skills"
        """
        ...

    @property
    def plugin_references_collection(self) -> str | None:
        """MongoDB collection name for plugin-scoped resource documents.

        Expected environment variable: PLUGIN_REFERENCES_COLLECTION
        Default: "plugin_references"
        """
        ...

    @property
    def plugin_scripts_collection(self) -> str | None:
        """MongoDB collection name for plugin-scoped script documents.

        Expected environment variable: PLUGIN_SCRIPTS_COLLECTION
        Default: "plugin_scripts"
        """
        ...

    @property
    def plugins_marketplace_cache_folder(self) -> str | None:
        """Local directory for caching github:// marketplace downloads.

        When set, the marketplace downloader stores its git cache here
        instead of the default ``.marketplace-git-cache`` relative path.
        Supports ``{pid}`` placeholder for per-worker isolation under
        multi-worker deployments.

        Expected environment variable: PLUGINS_MARKETPLACE_CACHE_FOLDER
        Example: "/usr/src/marketplace_cache/{pid}"
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
