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

    @property
    def github_app_id(self) -> str | None:
        """GitHub App ID for installation token authentication.

        When set alongside github_app_private_key and github_app_installation_id,
        the system uses GitHub App authentication instead of a static PAT.

        Expected environment variable: GITHUB_APP_ID
        """
        ...

    @property
    def github_app_private_key(self) -> str | None:
        """PEM-encoded RSA private key for the GitHub App.

        Used to sign JWTs for minting installation tokens.

        Expected environment variable: GITHUB_APP_PRIVATE_KEY
        """
        ...

    @property
    def github_app_installation_id(self) -> str | None:
        """Installation ID for the GitHub App on the target org/repo.

        Expected environment variable: GITHUB_APP_INSTALLATION_ID
        """
        ...

    @property
    def plugins_marketplace_publish_enabled(self) -> bool:
        """Whether marketplace publishing is enabled.

        When ``False``, the publisher is not created even if a github://
        marketplace URI and token are present.

        Expected environment variable: PLUGINS_MARKETPLACE_PUBLISH_ENABLED
        Default: ``false``
        """
        ...

    @property
    def plugins_marketplace_publish_branch(self) -> str:
        """Base branch for marketplace skill publish commits/PRs.

        Expected environment variable: PLUGINS_MARKETPLACE_PUBLISH_BRANCH
        Default: ``main``
        """
        ...

    @property
    def plugins_marketplace_publish_use_branch(self) -> bool:
        """Whether to publish via a PR branch or commit directly to base.

        When ``True`` (default), a deterministic branch is created and a
        pull request is opened or updated.  When ``False``, the commit is
        pushed directly to the base branch.

        Expected environment variable: PLUGINS_MARKETPLACE_PUBLISH_USE_BRANCH
        Default: ``true``
        """
        ...
