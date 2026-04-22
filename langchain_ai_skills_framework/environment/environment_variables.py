import os
import logging

from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)
_SKILLS_CACHE_TIMEOUT_ENV_VAR: str = "SKILLS_CACHE_TIMEOUT_SECONDS"
_DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS: int = 60 * 60


class LangchainAISkillsFrameworkEnvironmentVariables(EnvironmentVariables, SkillLoaderEnvironmentVariables):
    @property
    def skills_cache_timeout_seconds(self) -> int:
        """Return a validated TTL in seconds for skill reload behavior."""

        raw_value = os.getenv(_SKILLS_CACHE_TIMEOUT_ENV_VAR)
        if raw_value is None:
            return _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS
        try:
            ttl_seconds = int(raw_value)
        except ValueError:
            _LOGGER.warning(
                "Invalid %s value %r; using default %d seconds",
                _SKILLS_CACHE_TIMEOUT_ENV_VAR,
                raw_value,
                _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS,
            )
            return _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS
        if ttl_seconds <= 0:
            _LOGGER.warning(
                "%s must be a positive integer; got %d. Using default %d seconds",
                _SKILLS_CACHE_TIMEOUT_ENV_VAR,
                ttl_seconds,
                _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS,
            )
            return _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS
        return ttl_seconds

    @property
    def snapshot_cache_plugins_collection(self) -> str | None:
        return os.environ.get("SNAPSHOT_CACHE_PLUGINS_COLLECTION") or None

    @property
    def plugins_collection(self) -> str | None:
        return os.environ.get("PLUGINS_COLLECTION") or "plugins"

    @property
    def plugin_skills_collection(self) -> str | None:
        return os.environ.get("PLUGIN_SKILLS_COLLECTION") or "plugin_skills"

    @property
    def plugin_references_collection(self) -> str | None:
        return os.environ.get("PLUGIN_REFERENCES_COLLECTION") or "plugin_references"

    @property
    def plugin_scripts_collection(self) -> str | None:
        return os.environ.get("PLUGIN_SCRIPTS_COLLECTION") or "plugin_scripts"

    @property
    def skills_github_token(self) -> str | None:
        """Optional token used for authenticated github:// skill loading.

        Supports fine-grained PATs and GitHub App installation tokens.
        """
        token = os.environ.get("SKILLS_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token is None or not token.strip():
            return None
        return token.strip()

    @staticmethod
    def _resolve_path(value: str | None) -> str | None:
        """Replace ``{pid}`` with the current process ID.

        When multiple gunicorn workers share the same environment, this
        gives each worker its own directory tree so they don't collide
        on reads/writes.
        """
        if value and "{pid}" in value:
            return value.replace("{pid}", str(os.getpid()))
        return value

    @property
    def plugins_marketplace_cache_folder(self) -> str | None:
        """Local directory for caching github:// marketplace downloads.

        Supports {pid} substitution for per-worker isolation.
        """
        value = self._resolve_path(os.environ.get("PLUGINS_MARKETPLACE_CACHE_FOLDER"))
        if not value or not value.strip():
            return None
        return value.strip()

    @property
    def plugins_marketplace(self) -> str | None:
        """Optional github:// URI to a Claude plugin marketplace repository."""
        value = self._resolve_path(os.environ.get("PLUGINS_MARKETPLACE"))
        if not value or not value.strip():
            return None
        return value.strip()

    @property
    def plugins_marketplace_include(self) -> set[str] | None:
        """Allowlist of plugin names to load from the marketplace.

        When set, only matching plugins are loaded. When unset, all are loaded.
        """
        raw_value = os.environ.get("PLUGINS_MARKETPLACE_INCLUDE")
        if not raw_value or not raw_value.strip():
            return None
        return {item.strip() for item in raw_value.split(",") if item.strip()}

    @property
    def plugins_marketplace_exclude(self) -> set[str]:
        """Plugin names to exclude from the marketplace."""
        raw_value = os.environ.get("PLUGINS_MARKETPLACE_EXCLUDE")
        if not raw_value or not raw_value.strip():
            return set()
        return {item.strip() for item in raw_value.split(",") if item.strip()}

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

    @property
    def mongo_skills_uri(self) -> str:
        """MongoDB connection URI for skills storage.

        Falls back to ``MONGO_URL`` when the skills-specific variable is
        not set, matching the pattern used by other icanbwell services.
        """
        uri = os.environ.get("MONGO_SKILLS_URI") or os.environ.get("MONGO_URL")
        if not uri or not uri.strip():
            raise RuntimeError(
                "Neither MONGO_SKILLS_URI nor MONGO_URL is set. "
                "A MongoDB connection URI is required for user skill storage."
            )
        return uri.strip()

    @property
    def mongo_skills_db_name(self) -> str:
        """Database name for skills storage (default: ``llm_storage``)."""
        return os.environ.get("MONGO_SKILLS_DB_NAME") or os.environ.get("MONGO_DB_NAME") or "llm_storage"

    @property
    def mongo_skills_db_username(self) -> str | None:
        """Username for MongoDB skills storage authentication."""
        val = os.environ.get("MONGO_SKILLS_DB_USERNAME") or os.environ.get("MONGO_DB_USERNAME")
        if not val or not val.strip():
            return None
        return val.strip()

    @property
    def mongo_skills_db_password(self) -> str | None:
        """Password for MongoDB skills storage authentication."""
        val = os.environ.get("MONGO_SKILLS_DB_PASSWORD") or os.environ.get("MONGO_DB_PASSWORD")
        if not val or not val.strip():
            return None
        return val.strip()
