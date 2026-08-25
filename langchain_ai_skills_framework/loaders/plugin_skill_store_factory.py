"""Factory for creating the appropriate ``PluginSkillStore`` implementation."""

from __future__ import annotations

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.history_tracking_plugin_skill_store import (
    HistoryTrackingPluginSkillStore,
)
from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import (
    MongoPluginSkillLoader,
)
from langchain_ai_skills_framework.loaders.null_plugin_skill_store import (
    NullPluginSkillStore,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.persistence.error_writer import ErrorWriter
from langchain_ai_skills_framework.persistence.history_writer import HistoryWriter
from langchain_ai_skills_framework.persistence.mongo_database_factory import (
    MongoDatabaseFactory,
)


class PluginSkillStoreFactory:
    """Creates the appropriate ``PluginSkillStore`` based on environment configuration."""

    def __init__(
        self,
        *,
        mongo_database_factory: MongoDatabaseFactory,
        environment_variables: LangchainAISkillsFrameworkEnvironmentVariables,
    ) -> None:
        self._mongo_database_factory = mongo_database_factory
        self._environment_variables = environment_variables

    def create(self) -> PluginSkillStore:
        """Return a MongoDB-backed store when configured, otherwise a null store.

        When history tracking is enabled (ENABLE_SKILL_HISTORY=true, the default),
        wraps the store with HistoryTrackingPluginSkillStore.
        """
        try:
            _ = self._environment_variables.mongo_skills_uri
        except RuntimeError:
            return NullPluginSkillStore()

        database = self._mongo_database_factory.create_database()
        inner_store = MongoPluginSkillLoader(
            database=database,
            skills_collection_name=self._environment_variables.plugin_skills_collection or "plugin_skills",
            references_collection_name=self._environment_variables.plugin_references_collection or "plugin_references",
            scripts_collection_name=self._environment_variables.plugin_scripts_collection or "plugin_scripts",
            plugins_collection_name=self._environment_variables.plugins_collection or "plugins",
        )

        if not self._environment_variables.enable_skill_history:
            return inner_store

        history_writer = HistoryWriter(
            database=database,
            skills_history_collection_name=self._environment_variables.plugin_skills_history_collection,
            scripts_history_collection_name=self._environment_variables.plugin_scripts_history_collection,
            references_history_collection_name=self._environment_variables.plugin_references_history_collection,
            plugins_history_collection_name=self._environment_variables.plugins_history_collection,
        )

        error_writer = ErrorWriter(
            database=database,
            errors_collection_name=self._environment_variables.plugin_errors_collection,
        )

        return HistoryTrackingPluginSkillStore(
            inner_store=inner_store,
            history_writer=history_writer,
            error_writer=error_writer,
        )
