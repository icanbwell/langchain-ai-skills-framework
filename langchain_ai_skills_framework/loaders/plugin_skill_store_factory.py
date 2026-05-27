"""Factory for creating the appropriate ``PluginSkillStore`` implementation."""

from __future__ import annotations

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import (
    MongoPluginSkillLoader,
)
from langchain_ai_skills_framework.loaders.null_plugin_skill_store import (
    NullPluginSkillStore,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
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
        """Return a MongoDB-backed store when configured, otherwise a null store."""
        try:
            self._environment_variables.mongo_skills_uri
        except RuntimeError:
            return NullPluginSkillStore()

        database = self._mongo_database_factory.create_database()
        return MongoPluginSkillLoader(
            database=database,
            schema_version=self._environment_variables.skill_cache_schema_version,
            skills_collection_name=self._environment_variables.plugin_skills_collection or "plugin_skills",
            references_collection_name=self._environment_variables.plugin_references_collection or "plugin_references",
            scripts_collection_name=self._environment_variables.plugin_scripts_collection or "plugin_scripts",
            plugins_collection_name=self._environment_variables.plugins_collection or "plugins",
        )
