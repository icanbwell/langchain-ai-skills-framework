from __future__ import annotations

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.mongo_user_skill_loader import (
    MongoUserSkillLoader,
)
from langchain_ai_skills_framework.loaders.null_user_skill_store import (
    NullUserSkillStore,
)
from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.persistence.mongo_database_factory import (
    MongoDatabaseFactory,
)


class UserSkillStoreFactory:
    """Creates the appropriate ``UserSkillStore`` based on environment configuration."""

    def __init__(
        self,
        *,
        mongo_database_factory: MongoDatabaseFactory,
        environment_variables: LangchainAISkillsFrameworkEnvironmentVariables,
    ) -> None:
        self._mongo_database_factory = mongo_database_factory
        self._environment_variables = environment_variables

    def create(self) -> UserSkillStore:
        """Return a MongoDB-backed store when configured, otherwise a null store."""
        try:
            self._environment_variables.mongo_skills_uri
        except RuntimeError:
            return NullUserSkillStore()
        database = self._mongo_database_factory.create_database()
        return MongoUserSkillLoader(
            collection=database[MongoUserSkillLoader.COLLECTION_NAME],
            database=database,
        )
