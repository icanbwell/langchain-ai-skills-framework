from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.persistence.mongo_url_helpers import (
    MongoUrlHelpers,
)


class MongoDatabaseFactoryImpl:
    """Concrete ``MongoDatabaseFactory`` backed by environment variables.

    Reads connection details from
    :class:`LangchainAISkillsFrameworkEnvironmentVariables` and lazily
    creates a single ``AsyncIOMotorClient`` on first use.
    """

    def __init__(
        self,
        *,
        environment_variables: LangchainAISkillsFrameworkEnvironmentVariables,
    ) -> None:
        if environment_variables is None:
            raise ValueError("environment_variables must not be None")
        self._env = environment_variables
        self._client: AsyncIOMotorClient | None = None  # type: ignore[type-arg]

    def create_database(self) -> AsyncIOMotorDatabase:  # type: ignore[type-arg]
        """Return an ``AsyncIOMotorDatabase`` for skills storage."""
        if self._client is None:
            connection_string = MongoUrlHelpers.add_credentials_to_mongo_url(
                mongo_url=self._env.mongo_skills_uri,
                username=self._env.mongo_skills_db_username,
                password=self._env.mongo_skills_db_password,
            )
            self._client = AsyncIOMotorClient(connection_string)
        return self._client[self._env.mongo_skills_db_name]