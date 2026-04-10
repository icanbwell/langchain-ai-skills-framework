from __future__ import annotations

import os

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

    def __init__(self, *, mongo_database_factory: MongoDatabaseFactory) -> None:
        self._mongo_database_factory = mongo_database_factory

    def create(self) -> UserSkillStore:
        """Return a MongoDB-backed store when configured, otherwise a null store."""
        if os.environ.get("MONGO_SKILLS_URI") or os.environ.get("MONGO_URL"):
            return MongoUserSkillLoader(
                collection=self._mongo_database_factory.create_database()[
                    MongoUserSkillLoader.COLLECTION_NAME
                ],
            )
        return NullUserSkillStore()
