import os

from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.mongo_user_skill_loader import (
    MongoUserSkillLoader,
)
from langchain_ai_skills_framework.loaders.null_user_skill_store import (
    NullUserSkillStore,
)
from langchain_ai_skills_framework.loaders.skill_directory_loader import (
    SkillDirectoryLoader,
)
from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.persistence.mongo_database_factory import (
    MongoDatabaseFactory,
)
from langchain_ai_skills_framework.persistence.mongo_database_factory_impl import (
    MongoDatabaseFactoryImpl,
)
from simple_container.container.interfaces import IContainer
from simple_container.container.simple_container import SimpleContainer
from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
)
from langchain_ai_skills_framework.tools.skills_tool_manager import SkillsToolManager


def _is_mongo_configured() -> bool:
    """Return True when a MongoDB connection URI is available."""
    return bool(
        (
            os.environ.get("MONGO_SKILLS_URI") or os.environ.get("MONGO_URL") or ""
        ).strip()
    )


def _create_user_skill_store(container: IContainer) -> UserSkillStore:
    """Return a MongoDB-backed store when configured, otherwise a null store."""
    if _is_mongo_configured():
        return MongoUserSkillLoader(
            collection=container.resolve(MongoDatabaseFactory).create_database()[
                MongoUserSkillLoader.COLLECTION_NAME
            ],
        )
    return NullUserSkillStore()


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *,
        container: SimpleContainer,
    ) -> SimpleContainer:

        container.singleton(GithubSkillDownloader, lambda c: GithubSkillDownloader())

        container.singleton(
            SkillDirectoryLoader,
            lambda c: SkillDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
                github_skill_downloader=c.resolve(GithubSkillDownloader),
            ),
        )

        # GitHub/filesystem skills are always loaded first.
        container.singleton(
            SkillkitDirectoryLoader,
            lambda c: SkillkitDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
                github_skill_downloader=c.resolve(GithubSkillDownloader),
            ),
        )

        container.singleton(
            MongoDatabaseFactory,
            lambda c: MongoDatabaseFactoryImpl(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
            ),
        )

        container.singleton(UserSkillStore, _create_user_skill_store)

        container.singleton(
            CompositeSkillLoader,
            lambda c: CompositeSkillLoader(
                shared_loader=c.resolve(SkillkitDirectoryLoader),
                user_loader=c.resolve(UserSkillStore),
            ),
        )

        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(CompositeSkillLoader),
        )

        container.singleton(
            SkillsToolManager,
            lambda c: SkillsToolManager(
                skill_loader=c.resolve(SkillLoaderProtocol),
            ),
        )

        return container
