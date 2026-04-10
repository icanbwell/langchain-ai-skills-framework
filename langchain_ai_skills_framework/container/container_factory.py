from typing import Optional

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.mongo_user_skill_loader import (
    MongoUserSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_directory_loader import (
    SkillDirectoryLoader,
)
from simple_container.container.simple_container import SimpleContainer
from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
)
from langchain_ai_skills_framework.tools.skills_tool_manager import SkillsToolManager


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *,
        container: SimpleContainer,
        mongo_database: Optional[AsyncIOMotorDatabase] = None,  # type: ignore[type-arg]
    ) -> SimpleContainer:

        container.singleton(GithubSkillDownloader, lambda c: GithubSkillDownloader())

        container.singleton(
            SkillDirectoryLoader,
            lambda c: SkillDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
                github_skill_downloader=c.resolve(GithubSkillDownloader),
            ),
        )

        container.singleton(
            SkillkitDirectoryLoader,
            lambda c: SkillkitDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
                github_skill_downloader=c.resolve(GithubSkillDownloader),
            ),
        )

        if mongo_database is not None:
            collection: AsyncIOMotorCollection = mongo_database[  # type: ignore[type-arg]
                MongoUserSkillLoader.COLLECTION_NAME
            ]

            container.singleton(
                MongoUserSkillLoader,
                lambda c: MongoUserSkillLoader(collection=collection),
            )

            container.singleton(
                CompositeSkillLoader,
                lambda c: CompositeSkillLoader(
                    shared_loader=c.resolve(SkillkitDirectoryLoader),
                    user_loader=c.resolve(MongoUserSkillLoader),
                ),
            )

            container.singleton(
                SkillLoaderProtocol,
                lambda c: c.resolve(CompositeSkillLoader),
            )
        else:
            container.singleton(
                SkillLoaderProtocol,
                lambda c: c.resolve(SkillkitDirectoryLoader),
            )

        container.singleton(
            SkillsToolManager,
            lambda c: SkillsToolManager(
                skill_loader=c.resolve(SkillLoaderProtocol),
            ),
        )

        return container
