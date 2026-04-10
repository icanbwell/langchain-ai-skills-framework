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
from langchain_ai_skills_framework.persistence.mongo_database_factory import (
    MongoDatabaseFactory,
)
from langchain_ai_skills_framework.persistence.mongo_database_factory_impl import (
    MongoDatabaseFactoryImpl,
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

from typing import Any


def _is_registered(container: SimpleContainer, service_type: type[Any]) -> bool:
    """Check whether *service_type* already has a factory in *container*.

    ``IContainer`` declares ``_factories`` on its protocol, so the
    attribute is part of the public contract despite the underscore.
    We isolate access here so callers don't spread the coupling.
    """
    return service_type in container._factories


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

        container.singleton(
            SkillkitDirectoryLoader,
            lambda c: SkillkitDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
                github_skill_downloader=c.resolve(GithubSkillDownloader),
            ),
        )

        # Register the default MongoDatabaseFactory only if the consuming
        # application has not already registered its own implementation.
        if not _is_registered(container, MongoDatabaseFactory):
            container.singleton(
                MongoDatabaseFactory,
                lambda c: MongoDatabaseFactoryImpl(
                    environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
                ),
            )

        container.singleton(
            MongoUserSkillLoader,
            lambda c: MongoUserSkillLoader(
                collection=c.resolve(MongoDatabaseFactory).create_database()[
                    MongoUserSkillLoader.COLLECTION_NAME
                ]
            ),
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

        container.singleton(
            SkillsToolManager,
            lambda c: SkillsToolManager(
                skill_loader=c.resolve(SkillLoaderProtocol),
            ),
        )

        return container
