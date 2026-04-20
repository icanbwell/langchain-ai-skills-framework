import logging
from typing import cast

from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.github_directory_downloader import (
    GithubDirectoryDownloader,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.marketplace_directory_loader import (
    MarketplaceDirectoryLoader,
)
from langchain_ai_skills_framework.loaders.multi_source_skill_loader import (
    MultiSourceSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_sync import SkillSync
from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.loaders.user_skill_store_factory import (
    UserSkillStoreFactory,
)
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

logger = logging.getLogger(__name__)


def _build_shared_loader(c: IContainer) -> SkillLoaderProtocol:
    """Build the shared skill loader, optionally including marketplace skills.

    When PLUGINS_MARKETPLACE is configured, creates a MultiSourceSkillLoader
    that merges the primary SkillkitDirectoryLoader (highest precedence) with
    a MarketplaceDirectoryLoader (lower precedence).
    """
    directory_loader = c.resolve(SkillkitDirectoryLoader)
    env_vars = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))

    marketplace_uri = env_vars.plugins_marketplace
    if not marketplace_uri:
        return directory_loader

    try:
        marketplace_loader = MarketplaceDirectoryLoader(
            environment_variables=env_vars,
            github_directory_downloader=c.resolve(GithubDirectoryDownloader),
        )
    except Exception:
        logger.exception(
            "Failed to initialize MarketplaceDirectoryLoader for '%s'; falling back to directory-only skills.",
            marketplace_uri,
        )
        return directory_loader

    return MultiSourceSkillLoader(
        loaders=[directory_loader, marketplace_loader],
    )


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *,
        container: SimpleContainer,
    ) -> SimpleContainer:

        container.singleton(GithubSkillDownloader, lambda c: GithubSkillDownloader())
        container.singleton(GithubDirectoryDownloader, lambda c: GithubDirectoryDownloader())

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

        container.singleton(
            UserSkillStoreFactory,
            lambda c: UserSkillStoreFactory(
                mongo_database_factory=c.resolve(MongoDatabaseFactory),
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
            ),
        )

        container.singleton(
            UserSkillStore,
            lambda c: c.resolve(UserSkillStoreFactory).create(),
        )

        # Register the shared loader as a singleton so CompositeSkillLoader
        # and SkillSync share the same instance (avoiding redundant downloads
        # and inconsistent cache state).
        container.singleton(
            MultiSourceSkillLoader,
            lambda c: _build_shared_loader(c),
        )

        container.singleton(
            CompositeSkillLoader,
            lambda c: CompositeSkillLoader(
                shared_loader=c.resolve(MultiSourceSkillLoader),
                user_loader=c.resolve(UserSkillStore),
            ),
        )

        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(CompositeSkillLoader),
        )

        container.singleton(
            SkillSync,
            lambda c: SkillSync(
                shared_loader=c.resolve(MultiSourceSkillLoader),
                user_store=c.resolve(UserSkillStore),
            ),
        )

        return container
