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

    When SKILLS_DIRECTORY is not configured, the directory loader is skipped
    and only marketplace skills are loaded.
    """
    env_vars = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))
    loaders: list[SkillLoaderProtocol] = []

    # Primary directory loader (optional when skills_directory is None)
    if env_vars.skills_directory:
        try:
            loaders.append(c.resolve(SkillkitDirectoryLoader))
        except Exception:
            logger.exception("Failed to initialize SkillkitDirectoryLoader; skipping.")

    # Marketplace loader (optional when plugins_marketplace is set)
    marketplace_uri = env_vars.plugins_marketplace
    if marketplace_uri:
        try:
            # BaseStore is registered by language-model-common's container
            # factory.  It may not be available at this point if the skills
            # framework container runs first; treat as optional.
            from key_value.aio.stores.base import BaseStore
            snapshot_cache_store: BaseStore | None = None
            try:
                snapshot_cache_store = c.resolve(BaseStore)
            except Exception:
                pass

            marketplace_loader = MarketplaceDirectoryLoader(
                environment_variables=env_vars,
                github_directory_downloader=c.resolve(GithubDirectoryDownloader),
                snapshot_cache_store=snapshot_cache_store,
            )
            loaders.append(marketplace_loader)
        except Exception:
            logger.exception(
                "Failed to initialize MarketplaceDirectoryLoader for '%s'; skipping.",
                marketplace_uri,
            )

    if not loaders:
        logger.warning("No skill loaders configured (neither SKILLS_DIRECTORY nor PLUGINS_MARKETPLACE is set).")
        # Fall back to the directory loader which will raise a clear error on access
        return c.resolve(SkillkitDirectoryLoader)

    if len(loaders) == 1:
        return loaders[0]

    return MultiSourceSkillLoader(loaders=loaders)


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
