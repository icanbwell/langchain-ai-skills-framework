import logging
from typing import cast

from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.github_directory_downloader import (
    GithubDirectoryDownloader,
)
from langchain_ai_skills_framework.loaders.marketplace_directory_loader import (
    MarketplaceDirectoryLoader,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.plugin_skill_store_factory import (
    PluginSkillStoreFactory,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_sync import SkillSync
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

logger = logging.getLogger(__name__)


def _build_shared_loader(c: IContainer) -> SkillLoaderProtocol:
    """Build the shared skill loader from the plugin marketplace.

    Skills are loaded from the marketplace structure (plugins/*/skills/).
    The SkillkitDirectoryLoader (SKILLS_DIRECTORY) has been removed —
    all skills come from plugins.
    """
    env_vars = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))

    marketplace_uri = env_vars.plugins_marketplace
    if not marketplace_uri:
        raise RuntimeError(
            "PLUGINS_MARKETPLACE is not configured. A marketplace URI is required since skills are loaded from plugins."
        )

    # BaseStore is registered by language-model-common's container
    # factory.  It may not be available at this point if the skills
    # framework container runs first; treat as optional.
    from key_value.aio.stores.base import BaseStore

    snapshot_cache_store: BaseStore | None = None
    try:
        snapshot_cache_store = c.resolve(BaseStore)
    except Exception:
        logger.debug("SnapshotCacheStore not available; proceeding without cache.")

    return MarketplaceDirectoryLoader(
        environment_variables=env_vars,
        github_directory_downloader=c.resolve(GithubDirectoryDownloader),
        snapshot_cache_store=snapshot_cache_store,
    )


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *,
        container: SimpleContainer,
    ) -> SimpleContainer:

        container.singleton(GithubDirectoryDownloader, lambda c: GithubDirectoryDownloader())

        container.singleton(
            MongoDatabaseFactory,
            lambda c: MongoDatabaseFactoryImpl(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
            ),
        )

        container.singleton(
            PluginSkillStoreFactory,
            lambda c: PluginSkillStoreFactory(
                mongo_database_factory=c.resolve(MongoDatabaseFactory),
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
            ),
        )

        container.singleton(
            PluginSkillStore,
            lambda c: c.resolve(PluginSkillStoreFactory).create(),
        )

        # Register the shared loader (MarketplaceDirectoryLoader) as a
        # singleton so CompositeSkillLoader and SkillSync share the same
        # instance (avoiding redundant downloads and inconsistent cache state).
        container.singleton(
            MarketplaceDirectoryLoader,
            lambda c: _build_shared_loader(c),
        )

        container.singleton(
            CompositeSkillLoader,
            lambda c: CompositeSkillLoader(
                shared_loader=c.resolve(MarketplaceDirectoryLoader),
                user_loader=c.resolve(PluginSkillStore),
            ),
        )

        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(CompositeSkillLoader),
        )

        container.singleton(
            SkillSync,
            lambda c: SkillSync(
                shared_loader=c.resolve(MarketplaceDirectoryLoader),
                user_store=c.resolve(PluginSkillStore),
            ),
        )

        return container
