import logging
from typing import cast

from key_value.aio.stores.base import BaseStore
from key_value.aio.stores.mongodb import MongoDBStore
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
from langchain_ai_skills_framework.persistence.mongo_url_helpers import (
    MongoUrlHelpers,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from simple_container.container.interfaces import IContainer
from simple_container.container.simple_container import SimpleContainer
from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)

logger = logging.getLogger(__name__)


def _build_key_value_store(*, c: IContainer) -> BaseStore:
    from langchain_ai_skills_framework.environment.environment_variables import (
        LangchainAISkillsFrameworkEnvironmentVariables,
    )

    env = cast(
        LangchainAISkillsFrameworkEnvironmentVariables,
        c.resolve(EnvironmentVariables),
    )

    store_type = env.key_value_store_type

    if store_type == "redis":
        from key_value.aio.stores.redis import RedisStore

        return RedisStore(url=env.redis_url)

    url = MongoUrlHelpers.add_credentials_to_mongo_url(
        mongo_url=env.mongo_skills_uri,
        username=env.mongo_skills_db_username,
        password=env.mongo_skills_db_password,
    )
    return MongoDBStore(url=url, db_name=env.mongo_skills_db_name)


def _build_shared_loader(*, c: IContainer) -> SkillLoaderProtocol:
    """Build the shared skill loader from the plugin marketplace.

    Skills are loaded from the marketplace structure (plugins/*/skills/).
    The SkillkitDirectoryLoader (SKILLS_DIRECTORY) has been removed —
    all skills come from plugins.
    """
    env_vars = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))

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


def _build_marketplace_publisher(*, c: IContainer) -> GitHubMarketplacePublisher | None:
    """Build the marketplace publisher from the PLUGINS_MARKETPLACE URI.

    Publishing is enabled when PLUGINS_MARKETPLACE is a github:// URI
    and a GitHub token is available.  The target repo is extracted from
    the URI; PLUGINS_MARKETPLACE_PUBLISH_BRANCH and
    PLUGINS_MARKETPLACE_PUBLISH_USE_BRANCH control commit behaviour.
    """
    env = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))
    if not env.plugins_marketplace_publish_enabled:
        return None
    marketplace_uri = env.plugins_marketplace
    token = env.skills_github_token
    if not marketplace_uri or not token or not marketplace_uri.startswith("github://"):
        return None

    git_location = GithubDirectoryDownloader.parse_github_uri(marketplace_uri)
    repo = f"{git_location.owner}/{git_location.repository}"

    return GitHubMarketplacePublisher(
        access_token=token,
        repo=repo,
        base_branch=env.plugins_marketplace_publish_branch,
        use_branch=env.plugins_marketplace_publish_use_branch,
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
            BaseStore,
            lambda c: _build_key_value_store(c=c),
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
            lambda c: _build_shared_loader(c=c),
        )

        container.singleton(
            CompositeSkillLoader,
            lambda c: CompositeSkillLoader(
                shared_loader=c.resolve(MarketplaceDirectoryLoader),
                user_loader=c.resolve(PluginSkillStore),
                marketplace_publisher=_build_marketplace_publisher(c=c),
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
