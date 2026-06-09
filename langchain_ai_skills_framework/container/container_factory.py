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
from langchain_ai_skills_framework.github.token_provider import (
    GitHubAppTokenProvider,
    GitHubTokenProvider,
    StaticTokenProvider,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from simple_container.container.interfaces import IContainer
from simple_container.container.simple_container import SimpleContainer
from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.executors.script_executor_protocol import (
    ScriptExecutorProtocol,
)
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
    """Build the shared skill loader from the plugin marketplace."""
    env_vars = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))

    snapshot_cache_store: BaseStore | None = None
    try:
        snapshot_cache_store = c.resolve(BaseStore)
    except Exception:
        logger.debug("SnapshotCacheStore not available; proceeding without cache.")

    return MarketplaceDirectoryLoader(
        environment_variables=env_vars,
        github_directory_downloader=c.resolve(GithubDirectoryDownloader),
        script_executor=c.resolve(ScriptExecutorProtocol),
        snapshot_cache_store=snapshot_cache_store,
        token_provider=c.resolve(GitHubTokenProvider),
    )


def _build_token_provider(*, c: IContainer) -> GitHubTokenProvider | None:
    """Build GitHub token provider: App credentials > static PAT > None."""
    env = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))

    app_id = env.github_app_id
    private_key = env.github_app_private_key
    installation_id = env.github_app_installation_id

    if app_id and private_key and installation_id:
        logger.info("GitHub authentication: using GitHub App (app_id=%s)", app_id)
        return GitHubAppTokenProvider(
            app_id=app_id,
            private_key=private_key,
            installation_id=installation_id,
        )

    token = env.skills_github_token
    if token:
        logger.info("GitHub authentication: using static token (PAT)")
        return StaticTokenProvider(token=token)

    logger.info("GitHub authentication: not configured (marketplace features disabled)")
    return None


def _build_marketplace_publisher(*, c: IContainer) -> GitHubMarketplacePublisher | None:
    env = cast(SkillLoaderEnvironmentVariables, c.resolve(EnvironmentVariables))
    if not env.plugins_marketplace_publish_enabled:
        return None
    marketplace_uri = env.plugins_marketplace
    if not marketplace_uri or not marketplace_uri.startswith("github://"):
        return None

    token_provider: GitHubTokenProvider | None = c.resolve(GitHubTokenProvider)
    if token_provider is None:
        return None

    git_location = GithubDirectoryDownloader.parse_github_uri(marketplace_uri)
    repo = f"{git_location.owner}/{git_location.repository}"

    return GitHubMarketplacePublisher(
        token_provider=token_provider,
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

        container.singleton(
            GitHubTokenProvider,
            lambda c: _build_token_provider(c=c),
        )

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
                script_executor=c.resolve(ScriptExecutorProtocol),
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
