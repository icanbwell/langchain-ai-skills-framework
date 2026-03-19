import os
import logging

from langchain_ai_skills_framework.cache.skill_cache import SkillCache
from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderProtocol,
)
from simple_container.container.simple_container import SimpleContainer


_LOGGER: logging.Logger = logging.getLogger(__name__)
_SKILLS_CACHE_TIMEOUT_ENV_VAR: str = "SKILLS_CACHE_TIMEOUT_SECONDS"
_DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS: int = 60 * 60


def _get_skills_cache_timeout_seconds() -> int:
    """Return a validated TTL in seconds for SkillCache based on environment."""
    raw_value = os.getenv(_SKILLS_CACHE_TIMEOUT_ENV_VAR)
    if raw_value is None:
        return _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS
    try:
        ttl_seconds = int(raw_value)
    except ValueError:
        _LOGGER.warning(
            "Invalid %s value %r; using default %d seconds",
            _SKILLS_CACHE_TIMEOUT_ENV_VAR,
            raw_value,
            _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS,
        )
        return _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS
    if ttl_seconds <= 0:
        _LOGGER.warning(
            "%s must be a positive integer; got %d. Using default %d seconds",
            _SKILLS_CACHE_TIMEOUT_ENV_VAR,
            ttl_seconds,
            _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS,
        )
        return _DEFAULT_SKILLS_CACHE_TIMEOUT_SECONDS
    return ttl_seconds


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *, container: SimpleContainer
    ) -> SimpleContainer:

        container.singleton(
            SkillCache,
            lambda c: SkillCache(
                ttl_seconds=_get_skills_cache_timeout_seconds(),
            ),
        )

        container.singleton(
            LangchainAISkillsFrameworkEnvironmentVariables,
            lambda c: LangchainAISkillsFrameworkEnvironmentVariables(),
        )

        container.singleton(
            SkillDirectoryLoader,
            lambda c: SkillDirectoryLoader(
                cache=c.resolve(SkillCache),
                environment_variables=c.resolve(
                    LangchainAISkillsFrameworkEnvironmentVariables
                ),
            ),
        )
        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(SkillDirectoryLoader),
        )

        return container
