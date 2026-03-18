import os

from langchain_ai_skills_framework.cache.skill_cache import SkillCache
from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderProtocol,
)

class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *, container: SimpleContainer
    ) -> SimpleContainer:

        container.singleton(
            SkillCache,
            lambda c: SkillCache(
                ttl_seconds=(
                    int(os.environ["SKILLS_CACHE_TIMEOUT_SECONDS"])
                    if os.environ.get("SKILLS_CACHE_TIMEOUT_SECONDS")
                    else 60 * 60
                )
            ),
        )
        container.singleton(
            SkillDirectoryLoader,
            lambda c: SkillDirectoryLoader(
                skills_directory=c.resolve(EnvironmentVariables).skills_directory,
                cache=c.resolve(SkillCache),
                environment_variables=c.resolve(EnvironmentVariables),
            ),
        )
        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(SkillDirectoryLoader),
        )
