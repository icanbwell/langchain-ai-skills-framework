from langchain_ai_skills_framework.cache.skill_cache import SkillCache
from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderProtocol,
)
from simple_container.container.simple_container import SimpleContainer


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *, container: SimpleContainer
    ) -> SimpleContainer:

        container.singleton(
            LangchainAISkillsFrameworkEnvironmentVariables,
            lambda c: LangchainAISkillsFrameworkEnvironmentVariables(),
        )

        container.singleton(
            SkillCache,
            lambda c: SkillCache(
                environment_variables=c.resolve(
                    LangchainAISkillsFrameworkEnvironmentVariables
                ),
            ),
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
