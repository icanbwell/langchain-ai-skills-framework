from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderProtocol,
)
from simple_container.container.simple_container import SimpleContainer
from simple_container.environment.environment_variables import EnvironmentVariables

from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
)


class LangchainAISkillsFrameworkContainerFactory:
    @staticmethod
    def register_services_in_container(
        *, container: SimpleContainer
    ) -> SimpleContainer:

        container.singleton(
            SkillDirectoryLoader,
            lambda c: SkillDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
            ),
        )

        container.singleton(
            SkillkitDirectoryLoader,
            lambda c: SkillkitDirectoryLoader(
                environment_variables=c.resolve(EnvironmentVariables),  # type: ignore[arg-type]
            ),
        )

        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(SkillDirectoryLoader),
        )

        return container
