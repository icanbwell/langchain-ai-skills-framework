from langchain_ai_skills_framework.loaders.client_scoped_skill_loader import (
    ClientScopedSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderProtocol,
)
from simple_container.container.simple_container import SimpleContainer
from simple_container.environment.environment_variables import EnvironmentVariables


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
            ClientScopedSkillLoader,
            lambda c: ClientScopedSkillLoader(
                base_loader=c.resolve(SkillDirectoryLoader),
            ),
        )
        container.singleton(
            SkillLoaderProtocol,
            lambda c: c.resolve(ClientScopedSkillLoader),
        )

        return container
