from __future__ import annotations

from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from langchain_ai_skills_framework.langchain.tools.delete_skill_tool import DeleteSkillTool
from langchain_ai_skills_framework.langchain.tools.list_plugins_tool import ListPluginsTool
from langchain_ai_skills_framework.langchain.tools.list_skills_tool import ListSkillsTool
from langchain_ai_skills_framework.langchain.tools.load_skill_tool import LoadSkillTool
from langchain_ai_skills_framework.langchain.tools.publish_skill_tool import PublishSkillTool
from langchain_ai_skills_framework.langchain.tools.read_skill_resource_tool import (
    ReadSkillResourceTool,
)
from langchain_ai_skills_framework.langchain.tools.run_skill_script_tool import (
    RunSkillScriptTool,
)
from langchain_ai_skills_framework.langchain.tools.save_skill_resource_tool import (
    SaveSkillResourceTool,
)
from langchain_ai_skills_framework.langchain.tools.save_skill_script_tool import (
    SaveSkillScriptTool,
)
from langchain_ai_skills_framework.langchain.tools.save_skill_tool import SaveSkillTool


def build_skill_tools(
    *,
    skill_loader: SkillLoaderProtocol,
    user_skill_store: PluginSkillStore,
    marketplace_publisher: GitHubMarketplacePublisher | None = None,
) -> list[BaseTool]:
    """Build the full set of LangChain skill tools.

    This factory keeps all ``langchain`` imports out of the core loaders,
    so the ``langchain`` optional dependency is only required when the
    caller actually constructs tools.
    """
    return [
        ListPluginsTool(mongo_skill_loader=user_skill_store),
        ListSkillsTool(skill_loader=skill_loader),
        LoadSkillTool(skill_loader=skill_loader, user_skill_store=user_skill_store),
        ReadSkillResourceTool(skill_loader=skill_loader),
        RunSkillScriptTool(skill_loader=skill_loader),
        SaveSkillTool(mongo_skill_loader=user_skill_store),
        SaveSkillResourceTool(mongo_skill_loader=user_skill_store),
        SaveSkillScriptTool(mongo_skill_loader=user_skill_store),
        DeleteSkillTool(mongo_skill_loader=user_skill_store),
        PublishSkillTool(
            mongo_skill_loader=user_skill_store,
            marketplace_publisher=marketplace_publisher,
        ),
    ]
