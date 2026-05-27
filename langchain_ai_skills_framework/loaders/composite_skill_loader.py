from __future__ import annotations

import asyncio
import logging
from html import escape
from types import MappingProxyType
from typing import Any, Sequence

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import (
    MyScriptExecutor,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class CompositeSkillLoader(SkillLoaderProtocol):
    """Merges a shared/global skill loader with a per-user MongoDB loader.

    The *shared_loader* handles filesystem / GitHub skills (with scripts
    and resources).  The *user_loader* handles MongoDB-persisted skills
    including their resources and scripts.

    This class is a **singleton**.  Per-user context (``user_id``) is
    provided on each call that needs it, not at construction time.
    """

    def __init__(
        self,
        *,
        shared_loader: SkillLoaderProtocol,
        user_loader: PluginSkillStore,
        marketplace_publisher: GitHubMarketplacePublisher | None = None,
    ) -> None:
        if shared_loader is None:
            raise ValueError("shared_loader must not be None")
        if user_loader is None:
            raise ValueError("user_loader must not be None")

        self._shared_loader = shared_loader
        self._user_loader = user_loader
        self._marketplace_publisher = marketplace_publisher

    @property
    def shared_loader(self) -> SkillLoaderProtocol:
        return self._shared_loader

    @property
    def user_loader(self) -> PluginSkillStore:
        return self._user_loader

    # --- SkillLoaderProtocol implementation ----------------------------------

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        """Return shared skill summaries (user skills require async — see ``list_all_summaries``)."""
        return self._shared_loader.list_skill_summaries(allowed_skills)

    async def list_all_summaries(self, *, user_id: str, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        """Return merged summaries from shared + user skills."""
        snapshot = await self._merged_snapshot(user_id=user_id)
        return snapshot.ordered_summaries

    def get_skill_details(self, skill_name: str, *, plugin_name: str | None = None) -> SkillDetails:
        """Get skill details from shared loader only (sync)."""
        return self._shared_loader.get_skill_details(skill_name, plugin_name=plugin_name)

    async def get_skill_details_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> SkillDetails:
        """Get skill details checking user skills first, then shared DB, then GitHub."""
        normalized = normalize_skill_name(skill_name)
        # 1. User's own skills (highest precedence)
        try:
            return await self._user_loader.get_skill_details(
                user_id=user_id, plugin_name=plugin_name, skill_name=normalized
            )
        except SkillNotFoundError:
            logger.debug(
                "Skill '%s' not found in user skills for user '%s', trying shared skills",
                normalized,
                user_id,
            )

        # 2. Shared DB skills from other users
        shared_snapshot = await self._user_loader.load_shared_snapshot(plugin_name=plugin_name)
        if normalized in shared_snapshot.details_by_name:
            return shared_snapshot.details_by_name[normalized]

        # 3. GitHub/filesystem skills (lowest precedence)
        return self._shared_loader.get_skill_details(normalized, plugin_name=plugin_name)

    def refresh(self) -> None:
        self._shared_loader.refresh()

    async def refresh_async(self) -> None:
        await self._shared_loader.refresh_async()

    async def get_instructions(self) -> str:
        """Return shared skill instructions (no user context available here)."""
        return await self._shared_loader.get_instructions()

    async def get_instructions_for_user(self, *, user_id: str) -> str:
        """Return merged skill instructions including user skills."""
        summaries = await self.list_all_summaries(user_id=user_id, allowed_skills=set())
        if not summaries:
            return await self._shared_loader.get_instructions()

        # Batch fetch all usage counts in one aggregation query
        skill_names = [s.name for s in summaries]
        usage_counts = await self._user_loader.get_skill_usage_counts(skill_names=skill_names)

        lines: list[str] = []
        for skill in summaries:
            escaped_name = escape(skill.name, quote=True)
            escaped_description = escape(skill.description.strip(), quote=True)
            lines.append("<skill>")
            if skill.plugin_name:
                escaped_plugin_name = escape(skill.plugin_name, quote=True)
                lines.append(f"<plugin_name>{escaped_plugin_name}</plugin_name>")
            lines.append(f"<name>{escaped_name}</name>")
            lines.append(f"<description>{escaped_description}</description>")
            lines.append(f"<usage_count>{usage_counts.get(skill.name, 0)}</usage_count>")
            author = skill.metadata.get("user_id") if skill.metadata else None
            if author:
                escaped_author = escape(str(author), quote=True)
                lines.append(f"<author>{escaped_author}</author>")
            lines.append("</skill>")
        skills_list = "\n".join(lines)

        return (
            "You have three access to a collection of skills containing domain-specific "
            "knowledge and capabilities.\n"
            "Each skill provides specialized instructions for specific tasks.\n\n"
            f"<available_skills>\n{skills_list}\n</available_skills>\n\n"
            "When a task falls within a skill's domain:\n"
            "1. Use `list_plugins` to see all registered plugins\n"
            "2. Use `load_skill` with plugin_name to read the complete skill instructions\n"
            "3. Follow the skill's guidance to complete the task\n"
            "4. Use `read_skill_resource` with plugin_name to read files referenced by the skill\n"
            "5. Use `run_skill_script` with plugin_name to run scripts provided by the skill\n"
            "6. Use `save_skill` with plugin_name to save a new or updated skill for the current user\n"
            "7. Use `save_skill_resource` with plugin_name to save a resource file for a skill\n"
            "8. Use `save_skill_script` with plugin_name to save a script file for a skill\n"
            "9. Use `delete_skill` with plugin_name to remove a previously saved skill\n"
            "10. Use `publish_skill` with plugin_name to publish a skill to the marketplace or unpublish it\n\n"
            "All skill tools require a `plugin_name` parameter to scope the operation to a specific plugin.\n"
            "Use progressive disclosure: load only what you need, when you need it."
        )

    def read_skill_resource(self, skill_name: str, resource_name: str, *, plugin_name: str | None = None) -> str:
        return self._shared_loader.read_skill_resource(skill_name, resource_name, plugin_name=plugin_name)

    async def read_skill_resource_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str, resource_name: str
    ) -> str:
        """Read a resource, checking user's MongoDB skills first, then shared loader."""
        normalized = normalize_skill_name(skill_name)

        # Check user's own skills first
        try:
            return await self._user_loader.read_resource(
                user_id=user_id, plugin_name=plugin_name, skill_name=normalized, resource_name=resource_name
            )
        except SkillNotFoundError:
            logger.debug(
                "Resource '%s' not found in user skill '%s' for user '%s', trying shared skills",
                resource_name,
                normalized,
                user_id,
            )

        # Check shared DB skills
        shared_snapshot = await self._user_loader.load_shared_snapshot(plugin_name=plugin_name)
        if normalized in shared_snapshot.details_by_name:
            shared_detail = shared_snapshot.details_by_name[normalized]
            owner_user_id = str(
                shared_detail.summary.metadata.get("user_id", "") if shared_detail.summary.metadata else ""
            )
            if owner_user_id:
                try:
                    return await self._user_loader.read_resource(
                        user_id=owner_user_id,
                        plugin_name=plugin_name,
                        skill_name=normalized,
                        resource_name=resource_name,
                    )
                except SkillNotFoundError:
                    pass

        # Fall back to shared filesystem loader
        return self._shared_loader.read_skill_resource(normalized, resource_name, plugin_name=plugin_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None, *, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        return await self._shared_loader.run_skill_script(skill_name, script_name, arguments, plugin_name=plugin_name)

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        """Run a script, checking user's MongoDB skills first, then shared loader.

        MongoDB-stored scripts are executed in a subprocess with the script
        content written to a temporary file.
        """
        normalized = normalize_skill_name(skill_name)

        # Check user's own scripts first
        try:
            script_content = await self._user_loader.read_script(
                user_id=user_id, plugin_name=plugin_name, skill_name=normalized, script_name=script_name
            )
            return await self._execute_script_content(
                script_content=script_content,
                script_name=script_name,
                arguments=arguments,
            )
        except SkillNotFoundError:
            pass

        # Check shared DB skills
        shared_snapshot = await self._user_loader.load_shared_snapshot(plugin_name=plugin_name)
        if normalized in shared_snapshot.details_by_name:
            shared_detail = shared_snapshot.details_by_name[normalized]
            owner_user_id = str(
                shared_detail.summary.metadata.get("user_id", "") if shared_detail.summary.metadata else ""
            )
            if owner_user_id:
                try:
                    script_content = await self._user_loader.read_script(
                        user_id=owner_user_id,
                        plugin_name=plugin_name,
                        skill_name=normalized,
                        script_name=script_name,
                    )
                    return await self._execute_script_content(
                        script_content=script_content,
                        script_name=script_name,
                        arguments=arguments,
                    )
                except SkillNotFoundError:
                    pass

        # Fall back to shared filesystem loader
        return await self._shared_loader.run_skill_script(normalized, script_name, arguments, plugin_name=plugin_name)

    def list_skill_script_names(self, skill_name: str, *, plugin_name: str | None = None) -> Sequence[str]:
        return self._shared_loader.list_skill_script_names(skill_name, plugin_name=plugin_name)

    async def list_skill_script_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        """List scripts, merging user MongoDB scripts with shared loader scripts."""
        normalized = normalize_skill_name(skill_name)
        names: set[str] = set()

        # User's own scripts
        try:
            user_scripts = await self._user_loader.list_script_names(
                user_id=user_id, plugin_name=plugin_name, skill_name=normalized
            )
            names.update(user_scripts)
        except (SkillNotFoundError, ValueError):
            pass

        # Shared loader scripts
        try:
            shared_scripts = self._shared_loader.list_skill_script_names(normalized, plugin_name=plugin_name)
            names.update(shared_scripts)
        except SkillNotFoundError:
            pass

        return sorted(names)

    def list_skill_resource_names(self, skill_name: str, *, plugin_name: str | None = None) -> Sequence[str]:
        return self._shared_loader.list_skill_resource_names(skill_name, plugin_name=plugin_name)

    async def list_skill_resource_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        """List resources, merging user MongoDB resources with shared loader resources."""
        normalized = normalize_skill_name(skill_name)
        names: set[str] = set()

        # User's own resources
        try:
            user_resources = await self._user_loader.list_resource_names(
                user_id=user_id, plugin_name=plugin_name, skill_name=normalized
            )
            names.update(user_resources)
        except (SkillNotFoundError, ValueError):
            pass

        # Shared loader resources
        try:
            shared_resources = self._shared_loader.list_skill_resource_names(normalized, plugin_name=plugin_name)
            names.update(shared_resources)
        except SkillNotFoundError:
            pass

        return sorted(names)

    async def get_plugin_mcp_configs(self) -> Sequence[PluginMcpServerEntry]:
        shared_configs = await self._shared_loader.get_plugin_mcp_configs()
        if shared_configs:
            return shared_configs
        plugins = await self.list_plugin_definitions()
        entries: list[PluginMcpServerEntry] = []
        for plugin in plugins:
            entries.extend(plugin.mcp_servers)
        return entries

    async def list_plugin_definitions(self) -> Sequence[PluginDefinition]:
        shared_defs = await self._shared_loader.list_plugin_definitions()
        if shared_defs:
            return shared_defs
        try:
            return await self._list_plugin_definitions_from_mongo()
        except Exception:
            logger.exception("list_plugin_definitions: _list_plugin_definitions_from_mongo failed")
            return []

    async def _list_plugin_definitions_from_mongo(self) -> Sequence[PluginDefinition]:
        """Reconstruct PluginDefinition objects from MongoDB plugin documents."""
        from pathlib import Path

        docs = await self._user_loader.list_plugins()
        logger.info(
            "_list_plugin_definitions_from_mongo: list_plugins returned %d doc(s): %s",
            len(docs),
            [d.plugin_name for d in docs],
        )
        definitions: list[PluginDefinition] = []
        for doc in docs:
            try:
                mcp_entries: list[PluginMcpServerEntry] = []
                for mcp_dict in doc.mcp_servers:
                    mcp_entries.append(
                        PluginMcpServerEntry(
                            server_key=str(mcp_dict.get("server_key", "")),
                            plugin_name=str(mcp_dict.get("plugin_name", doc.plugin_name)),
                            plugin_root=Path("."),
                            url=mcp_dict.get("url"),
                            command=mcp_dict.get("command"),
                            args=tuple(mcp_dict.get("args", ())),
                            env=dict(mcp_dict.get("env", {})),
                            headers=dict(mcp_dict.get("headers", {})),
                            description=mcp_dict.get("description"),
                            display_name=mcp_dict.get("display_name"),
                            auth=mcp_dict.get("auth"),
                            oauth=mcp_dict.get("oauth"),
                        )
                    )
                skill_summaries = tuple(
                    SkillSummary(
                        name=s,
                        description="",
                        plugin_name=doc.plugin_name,
                        source_path=Path(f"mongodb://system/{doc.plugin_name}/{s}"),
                    )
                    for s in doc.skills
                )
                definitions.append(
                    PluginDefinition(
                        name=doc.plugin_name,
                        description=doc.description,
                        skills=skill_summaries,
                        mcp_servers=tuple(mcp_entries),
                    )
                )
            except Exception:
                logger.exception(
                    "_list_plugin_definitions_from_mongo: failed to build PluginDefinition "
                    "for plugin '%s' (mcp_servers=%d, skills=%d)",
                    doc.plugin_name,
                    len(doc.mcp_servers),
                    len(doc.skills),
                )
        return definitions

    # --- Merging -------------------------------------------------------------

    async def _merged_snapshot(self, *, user_id: str) -> SkillSnapshot:
        """Build a merged snapshot with precedence: GitHub -> shared DB -> user DB.

        GitHub/filesystem skills form the base.  Shared database skills
        overlay next (available to all users).  The requesting user's own
        skills win on name collision.
        """
        details: dict[str, SkillDetails] = {}

        # 1. GitHub / filesystem skills (lowest precedence, sync + cached)
        for summary in self._shared_loader.list_skill_summaries(allowed_skills=set()):
            detail = self._shared_loader.get_skill_details(summary.name, plugin_name=summary.plugin_name)
            details[summary.name] = detail

        # 2+3. Load shared and user snapshots concurrently
        shared_snapshot, user_snapshot = await asyncio.gather(
            self._user_loader.load_shared_snapshot(),
            self._user_loader.load_snapshot(user_id=user_id),
        )

        # Shared database skills (override GitHub on collision)
        for name, detail in shared_snapshot.details_by_name.items():
            details[name] = detail

        # User's own database skills (highest precedence)
        for name, detail in user_snapshot.details_by_name.items():
            details[name] = detail

        ordered = tuple(sorted(details.values(), key=lambda d: d.name))
        return SkillSnapshot(
            details_by_name=MappingProxyType(details),
            ordered_summaries=tuple(d.summary for d in ordered),
        )

    # --- Script execution ----------------------------------------------------

    @staticmethod
    async def _execute_script_content(
        *,
        script_content: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        """Execute a script stored as content in MongoDB."""
        executor = MyScriptExecutor()
        return await executor.execute_inline_script(
            script_name=script_name,
            script=script_content,
            arguments=arguments or {},
        )
