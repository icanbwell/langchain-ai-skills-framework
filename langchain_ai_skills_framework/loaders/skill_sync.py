from __future__ import annotations

import logging
from typing import Sequence

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.skills_model import SkillSummary
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])

SYSTEM_USER_ID = "system"


class SkillSync:
    """Syncs skills from a shared loader (GitHub/filesystem) into MongoDB.

    On each sync, skills, resources, and scripts from the shared loader
    are upserted into MongoDB under the system user — existing items are
    replaced with the latest marketplace content.
    """

    def __init__(
        self,
        *,
        shared_loader: SkillLoaderProtocol,
        user_store: PluginSkillStore,
    ) -> None:
        self._shared = shared_loader
        self._store = user_store

    async def sync(self) -> SyncResult:
        """Compare shared skills against MongoDB and insert any missing items.

        Returns a ``SyncResult`` summarizing what was added.
        """
        result = SyncResult()

        summaries: Sequence[SkillSummary] = self._shared.list_skill_summaries(allowed_skills=set())
        if not summaries:
            logger.info("SkillSync: no shared skills found; skipping skill sync.")
        else:
            logger.info("SkillSync: checking %d shared skills against MongoDB.", len(summaries))

            for summary in summaries:
                skill_name = summary.name
                plugin_name = summary.plugin_name
                if not plugin_name:
                    logger.warning("SkillSync: skill '%s' has no plugin_name; skipping.", skill_name)
                    continue
                try:
                    await self._sync_skill(skill_name=skill_name, plugin_name=plugin_name, result=result)
                except Exception:
                    logger.exception("SkillSync: failed to sync skill '%s'; skipping.", skill_name)
                    result.errors += 1

        # Sync plugin definitions regardless of whether skills were found —
        # plugins carry MCP server configs, descriptions, and other metadata.
        await self._sync_plugins(result=result)

        logger.info(
            "SkillSync: complete. plugins_synced=%d skills_synced=%d resources_synced=%d scripts_synced=%d errors=%d",
            result.plugins_synced,
            result.skills_added,
            result.resources_added,
            result.scripts_added,
            result.errors,
        )
        return result

    async def _sync_skill(self, *, skill_name: str, plugin_name: str, result: SyncResult) -> None:
        """Sync a single skill and its resources/scripts.

        Always upserts — replaces existing content with the latest from
        the marketplace so that updates to shared skills propagate on restart.
        """
        details = self._shared.get_skill_details(skill_name, plugin_name=plugin_name)
        await self._store.save_skill(
            user_id=SYSTEM_USER_ID,
            plugin_name=plugin_name,
            skill_name=skill_name,
            content=details.content,
            modified_by=SYSTEM_USER_ID,
        )
        await self._store.set_skill_published(
            user_id=SYSTEM_USER_ID, plugin_name=plugin_name, skill_name=skill_name, published=True
        )
        result.skills_added += 1
        logger.debug("SkillSync: upserted skill '%s' from plugin '%s'.", skill_name, plugin_name)

        # Sync resources
        try:
            resource_names = self._shared.list_skill_resource_names(skill_name, plugin_name=plugin_name)
        except Exception:
            logger.debug("SkillSync: could not list resources for skill '%s'.", skill_name)
            resource_names = []

        for resource_name in resource_names:
            try:
                content = self._shared.read_skill_resource(skill_name, resource_name, plugin_name=plugin_name)
                await self._store.save_resource(
                    user_id=SYSTEM_USER_ID,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                    resource_name=resource_name,
                    content=content,
                    modified_by=SYSTEM_USER_ID,
                )
                result.resources_added += 1
                logger.debug(
                    "SkillSync: upserted resource '%s' for skill '%s'.",
                    resource_name,
                    skill_name,
                )
            except Exception:
                logger.exception(
                    "SkillSync: failed to sync resource '%s' for skill '%s'.",
                    resource_name,
                    skill_name,
                )
                result.errors += 1

        # Sync scripts
        try:
            script_names = self._shared.list_skill_script_names(skill_name, plugin_name=plugin_name)
        except Exception:
            logger.debug("SkillSync: could not list scripts for skill '%s'.", skill_name)
            script_names = []

        for script_name in script_names:
            try:
                details = self._shared.get_skill_details(skill_name, plugin_name=plugin_name)
                if details.source_path:
                    skill_dir = details.source_path.parent
                    script_path = skill_dir / "scripts" / f"{script_name}.py"
                    if not script_path.is_file():
                        script_path = skill_dir / "scripts" / f"{script_name}.sh"
                    if not script_path.is_file():
                        # Fallback: check skill directory root (legacy layout)
                        script_path = skill_dir / f"{script_name}.py"
                    if not script_path.is_file():
                        script_path = skill_dir / script_name
                    if script_path.is_file():
                        content = script_path.read_text(encoding="utf-8")
                        await self._store.save_script(
                            user_id=SYSTEM_USER_ID,
                            plugin_name=plugin_name,
                            skill_name=skill_name,
                            script_name=script_name,
                            content=content,
                            modified_by=SYSTEM_USER_ID,
                        )
                        result.scripts_added += 1
                        logger.debug(
                            "SkillSync: upserted script '%s' for skill '%s'.",
                            script_name,
                            skill_name,
                        )
                    else:
                        logger.debug(
                            "SkillSync: script file for '%s' in skill '%s' not found on disk.",
                            script_name,
                            skill_name,
                        )
                else:
                    logger.debug(
                        "SkillSync: no source path for skill '%s'; cannot sync script '%s'.",
                        skill_name,
                        script_name,
                    )
            except Exception:
                logger.exception(
                    "SkillSync: failed to sync script '%s' for skill '%s'.",
                    script_name,
                    skill_name,
                )
                result.errors += 1

    async def _sync_plugins(self, *, result: SyncResult) -> None:
        """Write plugin definitions (name, description, skills, MCP config) to MongoDB."""
        try:
            plugin_defs: Sequence[PluginDefinition] = await self._shared.list_plugin_definitions()
        except Exception:
            logger.exception("SkillSync: could not list plugin definitions; skipping plugin sync.")
            return

        logger.info("SkillSync: found %d plugin definitions to sync.", len(plugin_defs))
        if not plugin_defs:
            return

        for plugin in plugin_defs:
            try:
                mcp_server_dicts: list[dict[str, object]] = []
                for mcp in plugin.mcp_servers:
                    mcp_dict: dict[str, object] = {
                        "server_key": mcp.server_key,
                        "plugin_name": mcp.plugin_name,
                    }
                    if mcp.url:
                        mcp_dict["url"] = mcp.url
                    if mcp.command:
                        mcp_dict["command"] = mcp.command
                    if mcp.args:
                        mcp_dict["args"] = list(mcp.args)
                    if mcp.env:
                        mcp_dict["env"] = dict(mcp.env)
                    if mcp.headers:
                        mcp_dict["headers"] = dict(mcp.headers)
                    if mcp.description:
                        mcp_dict["description"] = mcp.description
                    if mcp.display_name:
                        mcp_dict["display_name"] = mcp.display_name
                    if mcp.auth:
                        mcp_dict["auth"] = mcp.auth
                    if mcp.oauth:
                        mcp_dict["oauth"] = mcp.oauth
                    mcp_server_dicts.append(mcp_dict)

                await self._store.save_plugin(
                    plugin_name=plugin.name,
                    description=plugin.description or "",
                    skills=[s.name for s in plugin.skills],
                    mcp_servers=mcp_server_dicts,
                )
                result.plugins_synced += 1
                logger.info(
                    "SkillSync: synced plugin '%s' (skills=%d, mcp_servers=%d).",
                    plugin.name,
                    len(plugin.skills),
                    len(mcp_server_dicts),
                )
            except Exception:
                logger.exception("SkillSync: failed to sync plugin '%s'.", plugin.name)
                result.errors += 1


class SyncResult:
    """Tracks what was added during a sync operation."""

    def __init__(self) -> None:
        self.plugins_synced: int = 0
        self.skills_added: int = 0
        self.resources_added: int = 0
        self.scripts_added: int = 0
        self.errors: int = 0
