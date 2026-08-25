from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.skill_sync import SkillSync
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


async def initialize_skills(
    *,
    user_store: PluginSkillStore,
    skill_sync: SkillSync,
) -> None:
    """Run skill framework initialization on server startup.

    This should be called once during the application lifespan.  It:
    1. Creates MongoDB indexes (idempotent).
    2. Checks whether the plugins collection already has data.
       - If empty: syncs shared skills from GitHub/filesystem into MongoDB.
       - If not empty: skips the sync (plugins were already seeded).

    Use :func:`reload_plugins` to force a re-download and re-sync at any time.
    """
    logger.info("Skills startup: ensuring indexes...")
    await user_store.ensure_indexes()

    plugins_exist = await user_store.has_plugins()
    if plugins_exist:
        logger.info(
            "Skills startup: plugins collection is not empty — skipping marketplace sync. "
            "Use the 'reload_plugins' command to force a refresh."
        )
        return

    logger.info("Skills startup: plugins collection is empty — syncing shared skills to MongoDB...")
    try:
        result = await skill_sync.sync()
        logger.info(
            "Skills startup: sync complete — "
            "plugins_synced=%d skills_synced=%d resources_synced=%d "
            "scripts_synced=%d errors=%d",
            result.plugins_synced,
            result.skills_added,
            result.resources_added,
            result.scripts_added,
            result.errors,
        )
    except Exception:
        logger.exception(
            "Skills startup: marketplace sync failed — the gateway will continue with previously synced skills"
        )


async def reload_plugins(
    *,
    skill_loader: SkillLoaderProtocol,
    skill_sync: SkillSync,
) -> str:
    """Force re-download plugins from GitHub and re-sync all plugin collections.

    This can be called manually (e.g. via the ``reload_plugins`` system command)
    to refresh the ``plugins``, ``plugin_skills``, ``plugin_references``, and
    ``plugin_scripts`` collections from the marketplace source.
    """
    logger.info("reload_plugins: forcing marketplace refresh from GitHub...")
    await skill_loader.refresh_async()

    logger.info("reload_plugins: syncing refreshed plugins to MongoDB...")
    result = await skill_sync.sync()
    summary = (
        f"plugins_synced={result.plugins_synced} "
        f"skills_synced={result.skills_added} "
        f"resources_synced={result.resources_added} "
        f"scripts_synced={result.scripts_added} "
        f"errors={result.errors}"
    )
    logger.info("reload_plugins: complete — %s", summary)
    return summary
