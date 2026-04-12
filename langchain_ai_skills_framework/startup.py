from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.skill_sync import SkillSync
from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


async def initialize_skills(
    *,
    user_store: UserSkillStore,
    skill_sync: SkillSync,
) -> None:
    """Run skill framework initialization on server startup.

    This should be called once during the application lifespan.  It:
    1. Creates MongoDB indexes (idempotent).
    2. Syncs shared skills from GitHub/filesystem into MongoDB,
       inserting any that are missing without overwriting existing ones.
    """
    logger.info("Skills startup: ensuring indexes...")
    await user_store.ensure_indexes()

    logger.info("Skills startup: syncing shared skills to MongoDB...")
    result = await skill_sync.sync()
    logger.info(
        "Skills startup: sync complete — "
        "skills_added=%d resources_added=%d scripts_added=%d "
        "skills_skipped=%d errors=%d",
        result.skills_added,
        result.resources_added,
        result.scripts_added,
        result.skills_skipped,
        result.errors,
    )
