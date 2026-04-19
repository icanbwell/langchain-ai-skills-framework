from __future__ import annotations

import logging
from typing import Sequence

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.user_skill_store import UserSkillStore
from langchain_ai_skills_framework.models.skills_model import SkillSummary
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])

SYSTEM_USER_ID = "system"


class SkillSync:
    """Syncs skills from a shared loader (GitHub/filesystem) into MongoDB.

    On server startup, this compares the skills available from the shared
    loader against what is already stored in MongoDB under the system user.
    Missing skills, resources, and scripts are inserted. Existing items
    are never overwritten — this is insert-only to preserve any
    user modifications.
    """

    def __init__(
        self,
        *,
        shared_loader: SkillLoaderProtocol,
        user_store: UserSkillStore,
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
            logger.info("SkillSync: no shared skills found; nothing to sync.")
            return result

        logger.info("SkillSync: checking %d shared skills against MongoDB.", len(summaries))

        for summary in summaries:
            skill_name = summary.name
            try:
                await self._sync_skill(skill_name=skill_name, result=result)
            except Exception:
                logger.exception("SkillSync: failed to sync skill '%s'; skipping.", skill_name)
                result.errors += 1

        logger.info(
            "SkillSync: complete. skills_added=%d resources_added=%d scripts_added=%d skills_skipped=%d errors=%d",
            result.skills_added,
            result.resources_added,
            result.scripts_added,
            result.skills_skipped,
            result.errors,
        )
        return result

    async def _sync_skill(self, *, skill_name: str, result: SyncResult) -> None:
        """Sync a single skill and its resources/scripts."""
        # Sync the skill content
        skill_exists = await self._store.skill_exists(user_id=SYSTEM_USER_ID, skill_name=skill_name)
        if skill_exists:
            result.skills_skipped += 1
        else:
            details = self._shared.get_skill_details(skill_name)
            await self._store.save_skill(
                user_id=SYSTEM_USER_ID,
                skill_name=skill_name,
                content=details.content,
                modified_by=SYSTEM_USER_ID,
            )
            # Mark seeded skills as shared so all users can see them
            await self._store.set_skill_shared(user_id=SYSTEM_USER_ID, skill_name=skill_name, shared=True)
            result.skills_added += 1
            logger.debug("SkillSync: added skill '%s'.", skill_name)

        # Sync resources
        try:
            resource_names = self._shared.list_skill_resource_names(skill_name)
        except Exception:
            logger.debug("SkillSync: could not list resources for skill '%s'.", skill_name)
            resource_names = []

        for resource_name in resource_names:
            try:
                exists = await self._store.resource_exists(
                    user_id=SYSTEM_USER_ID,
                    skill_name=skill_name,
                    resource_name=resource_name,
                )
                if exists:
                    continue
                content = self._shared.read_skill_resource(skill_name, resource_name)
                await self._store.save_resource(
                    user_id=SYSTEM_USER_ID,
                    skill_name=skill_name,
                    resource_name=resource_name,
                    content=content,
                    modified_by=SYSTEM_USER_ID,
                )
                result.resources_added += 1
                logger.debug(
                    "SkillSync: added resource '%s' for skill '%s'.",
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
            script_names = self._shared.list_skill_script_names(skill_name)
        except Exception:
            logger.debug("SkillSync: could not list scripts for skill '%s'.", skill_name)
            script_names = []

        for script_name in script_names:
            try:
                exists = await self._store.script_exists(
                    user_id=SYSTEM_USER_ID,
                    skill_name=skill_name,
                    script_name=script_name,
                )
                if exists:
                    continue
                # Read script content from the filesystem via the shared loader's
                # skill directory structure.
                details = self._shared.get_skill_details(skill_name)
                script_path = details.source_path.parent / f"{script_name}.py"
                if not script_path.is_file():
                    # Try without .py extension
                    script_path = details.source_path.parent / script_name
                if script_path.is_file():
                    content = script_path.read_text(encoding="utf-8")
                    await self._store.save_script(
                        user_id=SYSTEM_USER_ID,
                        skill_name=skill_name,
                        script_name=script_name,
                        content=content,
                        modified_by=SYSTEM_USER_ID,
                    )
                    result.scripts_added += 1
                    logger.debug(
                        "SkillSync: added script '%s' for skill '%s'.",
                        script_name,
                        skill_name,
                    )
                else:
                    logger.debug(
                        "SkillSync: script file for '%s' in skill '%s' not found on disk.",
                        script_name,
                        skill_name,
                    )
            except Exception:
                logger.exception(
                    "SkillSync: failed to sync script '%s' for skill '%s'.",
                    script_name,
                    skill_name,
                )
                result.errors += 1


class SyncResult:
    """Tracks what was added during a sync operation."""

    def __init__(self) -> None:
        self.skills_added: int = 0
        self.skills_skipped: int = 0
        self.resources_added: int = 0
        self.scripts_added: int = 0
        self.errors: int = 0
