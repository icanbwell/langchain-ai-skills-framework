from __future__ import annotations

import asyncio
import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class PublishSkillService:
    """Toggle a skill between published and unpublished in the marketplace.

    When a ``marketplace_publisher`` is configured, publishing a skill also
    fires off a background task that publishes (or unpublishes) the skill
    to the GitHub marketplace repo.  The background task never blocks the
    local operation.

    A class-level task registry ensures that rapid toggles for the same
    skill are serialized: a new publish/unpublish request cancels any
    in-flight task for that skill before scheduling the replacement.
    """

    _pending_tasks: dict[str, asyncio.Task[None]] = {}

    def __init__(
        self,
        *,
        mongo_skill_loader: PluginSkillStore | None,
        marketplace_publisher: GitHubMarketplacePublisher | None = None,
    ) -> None:
        self._store = mongo_skill_loader
        self._publisher = marketplace_publisher

    async def execute(self, *, user_id: str, plugin_name: str, skill_name: str, shared: bool) -> str:
        if not user_id:
            raise SkillOperationError("user_id is required for publish_skill")
        if not skill_name or not skill_name.strip():
            raise SkillOperationError("skill_name must be a non-empty string.")
        if self._store is None:
            raise SkillOperationError("mongo_skill_loader is not configured.")

        try:
            doc = await self._store.set_skill_shared(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                shared=shared,
            )
            state = "published" if doc.shared else "unpublished"
            message = f"Skill '{doc.skill_name}' is now {state}."
            logger.info("PublishSkillService: %s (user=%s)", message, user_id)

            if self._publisher is not None:
                task_key = f"{doc.plugin_name}/{doc.skill_name}"
                previous = PublishSkillService._pending_tasks.get(task_key)
                if previous is not None and not previous.done():
                    previous.cancel()
                    logger.debug("Cancelled in-flight publish task for '%s'", task_key)

                task = asyncio.create_task(
                    self._try_publish(
                        user_id=user_id,
                        plugin_name=doc.plugin_name,
                        skill_name=doc.skill_name,
                        shared=shared,
                    ),
                    name=f"marketplace-{'publish' if shared else 'unpublish'}-{task_key}",
                )
                PublishSkillService._pending_tasks[task_key] = task

                def _cleanup(_t: asyncio.Task[None], _key: str = task_key) -> None:
                    PublishSkillService._pending_tasks.pop(_key, None)

                task.add_done_callback(_cleanup)

            return message
        except Exception as exc:
            logger.exception(
                "PublishSkillService failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise SkillOperationError(
                f"Unable to update publishing for skill '{skill_name}' due to an internal error."
            ) from exc

    # ------------------------------------------------------------------
    # Background publish logic
    # ------------------------------------------------------------------

    async def _try_publish(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        shared: bool,
    ) -> None:
        """Best-effort publish/unpublish. Exceptions are logged, never raised."""
        assert self._publisher is not None
        assert self._store is not None
        try:
            if shared:
                await self._publish_skill(
                    user_id=user_id,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                )
            else:
                result = await self._publisher.unpublish_skill(
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                    user_id=user_id,
                )
                logger.info(
                    "Skill '%s/%s' removal result: %s",
                    plugin_name,
                    skill_name,
                    result,
                )
        except Exception:
            logger.exception(
                "Failed to %s skill '%s/%s' in marketplace",
                "publish" if shared else "unpublish",
                plugin_name,
                skill_name,
            )

    async def _publish_skill(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> None:
        assert self._publisher is not None
        assert self._store is not None

        details = await self._store.get_skill_details(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )

        resource_names = await self._store.list_resource_names(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        resources: dict[str, str] = {}
        for name in resource_names:
            resources[name] = await self._store.read_resource(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=name,
            )

        script_names = await self._store.list_script_names(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        scripts: dict[str, str] = {}
        for name in script_names:
            scripts[name] = await self._store.read_script(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=name,
            )

        result = await self._publisher.publish_skill(
            plugin_name=plugin_name,
            skill_name=skill_name,
            skill_content=details.content,
            resources=resources,
            scripts=scripts,
            user_id=user_id,
        )
        logger.info(
            "Skill '%s/%s' publish result: %s",
            plugin_name,
            skill_name,
            result,
        )
