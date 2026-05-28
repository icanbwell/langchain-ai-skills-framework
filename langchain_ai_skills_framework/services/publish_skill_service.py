from __future__ import annotations

import asyncio
import logging

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from langchain_ai_skills_framework.services.save_skill_service import SaveSkillService
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_non_empty,
    require_store,
    require_user_id,
)
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

    # Intentionally class-level and shared across all instances.
    # PublishSkillService is instantiated per-request by PublishSkillTool,
    # but rapid publish/unpublish toggles for the *same* skill must be
    # serialized globally — a per-instance dict would lose track of the
    # previous task.  All access is single-threaded within one asyncio
    # event loop, so no mutex is needed.  The done-callback in execute()
    # removes completed entries to prevent unbounded growth.
    _pending_tasks: dict[str, asyncio.Task[None]] = {}

    def __init__(
        self,
        *,
        mongo_skill_loader: PluginSkillStore | None,
        marketplace_publisher: GitHubMarketplacePublisher | None = None,
    ) -> None:
        self._store = mongo_skill_loader
        self._publisher = marketplace_publisher

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str | None = None,
        content: str | None = None,
        published: bool,
        branch_name: str | None = None,
    ) -> str:
        """Toggle publish state for a skill.

        When ``skill_name`` is None, it is extracted from the ``name``
        field in the ``content`` frontmatter.
        """
        require_user_id(user_id=user_id, operation="publish_skill")

        if not skill_name:
            if content:
                skill_name = SaveSkillService.resolve_skill_name(content)
            if not skill_name:
                raise SkillOperationError(
                    "skill_name is required when content is not provided or does not contain a 'name' field."
                )

        require_non_empty(value=skill_name, label="skill_name")
        store = require_store(store=self._store)

        if published and self._publisher is None:
            raise SkillOperationError(
                f"Cannot publish skill '{skill_name}': GitHub marketplace publisher is not configured. "
                "Ensure PLUGINS_MARKETPLACE is set to a valid github:// URI and "
                "PLUGINS_MARKETPLACE_PUBLISH_ENABLED is true."
            )

        try:
            branch: str | None = None
            if self._publisher is not None and self._publisher.use_branch:
                if branch_name:
                    branch = branch_name
                else:
                    branch = (
                        f"skill-publish/{plugin_name}/{skill_name}"
                        if published
                        else f"skill-unpublish/{plugin_name}/{skill_name}"
                    )

            doc = await store.set_skill_published(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                published=published,
                published_branch=branch,
            )
            state = "published" if doc.published else "unpublished"
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
                        published=published,
                        branch_name=branch_name,
                    ),
                    name=f"marketplace-{'publish' if published else 'unpublish'}-{task_key}",
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
        published: bool,
        branch_name: str | None = None,
    ) -> None:
        """Best-effort publish/unpublish. Exceptions are logged, never raised."""
        if self._publisher is None or self._store is None:
            return
        try:
            if published:
                await self._publish_skill(
                    user_id=user_id,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                    branch_name=branch_name,
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
                "publish" if published else "unpublish",
                plugin_name,
                skill_name,
            )

    async def _publish_skill(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        branch_name: str | None = None,
    ) -> None:
        if self._publisher is None or self._store is None:
            return

        logger.info(
            "Gathering skill content for '%s/%s' (user=%s)",
            plugin_name,
            skill_name,
            user_id,
        )
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

        logger.info(
            "Publishing '%s/%s' to marketplace: %d resource(s), %d script(s)",
            plugin_name,
            skill_name,
            len(resources),
            len(scripts),
        )
        effective_branch = branch_name if self._publisher.use_branch else None
        result = await self._publisher.publish_skill(
            plugin_name=plugin_name,
            skill_name=skill_name,
            skill_content=details.content,
            resources=resources,
            scripts=scripts,
            user_id=user_id,
            branch_name=effective_branch,
        )
        logger.info(
            "Skill '%s/%s' publish result: %s",
            plugin_name,
            skill_name,
            result,
        )
