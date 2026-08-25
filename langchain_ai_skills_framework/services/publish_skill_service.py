from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
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
    """Submit a skill for review (published=True) or revert to draft (published=False).

    When ``published=True``, the service enforces a review gate:
    1. Ensures the skill exists in MongoDB (auto-saves from skill_loader if missing)
    2. Validates that the skill is in "staging" state
    3. Sets state to "in_review"
    4. Fires a background PR task (if marketplace_publisher is configured)

    When ``published=False``, the service:
    1. Sets state to "draft" (no state validation)
    2. Fires a background unpublish task

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
    _pending_tasks: ClassVar[dict[str, asyncio.Task[None]]] = {}

    def __init__(
        self,
        *,
        mongo_skill_loader: PluginSkillStore | None,
        marketplace_publisher: GitHubMarketplacePublisher | None = None,
        skill_loader: SkillLoaderProtocol | None = None,
    ) -> None:
        self._store = mongo_skill_loader
        self._publisher = marketplace_publisher
        self._skill_loader = skill_loader

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
        """Submit skill for review (published=True) or revert to draft (published=False).

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
            logger.info(
                "PublishSkillService: GitHub marketplace publisher is not configured. "
                "Skill '%s' state will be updated locally but no PR will be created.",
                skill_name,
            )

        try:
            # When publishing, enforce the review gate
            if published:
                await self._ensure_exists_in_staging(
                    store=store,
                    user_id=user_id,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                )
                await self._validate_staging_state(
                    store=store,
                    user_id=user_id,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                )

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

            # Set state: "in_review" for publish, "draft" for unpublish
            target_state = "in_review" if published else "draft"
            doc = await store.set_skill_state(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                state=target_state,
                published_branch=branch,
            )

            message = (
                f"Skill '{doc.skill_name}' submitted for review."
                if published
                else f"Skill '{doc.skill_name}' is now unpublished."
            )
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
        except SkillOperationError:
            raise
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
    # Review gate helpers
    # ------------------------------------------------------------------

    async def _ensure_exists_in_staging(
        self,
        *,
        store: PluginSkillStore,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> None:
        """Ensure skill exists in MongoDB. Auto-save from skill_loader if missing."""
        exists = await store.skill_exists(
            author=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        if exists:
            return

        # Skill not in MongoDB yet
        if self._skill_loader is None:
            raise SkillOperationError(
                f"Skill '{skill_name}' does not exist and no skill_loader is configured to load it."
            )

        try:
            details = await self._skill_loader.get_skill_details_for_user(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
        except SkillNotFoundError as exc:
            raise SkillOperationError(f"Skill '{skill_name}' not found in skill_loader.") from exc

        # Save with state="staging"
        await store.save_skill(
            author=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            content=details.content,
            state="staging",
            modified_by=user_id,
        )
        logger.info(
            "Auto-saved skill '%s/%s' with state='staging' (user=%s)",
            plugin_name,
            skill_name,
            user_id,
        )

    async def _validate_staging_state(
        self,
        *,
        store: PluginSkillStore,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> None:
        """Validate that the skill is in 'staging' state before publishing."""
        details = await store.get_skill_details(
            author=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        if details.summary.state != "staging":
            raise SkillOperationError(
                f"Skill '{skill_name}' must be in 'staging' state to be submitted for review. "
                f"Current state: '{details.summary.state}'."
            )

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
                skill_dir = await self._resolve_skill_dir(
                    user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
                )
                result = await self._publisher.unpublish_skill(
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                    user_id=user_id,
                    skill_dir=skill_dir,
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
            author=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )

        resource_docs = await self._store.list_resource_documents(
            author=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        resources: dict[str, str] = {doc.resource_name: doc.content for doc in resource_docs}
        resource_paths: dict[str, str] = {
            doc.resource_name: self._marketplace_path(stored=doc.path) for doc in resource_docs if doc.path
        }

        script_docs = await self._store.list_script_documents(
            author=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        scripts: dict[str, str] = {doc.script_name: doc.content for doc in script_docs}
        script_paths: dict[str, str] = {
            doc.script_name: self._marketplace_path(stored=doc.path) for doc in script_docs if doc.path
        }

        skill_path = self._marketplace_path(stored=details.summary.path) if details.summary.path else None

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
            skill_path=skill_path,
            resource_paths=resource_paths or None,
            script_paths=script_paths or None,
        )
        logger.info(
            "Skill '%s/%s' publish result: %s",
            plugin_name,
            skill_name,
            result,
        )

    async def _resolve_skill_dir(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> str | None:
        """Return the marketplace directory for a skill based on its stored path.

        Returns None when the skill is unknown or has no stored path; callers
        then fall back to the publisher's default ``plugins/{plugin}/skills/{skill}``
        layout.
        """
        if self._store is None:
            return None
        try:
            details = await self._store.get_skill_details(
                author=user_id, plugin_name=plugin_name, skill_name=skill_name
            )
        except SkillNotFoundError:
            return None
        stored = details.summary.path
        if not stored:
            return None
        # _marketplace_path always returns a path under "plugins/...", so it
        # contains at least one slash — rsplit safely yields the parent dir.
        return self._marketplace_path(stored=stored).rsplit("/", 1)[0]

    @staticmethod
    def _marketplace_path(*, stored: str) -> str:
        """Prefix a stored materialized path with the marketplace ``plugins/`` root.

        Stored paths use ``{plugin}/skills/[folder/]{name}/...``; the marketplace
        layout nests everything under ``plugins/``. Already-prefixed paths pass through.
        """
        cleaned = stored.strip().lstrip("/")
        if cleaned.startswith("plugins/"):
            return cleaned
        return f"plugins/{cleaned}"
