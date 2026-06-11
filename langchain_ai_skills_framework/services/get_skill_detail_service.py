from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_non_empty,
    require_user_id,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


@dataclass(frozen=True, slots=True)
class SkillDetailResult:
    """Result returned by :class:`GetSkillDetailService`."""

    content: str
    resources: list[str]
    scripts: list[str]
    folder: str | None
    state: str


class GetSkillDetailService:
    """Load skill content and metadata with user→system fallback for metadata."""

    def __init__(self, *, skill_loader: SkillLoaderProtocol, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._loader = skill_loader
        self._store = mongo_skill_loader

    async def execute(self, *, user_id: str, plugin_name: str, skill_name: str) -> SkillDetailResult:
        """Load skill details with user→system fallback for metadata.

        Raises:
            SkillOperationError: If skill content not found or validation fails
        """
        require_user_id(user_id=user_id, operation="get_skill_detail")
        require_non_empty(value=plugin_name, label="plugin_name")
        require_non_empty(value=skill_name, label="skill_name")

        # Load content from loader
        try:
            details = await self._loader.get_skill_details_for_user(
                user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
            )
        except SkillNotFoundError as exc:
            logger.warning(
                "Skill not found: user_id=%s plugin=%s skill=%s",
                user_id,
                plugin_name,
                skill_name,
            )
            raise SkillOperationError(f"Skill not found: {plugin_name}/{skill_name}") from exc

        # If no store, return basic info from loader
        if self._store is None:
            folder = details.summary.folder
            state = details.summary.state
            resources: list[str] = []
            scripts: list[str] = []
        else:
            # Marketplace-synced resources/scripts live under author="system";
            # user overrides live under the user's own id. Return the union so
            # users see the full set after the shared loader is null-ified
            # post-init (reads must come from Mongo only).
            resources = await self._list_with_system_fallback(
                kind="resource", user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
            )
            scripts = await self._list_with_system_fallback(
                kind="script", user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
            )

            # Resolve metadata with user→system fallback
            folder, state = await self._resolve_metadata(
                user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
            )

        logger.debug(
            "GetSkillDetailService: loaded %s/%s for user=%s (folder=%s, state=%s, %d resources, %d scripts)",
            plugin_name,
            skill_name,
            user_id,
            folder,
            state,
            len(resources),
            len(scripts),
        )

        return SkillDetailResult(
            content=details.content,
            resources=resources,
            scripts=scripts,
            folder=folder,
            state=state,
        )

    async def _list_with_system_fallback(
        self, *, kind: str, user_id: str, plugin_name: str, skill_name: str
    ) -> list[str]:
        assert self._store is not None
        list_fn = self._store.list_resource_names if kind == "resource" else self._store.list_script_names
        user_names = await list_fn(author=user_id, plugin_name=plugin_name, skill_name=skill_name)
        system_names = await list_fn(author="system", plugin_name=plugin_name, skill_name=skill_name)
        return sorted(set(user_names) | set(system_names))

    async def _resolve_metadata(self, *, user_id: str, plugin_name: str, skill_name: str) -> tuple[str | None, str]:
        """Resolve folder and state with user→system→default fallback."""
        if self._store is None:
            return None, "draft"

        # Try user author first
        try:
            details = await self._store.get_skill_details(
                author=user_id, plugin_name=plugin_name, skill_name=skill_name
            )
            return details.summary.folder, details.summary.state
        except SkillNotFoundError:
            logger.debug("Skill metadata not found for user=%s, trying system author", user_id)

        # Fallback to system author
        try:
            details = await self._store.get_skill_details(
                author="system", plugin_name=plugin_name, skill_name=skill_name
            )
            return details.summary.folder, details.summary.state
        except SkillNotFoundError:
            logger.debug("Skill metadata not found for system author, using defaults")

        # Default if both fail
        return None, "draft"
