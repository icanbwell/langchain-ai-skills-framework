from __future__ import annotations

import logging
from dataclasses import dataclass

from skills_ref.errors import ParseError
from skills_ref.parser import parse_frontmatter
from skills_ref.validator import validate_metadata

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_non_empty,
    require_store,
    require_user_id,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


@dataclass(frozen=True, slots=True)
class SaveSkillResult:
    """Outcome of a save_skill operation.

    ``ok=True`` means the skill was persisted. ``ok=False`` means a soft
    failure (validation rejected the content, or the skill already
    exists with ``update_if_exists=False``) that callers should surface
    as a client error — HTTP 400, not 200 — so downstream UI does not
    treat the save as successful and immediately try to load the
    nonexistent skill.
    """

    ok: bool
    message: str


class SaveSkillService:
    """Save or update a skill in the user skill store."""

    def __init__(self, *, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._store = mongo_skill_loader

    @staticmethod
    def resolve_skill_name(content: str) -> str | None:
        """Extract skill name from content frontmatter.

        Returns the name string or None if it cannot be determined.
        """
        try:
            metadata, _ = parse_frontmatter(content)
            name = metadata.get("name")
            return name.strip() if isinstance(name, str) and name.strip() else None
        except ParseError:
            return None

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str | None = None,
        content: str,
        update_if_exists: bool = True,
        folder: str | None = None,
        path: str | None = None,
        state: str | None = None,
    ) -> SaveSkillResult:
        """Validate and persist the skill, returning a structured result.

        Returns ``SaveSkillResult(ok=True, ...)`` when the skill is
        persisted, and ``SaveSkillResult(ok=False, ...)`` on soft
        failures (validation rejected the content, or the skill already
        exists with ``update_if_exists=False``). Raises
        ``SkillOperationError`` for hard failures that should surface as
        tool errors.

        When ``skill_name`` is None, it is extracted from the ``name``
        field in the content frontmatter.
        """
        require_user_id(user_id=user_id, operation="save_skill")
        require_non_empty(value=content, label="content")
        store = require_store(store=self._store)

        # Validate skill content
        try:
            metadata, _ = parse_frontmatter(content)
        except ParseError as exc:
            return SaveSkillResult(ok=False, message=f"Skill validation failed: {exc}")

        if not skill_name:
            skill_name = metadata.get("name")
            if isinstance(skill_name, str):
                skill_name = skill_name.strip()
            if not skill_name:
                return SaveSkillResult(
                    ok=False,
                    message="Skill validation failed: 'name' field missing from frontmatter and no skill_name provided.",
                )

        require_non_empty(value=skill_name, label="skill_name")

        validation_errors = validate_metadata(metadata)
        if validation_errors:
            error_details = "; ".join(validation_errors)
            return SaveSkillResult(
                ok=False,
                message=f"Skill validation failed ({len(validation_errors)} error(s)): {error_details}",
            )

        if not update_if_exists:
            exists = await store.skill_exists(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            if exists:
                return SaveSkillResult(
                    ok=False,
                    message=f"Skill '{skill_name}' already exists. Set update_if_exists=true to overwrite.",
                )

        try:
            doc = await store.save_skill(
                author=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                content=content,
                modified_by=user_id,
                folder=folder,
                path=path,
                state=state,
            )
            message = f"Skill '{doc.skill_name}' saved successfully."
            logger.info("SaveSkillService: %s (user=%s)", message, user_id)
            return SaveSkillResult(ok=True, message=message)
        except Exception as exc:
            logger.exception(
                "SaveSkillService failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise SkillOperationError(f"Unable to save skill '{skill_name}' due to an internal error.") from exc
