from __future__ import annotations

import logging

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
    ) -> str:
        """Validate and persist the skill, returning a status message.

        Returns the message as a plain string on success or on validation
        failure (soft error).  Raises ``SkillOperationError`` for hard
        failures that should surface as tool errors.

        When ``skill_name`` is None, it is extracted from the ``name``
        field in the content frontmatter.

        When ``update_if_exists`` is False, returns an error message if a
        skill with the same name already exists for this user/plugin.
        """
        require_user_id(user_id, "save_skill")
        require_non_empty(content, "content")
        store = require_store(self._store)

        # Validate skill content
        try:
            metadata, _ = parse_frontmatter(content)
        except ParseError as exc:
            message = f"Skill validation failed: {exc}"
            return message

        if not skill_name:
            skill_name = metadata.get("name")
            if isinstance(skill_name, str):
                skill_name = skill_name.strip()
            if not skill_name:
                return "Skill validation failed: 'name' field missing from frontmatter and no skill_name provided."

        require_non_empty(skill_name, "skill_name")

        validation_errors = validate_metadata(metadata)
        if validation_errors:
            error_details = "; ".join(validation_errors)
            return f"Skill validation failed ({len(validation_errors)} error(s)): {error_details}"

        if not update_if_exists:
            exists = await store.skill_exists(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            if exists:
                return f"Skill '{skill_name}' already exists. Set update_if_exists=true to overwrite."

        try:
            doc = await store.save_skill(
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                content=content,
                modified_by=user_id,
            )
            message = f"Skill '{doc.skill_name}' saved successfully."
            logger.info("SaveSkillService: %s (user=%s)", message, user_id)
            return message
        except Exception as exc:
            logger.exception(
                "SaveSkillService failed for skill_name=%s user=%s",
                skill_name,
                user_id,
            )
            raise SkillOperationError(f"Unable to save skill '{skill_name}' due to an internal error.") from exc
