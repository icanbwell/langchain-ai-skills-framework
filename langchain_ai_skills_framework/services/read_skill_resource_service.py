from __future__ import annotations

import logging

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.services.availability_helpers import (
    format_resource_availability,
    format_skill_availability,
)
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class ReadSkillResourceService:
    """Read a supplementary resource from a skill."""

    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self._loader = skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> tuple[str, str]:
        """Return ``(content, artifact)``."""
        normalized_name = skill_name.strip()
        if not normalized_name:
            raise SkillOperationError(
                await format_skill_availability(loader=self._loader, normalized_name=normalized_name, user_id=user_id)
            )

        normalized_resource_name = resource_name.strip()
        if not normalized_resource_name:
            return "No resource name provided.", ""

        content = await self._load_resource(
            skill_name=normalized_name,
            resource_name=normalized_resource_name,
            plugin_name=plugin_name,
            user_id=user_id,
        )
        logger.debug(
            "ReadSkillResourceService: Loaded resource_name=%s from skill_name=%s",
            normalized_resource_name,
            normalized_name,
        )
        return content, content

    async def _load_resource(
        self,
        *,
        skill_name: str,
        resource_name: str,
        plugin_name: str,
        user_id: str,
    ) -> str:
        try:
            if user_id:
                resource: str = await self._loader.read_skill_resource_for_user(
                    author=user_id,
                    plugin_name=plugin_name,
                    skill_name=skill_name,
                    resource_name=resource_name,
                )
            else:
                resource = self._loader.read_skill_resource(
                    skill_name=skill_name, resource_name=resource_name, plugin_name=plugin_name
                )
            return resource
        except SkillNotFoundError:
            return await format_resource_availability(
                loader=self._loader,
                skill_name=skill_name,
                resource_name=resource_name,
                user_id=user_id,
                plugin_name=plugin_name,
            )
        except Exception as exc:
            logger.exception(
                "ReadSkillResourceService failed for skill_name=%s resource_name=%s",
                skill_name,
                resource_name,
            )
            raise SkillOperationError(
                f"Unable to read resource '{resource_name}' from skill '{skill_name}' due to an internal error."
            ) from exc
