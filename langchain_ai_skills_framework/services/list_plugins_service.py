from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
    require_store,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Lightweight summary returned by :class:`ListPluginsService`."""

    name: str
    description: str
    skills: list[str]


class ListPluginsService:
    """List plugins registered in the store."""

    def __init__(self, *, mongo_skill_loader: PluginSkillStore | None) -> None:
        self._store = mongo_skill_loader

    async def execute(self) -> Sequence[PluginInfo]:
        """Return registered plugins as a sorted sequence of ``PluginInfo``."""
        store = require_store(store=self._store)

        try:
            docs = await store.list_plugins()
        except Exception as exc:
            logger.exception("ListPluginsService: failed to list plugins")
            raise SkillOperationError("Unable to list plugins due to an internal error.") from exc

        results = sorted(
            (
                PluginInfo(
                    name=d.plugin_name,
                    description=d.description,
                    skills=list(d.skills),
                )
                for d in docs
            ),
            key=lambda p: p.name,
        )
        logger.debug("ListPluginsService: found %d plugins", len(results))
        return results

    @staticmethod
    def format_as_text(plugins: Sequence[PluginInfo]) -> str:
        """Format the plugin list as an XML string."""
        if not plugins:
            return "<available_plugins>\n</available_plugins>"
        elements = []
        for p in plugins:
            skills_str = ", ".join(p.skills) if p.skills else ""
            elements.append(
                f"<plugin>\n"
                f"<name>{p.name}</name>\n"
                f"<description>{p.description}</description>\n"
                f"<skills>{skills_str}</skills>\n"
                f"</plugin>"
            )
        return "<available_plugins>\n" + "\n".join(elements) + "\n</available_plugins>"
