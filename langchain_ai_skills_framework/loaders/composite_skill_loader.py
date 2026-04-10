from __future__ import annotations

import logging
from html import escape
from types import MappingProxyType
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.user_skill_store import (
    UserSkillStore,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.tools.delete_skill_tool import DeleteSkillTool
from langchain_ai_skills_framework.tools.load_skill_tool import LoadSkillTool
from langchain_ai_skills_framework.tools.read_skill_resource_tool import (
    ReadSkillResourceTool,
)
from langchain_ai_skills_framework.tools.run_skill_script_tool import (
    RunSkillScriptTool,
)
from langchain_ai_skills_framework.tools.save_skill_tool import SaveSkillTool
from langchain_ai_skills_framework.tools.toggle_skill_sharing_tool import (
    ToggleSkillSharingTool,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


class CompositeSkillLoader(SkillLoaderProtocol):
    """Merges a shared/global skill loader with a per-user MongoDB loader.

    The *shared_loader* handles filesystem / GitHub skills (with scripts
    and resources).  The *user_loader* handles MongoDB-persisted skills
    (content only — no scripts or resources).

    This class is a **singleton**.  Per-user context (``user_id``) is
    provided on each call that needs it, not at construction time.
    """

    def __init__(
        self,
        *,
        shared_loader: SkillLoaderProtocol,
        user_loader: UserSkillStore,
    ) -> None:
        if shared_loader is None:
            raise ValueError("shared_loader must not be None")
        if user_loader is None:
            raise ValueError("user_loader must not be None")

        self._shared = shared_loader
        self._user = user_loader

    @property
    def shared_loader(self) -> SkillLoaderProtocol:
        return self._shared

    @property
    def user_loader(self) -> UserSkillStore:
        return self._user

    # --- SkillLoaderProtocol implementation ----------------------------------

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        """Return shared skill summaries (user skills require async — see ``list_all_summaries``)."""
        return self._shared.list_skill_summaries(allowed_skills)

    async def list_all_summaries(
        self, *, user_id: str, allowed_skills: set[str]
    ) -> Sequence[SkillSummary]:
        """Return merged summaries from shared + user skills."""
        snapshot = await self._merged_snapshot(user_id=user_id)
        return snapshot.ordered_summaries

    def get_skill_details(self, skill_name: str) -> SkillDetails:
        """Get skill details from shared loader only (sync)."""
        return self._shared.get_skill_details(skill_name)

    async def get_skill_details_for_user(
        self, *, user_id: str, skill_name: str
    ) -> SkillDetails:
        """Get skill details checking user skills first, then shared."""
        normalized = skill_name.strip().lower().replace("_", "-")
        try:
            return await self._user.get_skill_details(
                user_id=user_id, skill_name=normalized
            )
        except SkillNotFoundError:
            return self._shared.get_skill_details(normalized)

    def refresh(self) -> None:
        self._shared.refresh()

    async def get_instructions(self) -> str:
        """Return shared skill instructions (no user context available here)."""
        return await self._shared.get_instructions()

    async def get_instructions_for_user(self, *, user_id: str) -> str:
        """Return merged skill instructions including user skills."""
        summaries = await self.list_all_summaries(user_id=user_id, allowed_skills=set())
        if not summaries:
            return await self._shared.get_instructions()

        lines: list[str] = []
        for skill in summaries:
            escaped_name = escape(skill.name, quote=True)
            escaped_description = escape(skill.description.strip(), quote=True)
            lines.append("<skill>")
            lines.append(f"<name>{escaped_name}</name>")
            lines.append(f"<description>{escaped_description}</description>")
            lines.append("</skill>")
        skills_list = "\n".join(lines)

        return (
            "You have access to a collection of skills containing domain-specific "
            "knowledge and capabilities.\n"
            "Each skill provides specialized instructions for specific tasks.\n\n"
            f"<available_skills>\n{skills_list}\n</available_skills>\n\n"
            "When a task falls within a skill's domain:\n"
            "1. Use `load_skill` to read the complete skill instructions\n"
            "2. Follow the skill's guidance to complete the task\n"
            "3. Use `read_skill_resource` to read files referenced by the skill\n"
            "4. Use `run_skill_script` to run scripts provided by the skill\n"
            "5. Use `save_skill` to save a new or updated skill for the current user\n"
            "6. Use `delete_skill` to remove a previously saved skill\n"
            "7. Use `toggle_skill_sharing` to share a skill with all users or make it private\n\n"
            "Use progressive disclosure: load only what you need, when you need it."
        )

    def get_tools(self) -> list[BaseTool]:
        """Return tools that resolve skills through this composite loader.

        Tools are constructed with ``skill_loader=self`` so that
        ``load_skill`` and availability lists include both shared and
        user-persisted skills, consistent with what
        ``get_instructions_for_user`` advertises.
        """
        return [
            LoadSkillTool(skill_loader=self),
            ReadSkillResourceTool(skill_loader=self),
            RunSkillScriptTool(skill_loader=self),
            SaveSkillTool(mongo_skill_loader=self._user),
            DeleteSkillTool(mongo_skill_loader=self._user),
            ToggleSkillSharingTool(mongo_skill_loader=self._user),
        ]

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        return self._shared.read_skill_resource(skill_name, resource_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        return await self._shared.run_skill_script(skill_name, script_name, arguments)

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return self._shared.list_skill_script_names(skill_name)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return self._shared.list_skill_resource_names(skill_name)

    # --- Merging -------------------------------------------------------------

    async def _merged_snapshot(self, *, user_id: str) -> SkillSnapshot:
        """Build a merged snapshot with precedence: GitHub → shared DB → user DB.

        GitHub/filesystem skills form the base.  Shared database skills
        overlay next (available to all users).  The requesting user's own
        skills win on name collision.
        """
        details: dict[str, SkillDetails] = {}

        # 1. GitHub / filesystem skills (lowest precedence)
        for summary in self._shared.list_skill_summaries(allowed_skills=set()):
            detail = self._shared.get_skill_details(summary.name)
            details[summary.name] = detail

        # 2. Shared database skills (override GitHub on collision)
        shared_snapshot = await self._user.load_shared_snapshot()
        for name, detail in shared_snapshot.details_by_name.items():
            details[name] = detail

        # 3. User's own database skills (highest precedence)
        user_snapshot = await self._user.load_snapshot(user_id=user_id)
        for name, detail in user_snapshot.details_by_name.items():
            details[name] = detail

        ordered = tuple(sorted(details.values(), key=lambda d: d.name))
        return SkillSnapshot(
            details_by_name=MappingProxyType(details),
            ordered_summaries=tuple(d.summary for d in ordered),
        )
