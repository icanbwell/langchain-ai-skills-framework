from __future__ import annotations

import asyncio
import logging
from html import escape
from types import MappingProxyType
from typing import Any, Sequence

from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import (
    MyScriptExecutor,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
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
from langchain_ai_skills_framework.tools.save_skill_resource_tool import (
    SaveSkillResourceTool,
)
from langchain_ai_skills_framework.tools.save_skill_script_tool import (
    SaveSkillScriptTool,
)
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
    including their resources and scripts.

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
        """Get skill details checking user skills first, then shared DB, then GitHub."""
        normalized = normalize_skill_name(skill_name)
        # 1. User's own skills (highest precedence)
        try:
            return await self._user.get_skill_details(
                user_id=user_id, skill_name=normalized
            )
        except SkillNotFoundError:
            pass

        # 2. Shared DB skills from other users
        shared_snapshot = await self._user.load_shared_snapshot()
        if normalized in shared_snapshot.details_by_name:
            return shared_snapshot.details_by_name[normalized]

        # 3. GitHub/filesystem skills (lowest precedence)
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

        # Batch fetch all usage counts in one aggregation query
        skill_names = [s.name for s in summaries]
        usage_counts = await self._user.get_skill_usage_counts(skill_names=skill_names)

        lines: list[str] = []
        for skill in summaries:
            escaped_name = escape(skill.name, quote=True)
            escaped_description = escape(skill.description.strip(), quote=True)
            lines.append("<skill>")
            lines.append(f"<name>{escaped_name}</name>")
            lines.append(f"<description>{escaped_description}</description>")
            lines.append(
                f"<usage_count>{usage_counts.get(skill.name, 0)}</usage_count>"
            )
            author = skill.metadata.get("user_id") if skill.metadata else None
            if author:
                escaped_author = escape(str(author), quote=True)
                lines.append(f"<author>{escaped_author}</author>")
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
            "6. Use `save_skill_resource` to save a resource file for a skill\n"
            "7. Use `save_skill_script` to save a script file for a skill\n"
            "8. Use `delete_skill` to remove a previously saved skill\n"
            "9. Use `toggle_skill_sharing` to share a skill with all users or make it private\n\n"
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
            LoadSkillTool(skill_loader=self, user_skill_store=self._user),
            ReadSkillResourceTool(skill_loader=self),
            RunSkillScriptTool(skill_loader=self),
            SaveSkillTool(mongo_skill_loader=self._user),
            SaveSkillResourceTool(mongo_skill_loader=self._user),
            SaveSkillScriptTool(mongo_skill_loader=self._user),
            DeleteSkillTool(mongo_skill_loader=self._user),
            ToggleSkillSharingTool(mongo_skill_loader=self._user),
        ]

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        return self._shared.read_skill_resource(skill_name, resource_name)

    async def read_skill_resource_for_user(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> str:
        """Read a resource, checking user's MongoDB skills first, then shared loader."""
        normalized = normalize_skill_name(skill_name)

        # Check user's own skills first
        try:
            return await self._user.read_resource(
                user_id=user_id, skill_name=normalized, resource_name=resource_name
            )
        except SkillNotFoundError:
            logger.debug(
                "Resource '%s' not found in user skill '%s' for user '%s', trying shared skills",
                resource_name,
                normalized,
                user_id,
            )

        # Check shared DB skills
        shared_snapshot = await self._user.load_shared_snapshot()
        if normalized in shared_snapshot.details_by_name:
            shared_detail = shared_snapshot.details_by_name[normalized]
            owner_user_id = str(
                shared_detail.summary.metadata.get("user_id", "")
                if shared_detail.summary.metadata
                else ""
            )
            if owner_user_id:
                try:
                    return await self._user.read_resource(
                        user_id=owner_user_id,
                        skill_name=normalized,
                        resource_name=resource_name,
                    )
                except SkillNotFoundError:
                    pass

        # Fall back to shared filesystem loader
        return self._shared.read_skill_resource(normalized, resource_name)

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        return await self._shared.run_skill_script(skill_name, script_name, arguments)

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        """Run a script, checking user's MongoDB skills first, then shared loader.

        MongoDB-stored scripts are executed in a subprocess with the script
        content written to a temporary file.
        """
        normalized = normalize_skill_name(skill_name)

        # Check user's own scripts first
        try:
            script_content = await self._user.read_script(
                user_id=user_id, skill_name=normalized, script_name=script_name
            )
            return await self._execute_script_content(
                script_content=script_content,
                script_name=script_name,
                arguments=arguments,
            )
        except SkillNotFoundError:
            pass

        # Check shared DB skills
        shared_snapshot = await self._user.load_shared_snapshot()
        if normalized in shared_snapshot.details_by_name:
            shared_detail = shared_snapshot.details_by_name[normalized]
            owner_user_id = str(
                shared_detail.summary.metadata.get("user_id", "")
                if shared_detail.summary.metadata
                else ""
            )
            if owner_user_id:
                try:
                    script_content = await self._user.read_script(
                        user_id=owner_user_id,
                        skill_name=normalized,
                        script_name=script_name,
                    )
                    return await self._execute_script_content(
                        script_content=script_content,
                        script_name=script_name,
                        arguments=arguments,
                    )
                except SkillNotFoundError:
                    pass

        # Fall back to shared filesystem loader
        return await self._shared.run_skill_script(normalized, script_name, arguments)

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return self._shared.list_skill_script_names(skill_name)

    async def list_skill_script_names_for_user(
        self, *, user_id: str, skill_name: str
    ) -> Sequence[str]:
        """List scripts, merging user MongoDB scripts with shared loader scripts."""
        normalized = normalize_skill_name(skill_name)
        names: set[str] = set()

        # User's own scripts
        try:
            user_scripts = await self._user.list_script_names(
                user_id=user_id, skill_name=normalized
            )
            names.update(user_scripts)
        except (SkillNotFoundError, ValueError):
            pass

        # Shared loader scripts
        try:
            shared_scripts = self._shared.list_skill_script_names(normalized)
            names.update(shared_scripts)
        except SkillNotFoundError:
            pass

        return sorted(names)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return self._shared.list_skill_resource_names(skill_name)

    async def list_skill_resource_names_for_user(
        self, *, user_id: str, skill_name: str
    ) -> Sequence[str]:
        """List resources, merging user MongoDB resources with shared loader resources."""
        normalized = normalize_skill_name(skill_name)
        names: set[str] = set()

        # User's own resources
        try:
            user_resources = await self._user.list_resource_names(
                user_id=user_id, skill_name=normalized
            )
            names.update(user_resources)
        except (SkillNotFoundError, ValueError):
            pass

        # Shared loader resources
        try:
            shared_resources = self._shared.list_skill_resource_names(normalized)
            names.update(shared_resources)
        except SkillNotFoundError:
            pass

        return sorted(names)

    # --- Merging -------------------------------------------------------------

    async def _merged_snapshot(self, *, user_id: str) -> SkillSnapshot:
        """Build a merged snapshot with precedence: GitHub -> shared DB -> user DB.

        GitHub/filesystem skills form the base.  Shared database skills
        overlay next (available to all users).  The requesting user's own
        skills win on name collision.
        """
        details: dict[str, SkillDetails] = {}

        # 1. GitHub / filesystem skills (lowest precedence, sync + cached)
        for summary in self._shared.list_skill_summaries(allowed_skills=set()):
            detail = self._shared.get_skill_details(summary.name)
            details[summary.name] = detail

        # 2+3. Load shared and user snapshots concurrently
        shared_snapshot, user_snapshot = await asyncio.gather(
            self._user.load_shared_snapshot(),
            self._user.load_snapshot(user_id=user_id),
        )

        # Shared database skills (override GitHub on collision)
        for name, detail in shared_snapshot.details_by_name.items():
            details[name] = detail

        # User's own database skills (highest precedence)
        for name, detail in user_snapshot.details_by_name.items():
            details[name] = detail

        ordered = tuple(sorted(details.values(), key=lambda d: d.name))
        return SkillSnapshot(
            details_by_name=MappingProxyType(details),
            ordered_summaries=tuple(d.summary for d in ordered),
        )

    # --- Script execution ----------------------------------------------------

    @staticmethod
    async def _execute_script_content(
        *,
        script_content: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        """Execute a script stored as content in MongoDB."""
        executor = MyScriptExecutor()
        return await executor.execute_inline_script(
            script_name=script_name,
            script=script_content,
            arguments=arguments or {},
        )
