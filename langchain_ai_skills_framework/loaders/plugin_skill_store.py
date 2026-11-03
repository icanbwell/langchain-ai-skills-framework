"""Protocol for plugin-scoped skill persistence.

Every operation is keyed by ``(author, plugin_name, skill_name)`` so that
marketplace-synced skills (``author="system"``) and user-saved overrides
coexist in the same collections.

Replaces the legacy ``UserSkillStore`` protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from langchain_ai_skills_framework.loaders.summary_page_cursor import DEFAULT_PAGE_LIMIT
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
    MongoPluginSkillUsageDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    AllAuthorsSkillSnapshot,
    AllAuthorsSummaryPage,
    SkillDetails,
    SkillSnapshot,
)


@runtime_checkable
class AllAuthorsSnapshotStore(Protocol):
    """Optional capability: read every author's skills.

    Kept separate from ``PluginSkillStore`` so existing external implementers of that
    protocol keep satisfying it (a new required method would break them, and runtime
    ``isinstance`` checks against it).
    """

    async def load_all_snapshot(self, *, include_staging: bool = False) -> AllAuthorsSkillSnapshot:
        """Return every author's skills, keyed by ``(author, plugin_name, skill_name)``.

        Args:
            include_staging: Also include skills in the ``staging`` state.

        Returns:
            An immutable snapshot spanning all authors, drafts included. The read is unbounded: it loads every
            skill document plus every resource and script document of those skills. For listing, prefer
            ``AllAuthorsSummaryPageStore.list_all_summaries_page``.

        The store performs no authorization: the caller must verify the requesting user may
        see other users' skills before calling this.
        """
        ...


@runtime_checkable
class AllAuthorsSummaryPageStore(Protocol):
    """Optional capability: list every author's skills as bounded, summaries-only pages (BAI-1509).

    Separate from ``AllAuthorsSnapshotStore`` so implementers of that protocol keep satisfying it. Discover it with
    ``isinstance`` and raise ``TypeError`` when absent, as the all-authors path does.
    """

    async def list_all_summaries_page(
        self, *, include_staging: bool = False, limit: int = DEFAULT_PAGE_LIMIT, cursor: str | None = None
    ) -> AllAuthorsSummaryPage:
        """Return one page of every author's skill summaries.

        Args:
            include_staging: Also include skills in the ``staging`` state.
            limit: Maximum summaries in the page; at least 1, clamped to ``MAX_PAGE_LIMIT``.
            cursor: ``next_cursor`` of the previous page, or ``None`` for the first page.

        Returns:
            Up to ``limit`` summaries ordered by ``(plugin_name, skill_name, author)`` with no ``manifest``, and
            ``next_cursor`` (``None`` on the last page). Resource and script documents are never read.

        Raises:
            ValueError: If ``limit`` is below 1.
            InvalidPageCursorError: If ``cursor`` is malformed or from an incompatible version.

        The store performs no authorization: the caller must verify the requesting user may see other users' skills.
        """
        ...


@runtime_checkable
class PluginSkillStore(Protocol):
    """Abstraction over plugin-scoped skill persistence.

    Implementations may be backed by MongoDB or a null object.
    """

    async def ensure_indexes(self) -> None: ...

    # --- Skill operations ---

    async def save_skill(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
        folder: str | None = None,
        path: str | None = None,
        state: str | None = None,
    ) -> MongoPluginSkillDocument: ...

    async def set_skill_state(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        state: str,
        published_branch: str | None = None,
        modified_by: str = "",
    ) -> MongoPluginSkillDocument: ...

    async def delete_skill(self, *, author: str, plugin_name: str, skill_name: str, modified_by: str = "") -> bool: ...

    async def load_snapshot(
        self, *, author: str, plugin_name: str | None = None, include_staging: bool = False
    ) -> SkillSnapshot: ...

    async def load_shared_snapshot(
        self, *, plugin_name: str | None = None, include_staging: bool = False
    ) -> SkillSnapshot: ...

    async def get_skill_details(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> SkillDetails: ...

    async def skill_exists(self, *, author: str, plugin_name: str | None = None, skill_name: str) -> bool: ...

    # --- Resource operations ---

    async def save_resource(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        content: str,
        modified_by: str = "",
        folder: str | None = None,
        path: str | None = None,
    ) -> MongoPluginResourceDocument: ...

    async def delete_resource(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        modified_by: str = "",
    ) -> bool: ...

    async def read_resource(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> str: ...

    async def list_resource_names(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]: ...

    async def list_resource_documents(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[MongoPluginResourceDocument]: ...

    async def resource_exists(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> bool: ...

    # --- Script operations ---

    async def save_script(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        content: str,
        modified_by: str = "",
        folder: str | None = None,
        path: str | None = None,
    ) -> MongoPluginScriptDocument: ...

    async def delete_script(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        modified_by: str = "",
    ) -> bool: ...

    async def read_script(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> str: ...

    async def list_script_names(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]: ...

    async def list_script_documents(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[MongoPluginScriptDocument]: ...

    async def script_exists(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> bool: ...

    # --- Usage tracking ---

    async def record_skill_usage(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> MongoPluginSkillUsageDocument: ...

    async def get_skill_usage_count(self, *, skill_name: str) -> int: ...

    async def get_skill_usage_counts(self, *, skill_names: Sequence[str]) -> Mapping[str, int]: ...

    # --- Plugin catalog ---

    async def save_plugin(
        self,
        *,
        plugin_name: str,
        description: str,
        skills: Sequence[str],
        mcp_servers: Sequence[dict[str, object]],
        mcp_servers_skipped: Sequence[dict[str, object]] = (),
    ) -> MongoPluginDefinitionDocument: ...

    async def plugin_exists(self, *, plugin_name: str) -> bool: ...

    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]: ...

    async def has_plugins(self) -> bool: ...
