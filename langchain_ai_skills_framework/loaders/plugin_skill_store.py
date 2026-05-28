"""Protocol for plugin-scoped skill persistence.

Every operation is keyed by ``(user_id, plugin_name, skill_name)`` so that
marketplace-synced skills (``user_id="system"``) and user-saved overrides
coexist in the same collections.

Replaces the legacy ``UserSkillStore`` protocol.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
    MongoPluginSkillUsageDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
)


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
        user_id: str,
        plugin_name: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoPluginSkillDocument: ...

    async def set_skill_published(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        published: bool,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument: ...

    async def delete_skill(self, *, user_id: str, plugin_name: str, skill_name: str) -> bool: ...

    async def load_snapshot(self, *, user_id: str, plugin_name: str | None = None) -> SkillSnapshot: ...

    async def load_shared_snapshot(self, *, plugin_name: str | None = None) -> SkillSnapshot: ...

    async def get_skill_details(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> SkillDetails: ...

    async def skill_exists(self, *, user_id: str, plugin_name: str | None = None, skill_name: str) -> bool: ...

    # --- Resource operations ---

    async def save_resource(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoPluginResourceDocument: ...

    async def delete_resource(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool: ...

    async def read_resource(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> str: ...

    async def list_resource_names(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]: ...

    async def resource_exists(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> bool: ...

    # --- Script operations ---

    async def save_script(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoPluginScriptDocument: ...

    async def delete_script(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool: ...

    async def read_script(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> str: ...

    async def list_script_names(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]: ...

    async def script_exists(
        self,
        *,
        user_id: str,
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
    ) -> MongoPluginDefinitionDocument: ...

    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]: ...

    async def has_plugins(self) -> bool: ...
