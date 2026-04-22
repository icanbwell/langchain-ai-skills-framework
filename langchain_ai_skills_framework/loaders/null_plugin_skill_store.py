"""No-op ``PluginSkillStore`` used when MongoDB is not configured.

Reads return empty results.  Writes raise ``RuntimeError``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
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

_EMPTY_SNAPSHOT = SkillSnapshot(
    details_by_name=MappingProxyType({}),
    ordered_summaries=(),
)

_NOT_CONFIGURED_MSG = (
    "Plugin skill storage is not configured. Set MONGO_SKILLS_URI or MONGO_URL to enable plugin skills."
)


class NullPluginSkillStore:
    """No-op ``PluginSkillStore`` used when MongoDB is not configured."""

    async def ensure_indexes(self) -> None:
        return

    # --- Skill operations ---

    async def save_skill(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoPluginSkillDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def set_skill_shared(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        shared: bool,
    ) -> MongoPluginSkillDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def delete_skill(self, *, user_id: str, plugin_name: str, skill_name: str) -> bool:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def load_snapshot(self, *, user_id: str, plugin_name: str = "") -> SkillSnapshot:
        return _EMPTY_SNAPSHOT

    async def load_shared_snapshot(self, *, plugin_name: str = "") -> SkillSnapshot:
        return _EMPTY_SNAPSHOT

    async def get_skill_details(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> SkillDetails:
        raise SkillNotFoundError(
            f"Skill '{skill_name}' not found in plugin '{plugin_name}' — plugin skill storage is not configured."
        )

    async def skill_exists(self, *, user_id: str, plugin_name: str, skill_name: str) -> bool:
        return False

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
    ) -> MongoPluginResourceDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def delete_resource(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def read_resource(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> str:
        raise SkillNotFoundError(f"Resource '{resource_name}' not found — plugin skill storage is not configured.")

    async def list_resource_names(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> Sequence[str]:
        return ()

    async def resource_exists(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        return False

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
    ) -> MongoPluginScriptDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def delete_script(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def read_script(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> str:
        raise SkillNotFoundError(f"Script '{script_name}' not found — plugin skill storage is not configured.")

    async def list_script_names(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
    ) -> Sequence[str]:
        return ()

    async def script_exists(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool:
        return False

    # --- Usage tracking ---

    async def record_skill_usage(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> MongoPluginSkillUsageDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def get_skill_usage_count(self, *, skill_name: str) -> int:
        return 0

    async def get_skill_usage_counts(self, *, skill_names: Sequence[str]) -> Mapping[str, int]:
        return {name: 0 for name in skill_names}

    # --- Plugin catalog ---

    async def save_plugin(
        self,
        *,
        plugin_name: str,
        description: str,
        skills: Sequence[str],
        mcp_servers: Sequence[dict[str, object]],
    ) -> MongoPluginDefinitionDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]:
        return ()
