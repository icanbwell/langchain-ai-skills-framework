from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
    MongoSkillResourceDocument,
    MongoSkillScriptDocument,
    MongoSkillUsageDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
)


@runtime_checkable
class UserSkillStore(Protocol):
    """Abstraction over per-user skill persistence.

    Implementations may be backed by MongoDB, an in-memory dict, or a
    null object that always returns empty results.
    """

    async def ensure_indexes(self) -> None: ...

    # --- Skill operations ---

    async def save_skill(
        self,
        *,
        user_id: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoSkillDocument: ...

    async def set_skill_shared(
        self, *, user_id: str, skill_name: str, shared: bool
    ) -> MongoSkillDocument: ...

    async def delete_skill(self, *, user_id: str, skill_name: str) -> bool: ...

    async def load_snapshot(self, *, user_id: str) -> SkillSnapshot: ...

    async def load_shared_snapshot(self) -> SkillSnapshot: ...

    async def get_skill_details(
        self, *, user_id: str, skill_name: str
    ) -> SkillDetails: ...

    async def skill_exists(self, *, user_id: str, skill_name: str) -> bool: ...

    # --- Resource operations ---

    async def save_resource(
        self,
        *,
        user_id: str,
        skill_name: str,
        resource_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoSkillResourceDocument: ...

    async def delete_resource(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> bool: ...

    async def read_resource(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> str: ...

    async def list_resource_names(
        self, *, user_id: str, skill_name: str
    ) -> Sequence[str]: ...

    async def resource_exists(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> bool: ...

    # --- Script operations ---

    async def save_script(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoSkillScriptDocument: ...

    async def delete_script(
        self, *, user_id: str, skill_name: str, script_name: str
    ) -> bool: ...

    async def read_script(
        self, *, user_id: str, skill_name: str, script_name: str
    ) -> str: ...

    async def list_script_names(
        self, *, user_id: str, skill_name: str
    ) -> Sequence[str]: ...

    async def script_exists(
        self, *, user_id: str, skill_name: str, script_name: str
    ) -> bool: ...

    # --- Usage tracking ---

    async def record_skill_usage(
        self, *, skill_name: str, user_id: str
    ) -> MongoSkillUsageDocument: ...

    async def get_skill_usage_count(self, *, skill_name: str) -> int: ...

    async def get_skill_usage_counts(
        self, *, skill_names: Sequence[str]
    ) -> Mapping[str, int]: ...
