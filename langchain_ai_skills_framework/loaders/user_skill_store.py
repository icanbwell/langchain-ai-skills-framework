from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
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

    async def save_skill(
        self, *, user_id: str, skill_name: str, content: str
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
