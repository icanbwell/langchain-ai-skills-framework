from __future__ import annotations

from types import MappingProxyType

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
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
    "User skill storage is not configured. "
    "Set MONGO_SKILLS_URI or MONGO_URL to enable user skills."
)


class NullUserSkillStore:
    """No-op ``UserSkillStore`` used when MongoDB is not configured.

    Reads return empty results.  Writes raise ``RuntimeError`` with a
    message directing the deployer to set the required env vars.
    """

    async def ensure_indexes(self) -> None:
        return

    async def save_skill(
        self, *, user_id: str, skill_name: str, content: str
    ) -> MongoSkillDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def delete_skill(self, *, user_id: str, skill_name: str) -> bool:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def set_skill_shared(
        self, *, user_id: str, skill_name: str, shared: bool
    ) -> MongoSkillDocument:
        raise RuntimeError(_NOT_CONFIGURED_MSG)

    async def load_snapshot(self, *, user_id: str) -> SkillSnapshot:
        return _EMPTY_SNAPSHOT

    async def load_shared_snapshot(self) -> SkillSnapshot:
        return _EMPTY_SNAPSHOT

    async def get_skill_details(self, *, user_id: str, skill_name: str) -> SkillDetails:
        raise SkillNotFoundError(
            f"Skill '{skill_name}' not found — user skill storage is not configured."
        )
