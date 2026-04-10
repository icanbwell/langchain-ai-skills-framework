from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import yaml
from motor.motor_asyncio import AsyncIOMotorCollection

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MongoUserSkillLoader:
    """Reads and writes per-user skills in MongoDB.

    This is a **singleton** service — the ``user_id`` is provided on each
    call rather than at construction time, matching the gateway's pattern
    where tools receive the user identity as a tool-input parameter.
    """

    COLLECTION_NAME = "user_skills"
    INDEX_NAME = "ux_user_skill"

    def __init__(
        self,
        *,
        collection: AsyncIOMotorCollection,  # type: ignore[type-arg]
    ) -> None:
        if collection is None:
            raise ValueError("collection must not be None")
        self._collection = collection

    # --- Index management ---------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create the compound unique index if it does not already exist."""
        await self._collection.create_index(
            [("user_id", 1), ("skill_name", 1)],
            unique=True,
            name=self.INDEX_NAME,
        )

    # --- Write operations ----------------------------------------------------

    async def save_skill(
        self, *, user_id: str, skill_name: str, content: str
    ) -> MongoSkillDocument:
        """Upsert a skill for the given user.

        The description is extracted from YAML frontmatter in *content*.
        If no frontmatter is present the first non-empty line is used.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_name = self._normalize_skill_name(skill_name)
        if not normalized_name:
            raise ValueError("skill_name must be a non-empty string")

        description = self._extract_description(content)
        now = datetime.now(timezone.utc)

        await self._collection.update_one(
            {"user_id": user_id, "skill_name": normalized_name},
            {
                "$set": {
                    "content": content,
                    "description": description,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "skill_name": normalized_name,
                    "created_at": now,
                },
            },
            upsert=True,
        )

        return MongoSkillDocument(
            user_id=user_id,
            skill_name=normalized_name,
            description=description,
            content=content,
            created_at=now,
            updated_at=now,
        )

    async def delete_skill(self, *, user_id: str, skill_name: str) -> bool:
        """Delete a skill for the given user.  Returns True if a document was removed."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_name = self._normalize_skill_name(skill_name)
        result = await self._collection.delete_one(
            {"user_id": user_id, "skill_name": normalized_name}
        )
        return result.deleted_count > 0

    # --- Read operations -----------------------------------------------------

    async def load_snapshot(self, *, user_id: str) -> SkillSnapshot:
        """Load all skills for a user and return an immutable snapshot."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        details_map: dict[str, SkillDetails] = {}
        summaries: list[SkillSummary] = []

        async for raw in self._collection.find({"user_id": user_id}):
            doc = MongoSkillDocument.from_mongo_dict(raw)
            summary = SkillSummary(
                name=doc.skill_name,
                description=doc.description,
                source_path=Path(f"mongodb://{user_id}/{doc.skill_name}"),
                license=None,
                compatibility=None,
                metadata={"source": "mongodb", "user_id": user_id},
                allowed_tools=(),
            )
            detail = SkillDetails(
                summary=summary,
                content=doc.content,
                source_path=summary.source_path,
            )
            details_map[doc.skill_name] = detail
            summaries.append(summary)

        ordered = tuple(sorted(summaries, key=lambda s: s.name))
        return SkillSnapshot(
            details_by_name=MappingProxyType(details_map),
            ordered_summaries=ordered,
        )

    async def get_skill_details(self, *, user_id: str, skill_name: str) -> SkillDetails:
        """Load a single skill for a user."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_name = self._normalize_skill_name(skill_name)
        raw = await self._collection.find_one(
            {"user_id": user_id, "skill_name": normalized_name}
        )
        if raw is None:
            raise SkillNotFoundError(
                f"Skill '{skill_name}' not found for user '{user_id}'"
            )
        doc = MongoSkillDocument.from_mongo_dict(raw)
        summary = SkillSummary(
            name=doc.skill_name,
            description=doc.description,
            source_path=Path(f"mongodb://{user_id}/{doc.skill_name}"),
            license=None,
            compatibility=None,
            metadata={"source": "mongodb", "user_id": user_id},
            allowed_tools=(),
        )
        return SkillDetails(
            summary=summary,
            content=doc.content,
            source_path=summary.source_path,
        )

    # --- Helpers -------------------------------------------------------------

    @staticmethod
    def _normalize_skill_name(value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        normalized = re.sub(r"-+", "-", normalized)
        return normalized.strip("-")

    @staticmethod
    def _extract_description(content: str) -> str:
        """Extract description from YAML frontmatter, falling back to first line."""
        match = _FRONTMATTER_RE.match(content)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1))
                if isinstance(frontmatter, dict):
                    desc = frontmatter.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        return desc.strip()
            except yaml.YAMLError:
                pass

        for line in content.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:200]

        return "User-saved skill"
