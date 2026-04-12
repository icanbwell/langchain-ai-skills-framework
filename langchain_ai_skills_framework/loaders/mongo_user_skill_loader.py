from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Sequence

import yaml
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ReturnDocument

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.mongo_skill_document import (
    MongoSkillDocument,
    MongoSkillResourceDocument,
    MongoSkillScriptDocument,
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
    RESOURCES_COLLECTION_NAME = "user_skill_resources"
    SCRIPTS_COLLECTION_NAME = "user_skill_scripts"
    INDEX_NAME = "ux_user_skill"
    RESOURCE_INDEX_NAME = "ux_user_skill_resource"
    SCRIPT_INDEX_NAME = "ux_user_skill_script"

    def __init__(
        self,
        *,
        collection: AsyncIOMotorCollection,  # type: ignore[type-arg]
        database: AsyncIOMotorDatabase | None = None,  # type: ignore[type-arg]
    ) -> None:
        if collection is None:
            raise ValueError("collection must not be None")
        self._collection = collection
        self._database = database or collection.database
        self._resources_collection: AsyncIOMotorCollection = self._database[  # type: ignore[type-arg]
            self.RESOURCES_COLLECTION_NAME
        ]
        self._scripts_collection: AsyncIOMotorCollection = self._database[  # type: ignore[type-arg]
            self.SCRIPTS_COLLECTION_NAME
        ]

    # --- Index management ---------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create the compound unique indexes if they do not already exist."""
        await self._collection.create_index(
            [("user_id", 1), ("skill_name", 1)],
            unique=True,
            name=self.INDEX_NAME,
        )
        await self._resources_collection.create_index(
            [("user_id", 1), ("skill_name", 1), ("resource_name", 1)],
            unique=True,
            name=self.RESOURCE_INDEX_NAME,
        )
        await self._scripts_collection.create_index(
            [("user_id", 1), ("skill_name", 1), ("script_name", 1)],
            unique=True,
            name=self.SCRIPT_INDEX_NAME,
        )

    # --- Skill write operations ----------------------------------------------

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

        raw = await self._collection.find_one_and_update(
            {"user_id": user_id, "skill_name": normalized_name},
            {
                "$set": {
                    "content": content,
                    "description": description,
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "skill_name": normalized_name,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        return MongoSkillDocument.from_mongo_dict(raw)

    async def set_skill_shared(
        self, *, user_id: str, skill_name: str, shared: bool
    ) -> MongoSkillDocument:
        """Toggle the shared flag on an existing skill."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_name = self._normalize_skill_name(skill_name)
        if not normalized_name:
            raise ValueError("skill_name must be a non-empty string")

        now = datetime.now(timezone.utc)
        raw = await self._collection.find_one_and_update(
            {"user_id": user_id, "skill_name": normalized_name},
            {"$set": {"shared": shared, "date_modified": now}},
            return_document=ReturnDocument.AFTER,
        )
        if raw is None:
            raise SkillNotFoundError(
                f"Skill '{skill_name}' not found for user '{user_id}'"
            )
        return MongoSkillDocument.from_mongo_dict(raw)

    async def delete_skill(self, *, user_id: str, skill_name: str) -> bool:
        """Delete a skill and its associated resources and scripts.

        Returns True if the skill document was removed.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_name = self._normalize_skill_name(skill_name)

        # Delete associated resources and scripts
        await self._resources_collection.delete_many(
            {"user_id": user_id, "skill_name": normalized_name}
        )
        await self._scripts_collection.delete_many(
            {"user_id": user_id, "skill_name": normalized_name}
        )

        result = await self._collection.delete_one(
            {"user_id": user_id, "skill_name": normalized_name}
        )
        return result.deleted_count > 0

    # --- Resource write operations -------------------------------------------

    async def save_resource(
        self, *, user_id: str, skill_name: str, resource_name: str, content: str
    ) -> MongoSkillResourceDocument:
        """Upsert a resource for a user's skill."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        if not normalized_skill:
            raise ValueError("skill_name must be a non-empty string")
        if not resource_name or not resource_name.strip():
            raise ValueError("resource_name must be a non-empty string")

        now = datetime.now(timezone.utc)
        raw = await self._resources_collection.find_one_and_update(
            {
                "user_id": user_id,
                "skill_name": normalized_skill,
                "resource_name": resource_name.strip(),
            },
            {
                "$set": {
                    "content": content,
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MongoSkillResourceDocument.from_mongo_dict(raw)

    async def delete_resource(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> bool:
        """Delete a resource from a user's skill. Returns True if removed."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        result = await self._resources_collection.delete_one(
            {
                "user_id": user_id,
                "skill_name": normalized_skill,
                "resource_name": resource_name.strip(),
            }
        )
        return result.deleted_count > 0

    async def read_resource(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> str:
        """Read a resource's content. Raises SkillNotFoundError if not found."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        raw = await self._resources_collection.find_one(
            {
                "user_id": user_id,
                "skill_name": normalized_skill,
                "resource_name": resource_name.strip(),
            }
        )
        if raw is None:
            raise SkillNotFoundError(
                f"Resource '{resource_name}' not found in skill '{skill_name}' for user '{user_id}'"
            )
        return raw["content"]

    async def list_resource_names(
        self, *, user_id: str, skill_name: str
    ) -> Sequence[str]:
        """List all resource names for a user's skill."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        names: list[str] = []
        async for raw in self._resources_collection.find(
            {"user_id": user_id, "skill_name": normalized_skill},
            {"resource_name": 1},
        ):
            names.append(raw["resource_name"])
        return sorted(names)

    # --- Script write operations ---------------------------------------------

    async def save_script(
        self, *, user_id: str, skill_name: str, script_name: str, content: str
    ) -> MongoSkillScriptDocument:
        """Upsert a script for a user's skill."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        if not normalized_skill:
            raise ValueError("skill_name must be a non-empty string")
        if not script_name or not script_name.strip():
            raise ValueError("script_name must be a non-empty string")

        now = datetime.now(timezone.utc)
        raw = await self._scripts_collection.find_one_and_update(
            {
                "user_id": user_id,
                "skill_name": normalized_skill,
                "script_name": script_name.strip(),
            },
            {
                "$set": {
                    "content": content,
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MongoSkillScriptDocument.from_mongo_dict(raw)

    async def delete_script(
        self, *, user_id: str, skill_name: str, script_name: str
    ) -> bool:
        """Delete a script from a user's skill. Returns True if removed."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        result = await self._scripts_collection.delete_one(
            {
                "user_id": user_id,
                "skill_name": normalized_skill,
                "script_name": script_name.strip(),
            }
        )
        return result.deleted_count > 0

    async def read_script(
        self, *, user_id: str, skill_name: str, script_name: str
    ) -> str:
        """Read a script's content. Raises SkillNotFoundError if not found."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        raw = await self._scripts_collection.find_one(
            {
                "user_id": user_id,
                "skill_name": normalized_skill,
                "script_name": script_name.strip(),
            }
        )
        if raw is None:
            raise SkillNotFoundError(
                f"Script '{script_name}' not found in skill '{skill_name}' for user '{user_id}'"
            )
        return raw["content"]

    async def list_script_names(
        self, *, user_id: str, skill_name: str
    ) -> Sequence[str]:
        """List all script names for a user's skill."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        normalized_skill = self._normalize_skill_name(skill_name)
        names: list[str] = []
        async for raw in self._scripts_collection.find(
            {"user_id": user_id, "skill_name": normalized_skill},
            {"script_name": 1},
        ):
            names.append(raw["script_name"])
        return sorted(names)

    # --- Skill read operations -----------------------------------------------

    async def load_snapshot(self, *, user_id: str) -> SkillSnapshot:
        """Load all skills for a user and return an immutable snapshot."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        return await self._build_snapshot(
            query={"user_id": user_id}, owner_label=user_id
        )

    async def load_shared_snapshot(self) -> SkillSnapshot:
        """Load all skills marked as shared across all users."""
        return await self._build_snapshot(query={"shared": True}, owner_label="shared")

    async def _build_snapshot(
        self, *, query: dict[str, object], owner_label: str
    ) -> SkillSnapshot:
        details_map: dict[str, SkillDetails] = {}
        summaries: list[SkillSummary] = []

        async for raw in self._collection.find(query):
            doc = MongoSkillDocument.from_mongo_dict(raw)
            summary = SkillSummary(
                name=doc.skill_name,
                description=doc.description,
                source_path=Path(f"mongodb://{owner_label}/{doc.skill_name}"),
                license=None,
                compatibility=None,
                metadata={"source": "mongodb", "user_id": doc.user_id},
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
        normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
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
