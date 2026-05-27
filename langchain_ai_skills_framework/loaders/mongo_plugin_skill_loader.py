"""MongoDB-backed implementation of :class:`PluginSkillStore`.

Uses three collections (``plugin_skills``, ``plugin_references``,
``plugin_scripts``) with the **Materialized Paths** tree-structure pattern.
Every document carries a ``path`` field that mirrors the on-disk plugin
directory layout, enabling tree-style queries.

Replaces the legacy ``MongoUserSkillLoader``.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ReturnDocument

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
    MongoPluginSkillUsageDocument,
    build_resource_path,
    build_script_path,
    build_skill_path,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Default collection names — overridable via environment variables.
DEFAULT_SKILLS_COLLECTION = "plugin_skills"
DEFAULT_REFERENCES_COLLECTION = "plugin_references"
DEFAULT_SCRIPTS_COLLECTION = "plugin_scripts"
DEFAULT_USAGE_COLLECTION = "plugin_skill_usage"
DEFAULT_PLUGINS_COLLECTION = "plugins"


class MongoPluginSkillLoader:
    """Reads and writes plugin-scoped skills in MongoDB.

    This is a **singleton** service — ``user_id`` and ``plugin_name`` are
    provided on each call, matching the gateway pattern where tools
    receive identity as a tool-input parameter.
    """

    INDEX_NAME = "ux_plugin_skill"
    RESOURCE_INDEX_NAME = "ux_plugin_skill_resource"
    SCRIPT_INDEX_NAME = "ux_plugin_skill_script"
    PATH_INDEX_NAME = "ix_plugin_path"
    USAGE_INDEX_NAME = "ix_plugin_skill_usage_lookup"
    PLUGIN_INDEX_NAME = "ux_plugin_name"

    SCHEMA_VERSION_FIELD = "schema_version"

    def __init__(
        self,
        *,
        database: AsyncIOMotorDatabase[dict[str, object]],
        schema_version: int = 1,
        skills_collection_name: str = DEFAULT_SKILLS_COLLECTION,
        references_collection_name: str = DEFAULT_REFERENCES_COLLECTION,
        scripts_collection_name: str = DEFAULT_SCRIPTS_COLLECTION,
        usage_collection_name: str = DEFAULT_USAGE_COLLECTION,
        plugins_collection_name: str = DEFAULT_PLUGINS_COLLECTION,
    ) -> None:
        self._database = database
        self._schema_version = schema_version
        self._skills_collection: AsyncIOMotorCollection[dict[str, object]] = database[skills_collection_name]
        self._resources_collection: AsyncIOMotorCollection[dict[str, object]] = database[references_collection_name]
        self._scripts_collection: AsyncIOMotorCollection[dict[str, object]] = database[scripts_collection_name]
        self._usage_collection: AsyncIOMotorCollection[dict[str, object]] = database[usage_collection_name]
        self._plugins_collection: AsyncIOMotorCollection[dict[str, object]] = database[plugins_collection_name]

    # --- Index management ---------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create compound unique indexes and Materialized Paths index."""
        sv = self.SCHEMA_VERSION_FIELD
        await self._skills_collection.create_index(
            [(sv, 1), ("user_id", 1), ("plugin_name", 1), ("skill_name", 1)],
            unique=True,
            name=self.INDEX_NAME,
        )
        await self._resources_collection.create_index(
            [(sv, 1), ("user_id", 1), ("plugin_name", 1), ("skill_name", 1), ("resource_name", 1)],
            unique=True,
            name=self.RESOURCE_INDEX_NAME,
        )
        await self._scripts_collection.create_index(
            [(sv, 1), ("user_id", 1), ("plugin_name", 1), ("skill_name", 1), ("script_name", 1)],
            unique=True,
            name=self.SCRIPT_INDEX_NAME,
        )
        await self._skills_collection.create_index(
            [(sv, 1), ("plugin_name", 1), ("path", 1)],
            name=self.PATH_INDEX_NAME,
        )
        await self._usage_collection.create_index(
            [("skill_name", 1), ("user_id", 1), ("date_used", -1)],
            name=self.USAGE_INDEX_NAME,
        )
        await self._plugins_collection.create_index(
            [(sv, 1), ("plugin_name", 1)],
            unique=True,
            name=self.PLUGIN_INDEX_NAME,
        )

    def _version_filter(self, query: dict[str, object]) -> dict[str, object]:
        """Add schema_version to a query filter."""
        return {**query, self.SCHEMA_VERSION_FIELD: self._schema_version}

    # --- Skill write operations ----------------------------------------------

    async def save_skill(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoPluginSkillDocument:
        self._validate_user_id(user_id)
        normalized_name = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")

        description = self._extract_description(content)
        path = build_skill_path(plugin_name, normalized_name)
        now = datetime.now(timezone.utc)
        effective_modified_by = modified_by or user_id
        sv = self.SCHEMA_VERSION_FIELD

        raw = await self._skills_collection.find_one_and_update(
            self._version_filter({"user_id": user_id, "plugin_name": plugin_name, "skill_name": normalized_name}),
            {
                "$set": {
                    "content": content,
                    "description": description,
                    "path": path,
                    "modified_by": effective_modified_by,
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_name,
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        return MongoPluginSkillDocument.from_mongo_dict(raw)

    async def set_skill_published(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        published: bool,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument:
        self._validate_user_id(user_id)
        normalized_name = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")

        now = datetime.now(timezone.utc)
        update_fields: dict[str, object] = {
            "published": published,
            "published_date": now,
            "date_modified": now,
        }
        if published_branch is not None:
            update_fields["published_branch"] = published_branch
        raw = await self._skills_collection.find_one_and_update(
            self._version_filter({"user_id": user_id, "plugin_name": plugin_name, "skill_name": normalized_name}),
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        if raw is None:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in plugin '{plugin_name}' for user '{user_id}'")
        return MongoPluginSkillDocument.from_mongo_dict(raw)

    async def delete_skill(self, *, user_id: str, plugin_name: str, skill_name: str) -> bool:
        self._validate_user_id(user_id)
        normalized_name = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")

        filter_base = self._version_filter(
            {"user_id": user_id, "plugin_name": plugin_name, "skill_name": normalized_name}
        )
        await self._resources_collection.delete_many(filter_base)
        await self._scripts_collection.delete_many(filter_base)

        result = await self._skills_collection.delete_one(filter_base)
        return result.deleted_count > 0

    async def skill_exists(self, *, user_id: str, plugin_name: str | None = None, skill_name: str) -> bool:
        self._validate_user_id(user_id)
        normalized_name = self._normalize(skill_name)
        query: dict[str, object] = {"user_id": user_id, "skill_name": normalized_name}
        if plugin_name:
            query["plugin_name"] = plugin_name
        count = await self._skills_collection.count_documents(self._version_filter(query), limit=1)
        return count > 0

    # --- Resource write operations -------------------------------------------

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
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")
        self._validate_not_empty(resource_name.strip(), "resource_name")

        path = build_resource_path(plugin_name, normalized_skill, resource_name.strip())
        now = datetime.now(timezone.utc)
        effective_modified_by = modified_by or user_id
        sv = self.SCHEMA_VERSION_FIELD

        raw = await self._resources_collection.find_one_and_update(
            self._version_filter(
                {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                }
            ),
            {
                "$set": {
                    "content": content,
                    "path": path,
                    "modified_by": effective_modified_by,
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MongoPluginResourceDocument.from_mongo_dict(raw)

    async def delete_resource(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        result = await self._resources_collection.delete_one(
            self._version_filter(
                {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                }
            )
        )
        return result.deleted_count > 0

    async def read_resource(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> str:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "user_id": user_id,
            "skill_name": normalized_skill,
            "resource_name": resource_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        raw = await self._resources_collection.find_one(self._version_filter(query))
        if raw is None:
            raise SkillNotFoundError(
                f"Resource '{resource_name}' not found in skill '{skill_name}' "
                f"of plugin '{plugin_name}' for user '{user_id}'"
            )
        return str(raw["content"])

    async def list_resource_names(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {"user_id": user_id, "skill_name": normalized_skill}
        if plugin_name:
            query["plugin_name"] = plugin_name
        names: list[str] = []
        async for raw in self._resources_collection.find(self._version_filter(query), {"resource_name": 1}):
            names.append(raw["resource_name"])
        return sorted(names)

    async def resource_exists(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "user_id": user_id,
            "skill_name": normalized_skill,
            "resource_name": resource_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        count = await self._resources_collection.count_documents(self._version_filter(query), limit=1)
        return count > 0

    # --- Script write operations ---------------------------------------------

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
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")
        self._validate_not_empty(script_name.strip(), "script_name")

        path = build_script_path(plugin_name, normalized_skill, script_name.strip())
        now = datetime.now(timezone.utc)
        effective_modified_by = modified_by or user_id
        sv = self.SCHEMA_VERSION_FIELD

        raw = await self._scripts_collection.find_one_and_update(
            self._version_filter(
                {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                }
            ),
            {
                "$set": {
                    "content": content,
                    "path": path,
                    "modified_by": effective_modified_by,
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MongoPluginScriptDocument.from_mongo_dict(raw)

    async def delete_script(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        result = await self._scripts_collection.delete_one(
            self._version_filter(
                {
                    "user_id": user_id,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                }
            )
        )
        return result.deleted_count > 0

    async def read_script(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> str:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "user_id": user_id,
            "skill_name": normalized_skill,
            "script_name": script_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        raw = await self._scripts_collection.find_one(self._version_filter(query))
        if raw is None:
            raise SkillNotFoundError(
                f"Script '{script_name}' not found in skill '{skill_name}' "
                f"of plugin '{plugin_name}' for user '{user_id}'"
            )
        return str(raw["content"])

    async def list_script_names(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {"user_id": user_id, "skill_name": normalized_skill}
        if plugin_name:
            query["plugin_name"] = plugin_name
        names: list[str] = []
        async for raw in self._scripts_collection.find(self._version_filter(query), {"script_name": 1}):
            names.append(raw["script_name"])
        return sorted(names)

    async def script_exists(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> bool:
        self._validate_user_id(user_id)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "user_id": user_id,
            "skill_name": normalized_skill,
            "script_name": script_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        count = await self._scripts_collection.count_documents(self._version_filter(query), limit=1)
        return count > 0

    # --- Skill read operations -----------------------------------------------

    async def load_snapshot(self, *, user_id: str, plugin_name: str | None = None) -> SkillSnapshot:
        self._validate_user_id(user_id)
        query: dict[str, object] = {"user_id": user_id}
        if plugin_name:
            query["plugin_name"] = plugin_name
        return await self._build_snapshot(query=self._version_filter(query), owner_label=user_id)

    async def load_shared_snapshot(self, *, plugin_name: str | None = None) -> SkillSnapshot:
        query: dict[str, object] = {"$or": [{"published": True}, {"shared": True}]}
        if plugin_name:
            query["plugin_name"] = plugin_name
        return await self._build_snapshot(query=self._version_filter(query), owner_label="shared")

    async def get_skill_details(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> SkillDetails:
        self._validate_user_id(user_id)
        normalized_name = self._normalize(skill_name)
        query: dict[str, object] = {"user_id": user_id, "skill_name": normalized_name}
        if plugin_name:
            query["plugin_name"] = plugin_name
        raw = await self._skills_collection.find_one(self._version_filter(query))
        if raw is None:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in plugin '{plugin_name}' for user '{user_id}'")
        doc = MongoPluginSkillDocument.from_mongo_dict(raw)
        summary = SkillSummary(
            name=doc.skill_name,
            description=doc.description,
            plugin_name=doc.plugin_name,
            source_path=Path(f"mongodb://{user_id}/{doc.plugin_name}/{doc.skill_name}"),
            license=None,
            compatibility=None,
            metadata={"source": "mongodb", "user_id": doc.user_id, "plugin_name": doc.plugin_name},
            allowed_tools=doc.allowed_tools,
        )
        return SkillDetails(
            summary=summary,
            content=doc.content,
            source_path=summary.source_path,
        )

    # --- Usage tracking -------------------------------------------------------

    async def record_skill_usage(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> MongoPluginSkillUsageDocument:
        doc = MongoPluginSkillUsageDocument(
            plugin_name=plugin_name,
            skill_name=skill_name,
            user_id=user_id,
        )
        data = doc.to_mongo_dict()
        data[self.SCHEMA_VERSION_FIELD] = self._schema_version
        await self._usage_collection.insert_one(data)
        return doc

    async def get_skill_usage_count(self, *, skill_name: str) -> int:
        return int(await self._usage_collection.count_documents(self._version_filter({"skill_name": skill_name})))

    async def get_skill_usage_counts(self, *, skill_names: Sequence[str]) -> Mapping[str, int]:
        if not skill_names:
            return {}
        pipeline: list[dict[str, Any]] = [
            {"$match": {self.SCHEMA_VERSION_FIELD: self._schema_version, "skill_name": {"$in": list(skill_names)}}},
            {"$group": {"_id": "$skill_name", "count": {"$sum": 1}}},
        ]
        counts: dict[str, int] = {name: 0 for name in skill_names}
        async for doc in self._usage_collection.aggregate(pipeline):
            counts[doc["_id"]] = int(doc["count"])
        return counts

    # --- Snapshot builder ----------------------------------------------------

    async def _build_snapshot(self, *, query: dict[str, object], owner_label: str) -> SkillSnapshot:
        details_map: dict[str, SkillDetails] = {}
        summaries: list[SkillSummary] = []

        async for raw in self._skills_collection.find(query):
            doc = MongoPluginSkillDocument.from_mongo_dict(raw)
            summary = SkillSummary(
                name=doc.skill_name,
                description=doc.description,
                plugin_name=doc.plugin_name,
                source_path=Path(f"mongodb://{owner_label}/{doc.plugin_name}/{doc.skill_name}"),
                license=None,
                compatibility=None,
                metadata={"source": "mongodb", "user_id": doc.user_id, "plugin_name": doc.plugin_name},
                allowed_tools=doc.allowed_tools,
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

    # --- Helpers -------------------------------------------------------------

    @staticmethod
    def _normalize(value: str) -> str:
        return normalize_skill_name(value)

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")

    @staticmethod
    def _validate_not_empty(value: str, field_name: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    @staticmethod
    def _extract_description(content: str) -> str:
        match = _FRONTMATTER_RE.match(content)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1))
                if isinstance(frontmatter, dict):
                    desc = frontmatter.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        return desc.strip()
            except yaml.YAMLError as e:
                logger.debug("Failed to parse YAML frontmatter in skill content: %s", e)

        for line in content.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:200]

        return "Plugin skill"

    # --- Plugin catalog -------------------------------------------------------

    async def save_plugin(
        self,
        *,
        plugin_name: str,
        description: str,
        skills: Sequence[str],
        mcp_servers: Sequence[dict[str, object]],
    ) -> MongoPluginDefinitionDocument:
        """Upsert a plugin definition document."""
        now = datetime.now(timezone.utc)
        sv = self.SCHEMA_VERSION_FIELD
        logger.info(
            "save_plugin: upserting plugin '%s' to collection '%s'",
            plugin_name,
            self._plugins_collection.name,
        )
        raw = await self._plugins_collection.find_one_and_update(
            self._version_filter({"plugin_name": plugin_name}),
            {
                "$set": {
                    "description": description,
                    "skills": list(skills),
                    "mcp_servers": [dict(s) for s in mcp_servers],
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "plugin_name": plugin_name,
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        assert raw is not None
        return MongoPluginDefinitionDocument.from_mongo_dict(raw)

    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]:
        """Return all plugin definitions for the current schema version, skipping malformed documents."""
        cursor = self._plugins_collection.find(self._version_filter({})).sort("plugin_name", 1)
        results: list[MongoPluginDefinitionDocument] = []
        async for doc in cursor:
            try:
                results.append(MongoPluginDefinitionDocument.from_mongo_dict(doc))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "list_plugins: skipping malformed plugin document _id=%s: %s",
                    doc.get("_id"),
                    exc,
                )
        return results

    async def has_plugins(self) -> bool:
        """Return True if the plugins collection has at least one document for the current schema version."""
        count = await self._plugins_collection.count_documents(self._version_filter({}), limit=1)
        return count > 0
