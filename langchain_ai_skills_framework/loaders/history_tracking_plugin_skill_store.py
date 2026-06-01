"""PluginSkillStore decorator that records mutation history."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.models.history_record import HistoryRecord
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
from langchain_ai_skills_framework.persistence.history_writer import HistoryWriter

logger = logging.getLogger(__name__)


class HistoryTrackingPluginSkillStore:
    """Wraps a PluginSkillStore, recording mutation history via HistoryWriter."""

    def __init__(self, *, inner_store: PluginSkillStore, history_writer: HistoryWriter) -> None:
        self._inner = inner_store
        self._history = history_writer

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        await self._inner.ensure_indexes()
        await self._history.ensure_indexes()

    # ------------------------------------------------------------------
    # Skill mutations
    # ------------------------------------------------------------------

    async def save_skill(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
    ) -> MongoPluginSkillDocument:
        exists = await self._inner.skill_exists(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)
        action: Literal["created", "updated"] = "updated" if exists else "created"

        doc = await self._inner.save_skill(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            content=content,
            modified_by=modified_by,
        )

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=modified_by or user_id,
            timestamp=datetime.now(timezone.utc),
            source_collection="plugin_skills",
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        await self._history.write_skill_history(record=record)
        return doc

    async def set_skill_published(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        published: bool,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument:
        doc = await self._inner.set_skill_published(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            published=published,
            published_branch=published_branch,
        )

        action: Literal["published", "unpublished"] = "published" if published else "unpublished"
        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=user_id,
            timestamp=datetime.now(timezone.utc),
            source_collection="plugin_skills",
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
        )
        await self._history.write_skill_history(record=record)
        return doc

    async def delete_skill(self, *, user_id: str, plugin_name: str, skill_name: str) -> bool:
        snapshot: dict[str, object] = {}
        try:
            details = await self._inner.get_skill_details(
                user_id=user_id, plugin_name=plugin_name, skill_name=skill_name
            )
            snapshot = {
                "skill_name": details.name,
                "description": details.description,
                "content": details.content,
                "plugin_name": plugin_name,
                "user_id": user_id,
            }
        except Exception:
            logger.debug("Could not capture pre-delete snapshot for skill %s/%s", plugin_name, skill_name)

        deleted = await self._inner.delete_skill(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)

        if deleted:
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=user_id,
                timestamp=datetime.now(timezone.utc),
                source_collection="plugin_skills",
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            await self._history.write_skill_history(record=record)

        return deleted

    # ------------------------------------------------------------------
    # Resource mutations
    # ------------------------------------------------------------------

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
        exists = await self._inner.resource_exists(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )
        action: Literal["created", "updated"] = "updated" if exists else "created"

        doc = await self._inner.save_resource(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            resource_name=resource_name,
            content=content,
            modified_by=modified_by,
        )

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=modified_by or user_id,
            timestamp=datetime.now(timezone.utc),
            source_collection="plugin_references",
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            resource_name=resource_name,
        )
        await self._history.write_reference_history(record=record)
        return doc

    async def delete_resource(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        snapshot: dict[str, object] = {}
        try:
            content = await self._inner.read_resource(
                user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
            )
            snapshot = {
                "resource_name": resource_name,
                "content": content,
                "skill_name": skill_name,
                "plugin_name": plugin_name,
                "user_id": user_id,
            }
        except Exception:
            logger.debug(
                "Could not capture pre-delete snapshot for resource %s/%s/%s",
                plugin_name,
                skill_name,
                resource_name,
            )

        deleted = await self._inner.delete_resource(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )

        if deleted:
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=user_id,
                timestamp=datetime.now(timezone.utc),
                source_collection="plugin_references",
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=resource_name,
            )
            await self._history.write_reference_history(record=record)

        return deleted

    # ------------------------------------------------------------------
    # Script mutations
    # ------------------------------------------------------------------

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
        exists = await self._inner.script_exists(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )
        action: Literal["created", "updated"] = "updated" if exists else "created"

        doc = await self._inner.save_script(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            script_name=script_name,
            content=content,
            modified_by=modified_by,
        )

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=modified_by or user_id,
            timestamp=datetime.now(timezone.utc),
            source_collection="plugin_scripts",
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            script_name=script_name,
        )
        await self._history.write_script_history(record=record)
        return doc

    async def delete_script(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool:
        snapshot: dict[str, object] = {}
        try:
            content = await self._inner.read_script(
                user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
            )
            snapshot = {
                "script_name": script_name,
                "content": content,
                "skill_name": skill_name,
                "plugin_name": plugin_name,
                "user_id": user_id,
            }
        except Exception:
            logger.debug(
                "Could not capture pre-delete snapshot for script %s/%s/%s",
                plugin_name,
                skill_name,
                script_name,
            )

        deleted = await self._inner.delete_script(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )

        if deleted:
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=user_id,
                timestamp=datetime.now(timezone.utc),
                source_collection="plugin_scripts",
                user_id=user_id,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
            )
            await self._history.write_script_history(record=record)

        return deleted

    # ------------------------------------------------------------------
    # Plugin catalog mutation
    # ------------------------------------------------------------------

    async def save_plugin(
        self,
        *,
        plugin_name: str,
        description: str,
        skills: Sequence[str],
        mcp_servers: Sequence[dict[str, object]],
    ) -> MongoPluginDefinitionDocument:
        doc = await self._inner.save_plugin(
            plugin_name=plugin_name,
            description=description,
            skills=skills,
            mcp_servers=mcp_servers,
        )

        record = HistoryRecord(
            action="updated",
            document_snapshot=doc.to_mongo_dict(),
            changed_by="system",
            timestamp=datetime.now(timezone.utc),
            source_collection="plugins",
            user_id="system",
            plugin_name=plugin_name,
        )
        await self._history.write_plugin_history(record=record)
        return doc

    # ------------------------------------------------------------------
    # Read-only pass-through methods
    # ------------------------------------------------------------------

    async def load_snapshot(self, *, user_id: str, plugin_name: str | None = None) -> SkillSnapshot:
        return await self._inner.load_snapshot(user_id=user_id, plugin_name=plugin_name)

    async def load_shared_snapshot(self, *, plugin_name: str | None = None) -> SkillSnapshot:
        return await self._inner.load_shared_snapshot(plugin_name=plugin_name)

    async def get_skill_details(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> SkillDetails:
        return await self._inner.get_skill_details(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)

    async def skill_exists(self, *, user_id: str, plugin_name: str | None = None, skill_name: str) -> bool:
        return await self._inner.skill_exists(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)

    async def read_resource(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> str:
        return await self._inner.read_resource(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )

    async def list_resource_names(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        return await self._inner.list_resource_names(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)

    async def resource_exists(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        return await self._inner.resource_exists(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )

    async def read_script(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> str:
        return await self._inner.read_script(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )

    async def list_script_names(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        return await self._inner.list_script_names(user_id=user_id, plugin_name=plugin_name, skill_name=skill_name)

    async def script_exists(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> bool:
        return await self._inner.script_exists(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )

    async def record_skill_usage(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> MongoPluginSkillUsageDocument:
        return await self._inner.record_skill_usage(plugin_name=plugin_name, skill_name=skill_name, user_id=user_id)

    async def get_skill_usage_count(self, *, skill_name: str) -> int:
        return await self._inner.get_skill_usage_count(skill_name=skill_name)

    async def get_skill_usage_counts(self, *, skill_names: Sequence[str]) -> Mapping[str, int]:
        return await self._inner.get_skill_usage_counts(skill_names=skill_names)

    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]:
        return await self._inner.list_plugins()

    async def has_plugins(self) -> bool:
        return await self._inner.has_plugins()

    # ------------------------------------------------------------------
    # History query pass-through
    # ------------------------------------------------------------------

    async def get_skill_history(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_skill_history(
            user_id=user_id, plugin_name=plugin_name, skill_name=skill_name, limit=limit, offset=offset
        )

    async def get_resource_history(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_resource_history(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            resource_name=resource_name,
            limit=limit,
            offset=offset,
        )

    async def get_script_history(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_script_history(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=skill_name,
            script_name=script_name,
            limit=limit,
            offset=offset,
        )

    async def get_plugin_history(
        self,
        *,
        plugin_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_plugin_history(plugin_name=plugin_name, limit=limit, offset=offset)
