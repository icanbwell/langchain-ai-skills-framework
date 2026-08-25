"""Writes and queries history records in MongoDB *_history collections."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from langchain_ai_skills_framework.models.history_record import HistoryRecord

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_HISTORY_COLLECTION = "plugin_skills_history"
DEFAULT_SCRIPTS_HISTORY_COLLECTION = "plugin_scripts_history"
DEFAULT_REFERENCES_HISTORY_COLLECTION = "plugin_references_history"
DEFAULT_PLUGINS_HISTORY_COLLECTION = "plugins_history"


class HistoryWriter:
    """Handles writing and querying history records across four history collections."""

    SKILLS_HISTORY_INDEX = "ix_skill_history_lookup"
    SCRIPTS_HISTORY_INDEX = "ix_script_history_lookup"
    REFERENCES_HISTORY_INDEX = "ix_reference_history_lookup"
    PLUGINS_HISTORY_INDEX = "ix_plugin_history_lookup"

    def __init__(
        self,
        *,
        database: AsyncIOMotorDatabase[dict[str, object]],
        skills_history_collection_name: str = DEFAULT_SKILLS_HISTORY_COLLECTION,
        scripts_history_collection_name: str = DEFAULT_SCRIPTS_HISTORY_COLLECTION,
        references_history_collection_name: str = DEFAULT_REFERENCES_HISTORY_COLLECTION,
        plugins_history_collection_name: str = DEFAULT_PLUGINS_HISTORY_COLLECTION,
    ) -> None:
        self._skills_history: AsyncIOMotorCollection[dict[str, object]] = database[skills_history_collection_name]
        self._scripts_history: AsyncIOMotorCollection[dict[str, object]] = database[scripts_history_collection_name]
        self._references_history: AsyncIOMotorCollection[dict[str, object]] = database[
            references_history_collection_name
        ]
        self._plugins_history: AsyncIOMotorCollection[dict[str, object]] = database[plugins_history_collection_name]

    async def ensure_indexes(self) -> None:
        """Create indexes on history collections for efficient querying."""
        await self._skills_history.create_index(
            [("user_id", 1), ("plugin_name", 1), ("skill_name", 1), ("timestamp", -1)],
            name=self.SKILLS_HISTORY_INDEX,
        )
        await self._scripts_history.create_index(
            [("user_id", 1), ("plugin_name", 1), ("skill_name", 1), ("script_name", 1), ("timestamp", -1)],
            name=self.SCRIPTS_HISTORY_INDEX,
        )
        await self._references_history.create_index(
            [("user_id", 1), ("plugin_name", 1), ("skill_name", 1), ("resource_name", 1), ("timestamp", -1)],
            name=self.REFERENCES_HISTORY_INDEX,
        )
        await self._plugins_history.create_index(
            [("plugin_name", 1), ("timestamp", -1)],
            name=self.PLUGINS_HISTORY_INDEX,
        )

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    async def write_skill_history(self, *, record: HistoryRecord) -> None:
        await self._write(collection=self._skills_history, record=record, entity_type="skill")

    async def write_script_history(self, *, record: HistoryRecord) -> None:
        await self._write(collection=self._scripts_history, record=record, entity_type="script")

    async def write_reference_history(self, *, record: HistoryRecord) -> None:
        await self._write(collection=self._references_history, record=record, entity_type="reference")

    async def write_plugin_history(self, *, record: HistoryRecord) -> None:
        await self._write(collection=self._plugins_history, record=record, entity_type="plugin")

    # ------------------------------------------------------------------
    # Query methods
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
        return await self._query(
            collection=self._skills_history,
            query={"user_id": user_id, "plugin_name": plugin_name, "skill_name": skill_name},
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
        return await self._query(
            collection=self._scripts_history,
            query={
                "user_id": user_id,
                "plugin_name": plugin_name,
                "skill_name": skill_name,
                "script_name": script_name,
            },
            limit=limit,
            offset=offset,
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
        return await self._query(
            collection=self._references_history,
            query={
                "user_id": user_id,
                "plugin_name": plugin_name,
                "skill_name": skill_name,
                "resource_name": resource_name,
            },
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
        return await self._query(
            collection=self._plugins_history,
            query={"plugin_name": plugin_name},
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _write(
        *,
        collection: AsyncIOMotorCollection[dict[str, object]],
        record: HistoryRecord,
        entity_type: str,
    ) -> None:
        try:
            await collection.insert_one(record.to_mongo_dict())
        except Exception:
            logger.warning(
                "Failed to write %s history for %s/%s (action=%s)",
                entity_type,
                record.plugin_name,
                record.skill_name or record.plugin_name,
                record.action,
                exc_info=True,
            )

    @staticmethod
    async def _query(
        *,
        collection: AsyncIOMotorCollection[dict[str, object]],
        query: dict[str, object],
        limit: int,
        offset: int,
    ) -> Sequence[HistoryRecord]:
        cursor = collection.find(query).sort("timestamp", -1).skip(offset).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [HistoryRecord.from_mongo_dict(doc) for doc in docs]
