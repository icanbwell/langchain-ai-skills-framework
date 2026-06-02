"""Writes and queries error records in MongoDB for troubleshooting."""

from __future__ import annotations

import logging
from typing import Sequence

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from langchain_ai_skills_framework.models.error_record import ErrorRecord

logger = logging.getLogger(__name__)

DEFAULT_ERRORS_COLLECTION = "plugin_errors"


class ErrorWriter:
    """Handles writing and querying error records for troubleshooting skill operations."""

    ERRORS_INDEX = "ix_errors_lookup"
    ERRORS_TIMESTAMP_INDEX = "ix_errors_timestamp"

    def __init__(
        self,
        *,
        database: AsyncIOMotorDatabase[dict[str, object]],
        errors_collection_name: str = DEFAULT_ERRORS_COLLECTION,
    ) -> None:
        self._errors: AsyncIOMotorCollection[dict[str, object]] = database[errors_collection_name]

    async def ensure_indexes(self) -> None:
        """Create indexes on the errors collection for efficient querying."""
        await self._errors.create_index(
            [("user_id", 1), ("plugin_name", 1), ("skill_name", 1), ("timestamp", -1)],
            name=self.ERRORS_INDEX,
        )
        await self._errors.create_index(
            [("timestamp", -1)],
            name=self.ERRORS_TIMESTAMP_INDEX,
        )

    async def write_error(self, *, record: ErrorRecord) -> None:
        try:
            await self._errors.insert_one(record.to_mongo_dict())
        except Exception:
            logger.warning(
                "Failed to write error record for %s/%s (operation=%s)",
                record.plugin_name,
                record.skill_name,
                record.operation,
                exc_info=True,
            )

    async def get_errors(
        self,
        *,
        user_id: str | None = None,
        plugin_name: str | None = None,
        skill_name: str | None = None,
        operation: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ErrorRecord]:
        query: dict[str, object] = {}
        if user_id is not None:
            query["user_id"] = user_id
        if plugin_name is not None:
            query["plugin_name"] = plugin_name
        if skill_name is not None:
            query["skill_name"] = skill_name
        if operation is not None:
            query["operation"] = operation
        cursor = self._errors.find(query).sort("timestamp", -1).skip(offset).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [ErrorRecord.from_mongo_dict(doc) for doc in docs]
