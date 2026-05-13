from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import TracebackType

from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)


class MongoAdvisoryLock:
    """Distributed advisory lock backed by a MongoDB collection.

    Uses find_one_and_update with an expiry field so that stale locks
    are automatically released if the holder crashes.

    Usage:
        async with MongoAdvisoryLock(db, "skill_sync", ttl_seconds=300) as acquired:
            if acquired:
                await do_sync()
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase[dict[str, object]],
        lock_name: str,
        *,
        ttl_seconds: int = 300,
        collection_name: str = "advisory_locks",
    ) -> None:
        self._collection = db[collection_name]
        self._lock_name = lock_name
        self._ttl_seconds = ttl_seconds
        self._holder_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    async def __aenter__(self) -> bool:
        now = datetime.now(timezone.utc)
        result = await self._collection.find_one_and_update(
            {
                "_id": self._lock_name,
                "$or": [
                    {"expires_at": {"$lt": now}},
                    {"expires_at": {"$exists": False}},
                ],
            },
            {
                "$set": {
                    "holder": self._holder_id,
                    "expires_at": now + timedelta(seconds=self._ttl_seconds),
                    "acquired_at": now,
                }
            },
            upsert=True,
            return_document=True,
        )

        self._acquired = result is not None and result.get("holder") == self._holder_id
        if self._acquired:
            logger.info(
                "MongoAdvisoryLock: acquired lock '%s' (holder=%s, ttl=%ds)",
                self._lock_name,
                self._holder_id,
                self._ttl_seconds,
            )
        else:
            logger.info(
                "MongoAdvisoryLock: lock '%s' held by another process — skipping.",
                self._lock_name,
            )
        return self._acquired

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._acquired:
            await self._collection.delete_one({"_id": self._lock_name, "holder": self._holder_id})
            logger.info(
                "MongoAdvisoryLock: released lock '%s' (holder=%s)",
                self._lock_name,
                self._holder_id,
            )
