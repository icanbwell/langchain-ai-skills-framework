"""Shared snapshot caching behaviour for directory-based skill loaders.

``MarketplaceDirectoryLoader`` uses identical logic for reading/writing
snapshots to a persistent store (MongoDB, file, or in-memory) and for
TTL-based validity checks.  This mixin extracts that common logic so it
lives in one place.

Subclasses must provide these instance attributes (set in ``__init__``):

*  ``_snapshot_cache_store: BaseStore | None``
*  ``_snapshot_cache_collection: str | None``
*  ``_SNAPSHOT_CACHE_KEY: str``  (class-level constant)
*  ``_reload_ttl_seconds: float | None``
*  ``_snapshot: SkillSnapshot | None``
*  ``_snapshot_loaded_at: float | None``

And one abstract property:

*  ``_loader_display_name`` — used only for log messages.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.models.skills_model import SkillSnapshot
from langchain_ai_skills_framework.utilities.snapshot_serializer import (
    deserialize_snapshot,
    serialize_snapshot,
)

if TYPE_CHECKING:
    from key_value.aio.stores.base import BaseStore

logger = logging.getLogger(__name__)


class SnapshotCacheMixin:
    """Mixin that provides snapshot cache read/write and TTL helpers."""

    # -- Attributes expected on the concrete class --
    _snapshot_cache_store: BaseStore | None
    _snapshot_cache_collection: str | None
    _SNAPSHOT_CACHE_KEY: str
    _reload_ttl_seconds: float | None
    _snapshot: SkillSnapshot | None
    _snapshot_loaded_at: float | None

    @property
    def _loader_display_name(self) -> str:
        """Human-readable name for log messages (e.g. 'MarketplaceDirectoryLoader abc123')."""
        return self.__class__.__name__

    # -- Snapshot cache I/O --

    async def _read_from_snapshot_cache(self) -> SkillSnapshot | None:
        """Best-effort: any store or deserialization error returns ``None``."""
        if not self._snapshot_cache_store:
            return None
        try:
            data = await self._snapshot_cache_store.get(
                self._SNAPSHOT_CACHE_KEY,
                collection=self._snapshot_cache_collection,
            )
            if data is None:
                return None
            snapshot = deserialize_snapshot(data=data)
            logger.info(
                "%s loaded snapshot from cache (%d skills)",
                self._loader_display_name,
                len(snapshot.ordered_summaries),
            )
            return snapshot
        except Exception:
            logger.debug(
                "%s snapshot cache read failed",
                self._loader_display_name,
                exc_info=True,
            )
            return None

    async def _write_to_snapshot_cache(self, snapshot: SkillSnapshot) -> None:
        """Best-effort: a write failure must not prevent returning the snapshot."""
        if not self._snapshot_cache_store:
            return
        try:
            data = serialize_snapshot(snapshot=snapshot)
            await self._snapshot_cache_store.put(
                self._SNAPSHOT_CACHE_KEY,
                data,
                ttl=self._reload_ttl_seconds,
                collection=self._snapshot_cache_collection,
            )
            logger.debug(
                "%s wrote snapshot to cache (%d skills)",
                self._loader_display_name,
                len(snapshot.ordered_summaries),
            )
        except Exception:
            logger.debug(
                "%s snapshot cache write failed",
                self._loader_display_name,
                exc_info=True,
            )

    # -- TTL helpers --

    def _is_snapshot_valid_unlocked(self) -> bool:
        """Lock-held validity check based on snapshot presence and TTL age."""
        if self._snapshot is None:
            return False
        if self._reload_ttl_seconds is None:
            return True
        if self._snapshot_loaded_at is None:
            return False
        return (time.monotonic() - self._snapshot_loaded_at) < self._reload_ttl_seconds

    @staticmethod
    def _resolve_reload_ttl_seconds(
        *,
        environment_variables: SkillLoaderEnvironmentVariables,
    ) -> float | None:
        """Resolve loader TTL from environment, defaulting to one hour."""
        configured = environment_variables.skills_cache_timeout_seconds
        if isinstance(configured, bool):
            return 3600.0
        if not isinstance(configured, (int, float)):
            return 3600.0
        configured_seconds = float(configured)
        if configured_seconds <= 0:
            return None
        return configured_seconds
