from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["CONFIG"])


@dataclass(frozen=True, slots=True)
class SkillCacheSnapshot:
    """Immutable snapshot of loaded skills."""

    details_by_name: Mapping[str, SkillDetails]
    ordered_summaries: tuple[SkillSummary, ...]


@runtime_checkable
class SkillCacheEnvironmentVariables(Protocol):
    @property
    def skills_cache_timeout_seconds(self) -> int: ...


class SkillCache:
    """Thread-safe cache for Agent Skill definitions with optional TTL."""

    def __init__(
        self,
        *,
        ttl_seconds: Optional[float] = None,
        environment_variables: SkillCacheEnvironmentVariables,
    ) -> None:
        if environment_variables is None:
            raise ValueError("environment_variables cannot be None")
        if not isinstance(environment_variables, SkillCacheEnvironmentVariables):
            raise ValueError(
                "environment_variables must be a SkillCacheEnvironmentVariables"
            )
        configured_ttl_seconds = ttl_seconds
        if configured_ttl_seconds is None and environment_variables is not None:
            configured_ttl_seconds = float(
                environment_variables.skills_cache_timeout_seconds
            )

        self._ttl: Optional[float] = (
            configured_ttl_seconds
            if configured_ttl_seconds and configured_ttl_seconds > 0
            else None
        )
        self._lock = RLock()
        self._snapshot: Optional[SkillCacheSnapshot] = None
        self._timestamp: Optional[float] = None
        self._identifier: UUID = uuid4()

    def is_valid(self) -> bool:
        with self._lock:
            if self._snapshot is None:
                return False
            if self._ttl is None:
                return True
            assert self._timestamp is not None
            current_time = time.time()
            is_valid = current_time - self._timestamp < self._ttl
            logger.debug(
                "SkillCache %s validity check: %s (now=%s, ts=%s, ttl=%s)",
                self._identifier,
                is_valid,
                current_time,
                self._timestamp,
                self._ttl,
            )
            return is_valid

    def get(self) -> Optional[SkillCacheSnapshot]:
        with self._lock:
            if self.is_valid():
                logger.debug(
                    "SkillCache %s returning cached snapshot", self._identifier
                )
                return self._snapshot
            return None

    def set(self, snapshot: SkillCacheSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._timestamp = time.time()
            logger.debug(
                "SkillCache %s stored snapshot with %d skills",
                self._identifier,
                len(snapshot.ordered_summaries),
            )

    def clear(self) -> None:
        with self._lock:
            self._snapshot = None
            self._timestamp = None
            logger.debug("SkillCache %s cleared", self._identifier)
