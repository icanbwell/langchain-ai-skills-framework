"""Tests for :class:`DeleteSkillService`.

Pins down the structured-result contract: ``ok=True`` when a row was
actually removed, ``ok=False`` when nothing was found to delete
(idempotent — the route still returns 200 — but the distinction is
available to callers that want to act on it).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.services.delete_skill_service import DeleteSkillService
from langchain_ai_skills_framework.services.mutation_result import MutationResult
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


@pytest.mark.asyncio
async def test_delete_existing_skill_returns_ok_result() -> None:
    store = AsyncMock(spec=PluginSkillStore)
    store.delete_skill.return_value = True
    service = DeleteSkillService(mongo_skill_loader=store)

    result = await service.execute(user_id="user-1", plugin_name="test-plugin", skill_name="test-skill")

    assert isinstance(result, MutationResult)
    assert result.ok is True
    assert "deleted successfully" in result.message
    store.delete_skill.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_missing_skill_returns_not_ok_result() -> None:
    store = AsyncMock(spec=PluginSkillStore)
    store.delete_skill.return_value = False
    service = DeleteSkillService(mongo_skill_loader=store)

    result = await service.execute(user_id="user-1", plugin_name="test-plugin", skill_name="missing")

    assert result.ok is False
    assert "not found" in result.message
    store.delete_skill.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_store_failure_raises() -> None:
    store = AsyncMock(spec=PluginSkillStore)
    store.delete_skill.side_effect = RuntimeError("mongo unavailable")
    service = DeleteSkillService(mongo_skill_loader=store)

    with pytest.raises(SkillOperationError, match="Unable to delete skill"):
        await service.execute(user_id="user-1", plugin_name="test-plugin", skill_name="test-skill")
