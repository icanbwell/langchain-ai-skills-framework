"""Tests for :class:`SaveSkillResourceService`.

Pins down the structured-result contract introduced alongside
``SaveSkillResult`` to keep all skill-mutation services symmetric.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginResourceDocument,
)
from langchain_ai_skills_framework.services.save_skill_resource_service import (
    SaveSkillResourceResult,
    SaveSkillResourceService,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


def _make_resource_doc() -> MongoPluginResourceDocument:
    return MongoPluginResourceDocument(
        plugin_name="test-plugin",
        author="user-1",
        skill_name="test-skill",
        resource_name="forms.md",
        content="# forms",
        modified_by="user-1",
        date_created=datetime.now(timezone.utc),
        date_modified=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_save_resource_returns_ok_result() -> None:
    store = AsyncMock(spec=PluginSkillStore)
    store.save_resource.return_value = _make_resource_doc()
    service = SaveSkillResourceService(mongo_skill_loader=store)

    result = await service.execute(
        user_id="user-1",
        plugin_name="test-plugin",
        skill_name="test-skill",
        resource_name="forms.md",
        content="# forms",
    )

    assert isinstance(result, SaveSkillResourceResult)
    assert result.ok is True
    assert "forms.md" in result.message
    store.save_resource.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_resource_store_failure_raises() -> None:
    store = AsyncMock(spec=PluginSkillStore)
    store.save_resource.side_effect = RuntimeError("mongo unavailable")
    service = SaveSkillResourceService(mongo_skill_loader=store)

    with pytest.raises(SkillOperationError, match="Unable to save resource"):
        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="forms.md",
            content="# forms",
        )


@pytest.mark.asyncio
async def test_save_resource_missing_store_raises() -> None:
    service = SaveSkillResourceService(mongo_skill_loader=None)

    with pytest.raises(SkillOperationError, match="mongo_skill_loader is not configured"):
        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="forms.md",
            content="# forms",
        )
