"""Tests for :class:`SaveSkillService`.

These tests pin down the structured-result contract used to distinguish
real saves (HTTP 200) from soft-failure validation paths (HTTP 400) at
caller boundaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.services.save_skill_service import (
    SaveSkillResult,
    SaveSkillService,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)

VALID_CONTENT = "---\nname: test-skill\ndescription: A test skill\n---\n# Body"


def _make_store(*, exists: bool = False) -> AsyncMock:
    store = AsyncMock(spec=PluginSkillStore)
    store.skill_exists.return_value = exists
    store.save_skill.return_value = MongoPluginSkillDocument(
        plugin_name="test-plugin",
        author="user-1",
        skill_name="test-skill",
        path="test-plugin/skills/test-skill/SKILL.md",
        description="A test",
        content=VALID_CONTENT,
        modified_by="user-1",
        date_created=datetime.now(timezone.utc),
        date_modified=datetime.now(timezone.utc),
    )
    return store


@pytest.mark.asyncio
async def test_successful_save_returns_ok_result() -> None:
    store = _make_store()
    service = SaveSkillService(mongo_skill_loader=store)

    result = await service.execute(
        user_id="user-1",
        plugin_name="test-plugin",
        skill_name="test-skill",
        content=VALID_CONTENT,
    )

    assert isinstance(result, SaveSkillResult)
    assert result.ok is True
    assert "saved successfully" in result.message
    store.save_skill.assert_awaited_once()


@pytest.mark.asyncio
async def test_unparsable_frontmatter_returns_not_ok_and_does_not_persist() -> None:
    store = _make_store()
    service = SaveSkillService(mongo_skill_loader=store)

    result = await service.execute(
        user_id="user-1",
        plugin_name="test-plugin",
        skill_name="test-skill",
        content="# No frontmatter at all",
    )

    assert result.ok is False
    assert "validation failed" in result.message.lower()
    store.save_skill.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_metadata_returns_not_ok_and_does_not_persist() -> None:
    store = _make_store()
    service = SaveSkillService(mongo_skill_loader=store)

    bad_content = "---\nname: test-skill\ndescription: A test skill\nunexpected_field: nope\n---\n# Body"

    result = await service.execute(
        user_id="user-1",
        plugin_name="test-plugin",
        skill_name="test-skill",
        content=bad_content,
    )

    assert result.ok is False
    assert "validation failed" in result.message.lower()
    store.save_skill.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_skill_name_returns_not_ok() -> None:
    store = _make_store()
    service = SaveSkillService(mongo_skill_loader=store)

    content_without_name = "---\ndescription: only description\n---\n# Body"

    result = await service.execute(
        user_id="user-1",
        plugin_name="test-plugin",
        skill_name=None,
        content=content_without_name,
    )

    assert result.ok is False
    assert "validation failed" in result.message.lower()
    store.save_skill.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_exists_with_update_disabled_returns_not_ok() -> None:
    store = _make_store(exists=True)
    service = SaveSkillService(mongo_skill_loader=store)

    result = await service.execute(
        user_id="user-1",
        plugin_name="test-plugin",
        skill_name="test-skill",
        content=VALID_CONTENT,
        update_if_exists=False,
    )

    assert result.ok is False
    assert "already exists" in result.message
    store.save_skill.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_failure_raises_skill_operation_error() -> None:
    store = _make_store()
    store.save_skill.side_effect = RuntimeError("mongo unavailable")
    service = SaveSkillService(mongo_skill_loader=store)

    with pytest.raises(SkillOperationError, match="Unable to save skill"):
        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            content=VALID_CONTENT,
        )
