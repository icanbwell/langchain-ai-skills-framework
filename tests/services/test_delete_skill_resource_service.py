from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.delete_skill_resource_service import (
    DeleteSkillResourceService,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


@pytest.mark.asyncio
async def test_delete_skill_resource_success() -> None:
    """Test successful resource deletion."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    mock_store.delete_resource.return_value = True

    service = DeleteSkillResourceService(mongo_skill_loader=mock_store)
    result = await service.execute(
        user_id="user123",
        plugin_name="test-plugin",
        skill_name="test-skill",
        resource_name="test.txt",
    )

    assert "deleted successfully" in result
    mock_store.delete_resource.assert_called_once_with(
        author="user123",
        plugin_name="test-plugin",
        skill_name="test-skill",
        resource_name="test.txt",
    )


@pytest.mark.asyncio
async def test_delete_skill_resource_not_found() -> None:
    """Test resource not found raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    mock_store.delete_resource.return_value = False

    service = DeleteSkillResourceService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="not found"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="missing.txt",
        )


@pytest.mark.asyncio
async def test_delete_skill_resource_store_not_configured() -> None:
    """Test that missing store raises SkillOperationError."""
    service = DeleteSkillResourceService(mongo_skill_loader=None)

    with pytest.raises(SkillOperationError, match="not configured"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="test.txt",
        )


@pytest.mark.asyncio
async def test_delete_skill_resource_empty_user_id() -> None:
    """Test that empty user_id raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    service = DeleteSkillResourceService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="user_id is required"):
        await service.execute(
            user_id="",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="test.txt",
        )


@pytest.mark.asyncio
async def test_delete_skill_resource_empty_skill_name() -> None:
    """Test that empty skill_name raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    service = DeleteSkillResourceService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="skill_name must be"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="",
            resource_name="test.txt",
        )


@pytest.mark.asyncio
async def test_delete_skill_resource_empty_resource_name() -> None:
    """Test that empty resource_name raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    service = DeleteSkillResourceService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="resource_name must be"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="",
        )


@pytest.mark.asyncio
async def test_delete_skill_resource_store_exception() -> None:
    """Test that store exceptions are wrapped in SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    mock_store.delete_resource.side_effect = RuntimeError("DB connection failed")

    service = DeleteSkillResourceService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="internal error"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            resource_name="test.txt",
        )
