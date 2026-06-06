from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.services.read_skill_script_service import (
    ReadSkillScriptService,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


@pytest.mark.asyncio
async def test_read_skill_script_success() -> None:
    """Test successful script read."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    expected_content = "#!/bin/bash\necho 'Hello World'"
    mock_store.read_script.return_value = expected_content

    service = ReadSkillScriptService(mongo_skill_loader=mock_store)
    result = await service.execute(
        user_id="user123",
        plugin_name="test-plugin",
        skill_name="test-skill",
        script_name="test.sh",
    )

    assert result == expected_content
    mock_store.read_script.assert_called_once_with(
        author="user123",
        plugin_name="test-plugin",
        skill_name="test-skill",
        script_name="test.sh",
    )


@pytest.mark.asyncio
async def test_read_skill_script_store_not_configured() -> None:
    """Test that missing store raises SkillOperationError."""
    service = ReadSkillScriptService(mongo_skill_loader=None)

    with pytest.raises(SkillOperationError, match="not configured"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
        )


@pytest.mark.asyncio
async def test_read_skill_script_empty_user_id() -> None:
    """Test that empty user_id raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    service = ReadSkillScriptService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="user_id is required"):
        await service.execute(
            user_id="",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
        )


@pytest.mark.asyncio
async def test_read_skill_script_empty_skill_name() -> None:
    """Test that empty skill_name raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    service = ReadSkillScriptService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="skill_name must be"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="",
            script_name="test.sh",
        )


@pytest.mark.asyncio
async def test_read_skill_script_empty_script_name() -> None:
    """Test that empty script_name raises SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    service = ReadSkillScriptService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="script_name must be"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="",
        )


@pytest.mark.asyncio
async def test_read_skill_script_store_exception() -> None:
    """Test that store exceptions are wrapped in SkillOperationError."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    mock_store.read_script.side_effect = RuntimeError("DB connection failed")

    service = ReadSkillScriptService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="internal error"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
        )


@pytest.mark.asyncio
async def test_read_skill_script_not_found() -> None:
    """Test that script not found exceptions are wrapped."""
    mock_store = AsyncMock(spec=PluginSkillStore)
    mock_store.read_script.side_effect = FileNotFoundError("Script not found")

    service = ReadSkillScriptService(mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="internal error"):
        await service.execute(
            user_id="user123",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="missing.sh",
        )
