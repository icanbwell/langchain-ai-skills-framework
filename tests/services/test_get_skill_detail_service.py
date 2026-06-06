from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary
from langchain_ai_skills_framework.services.get_skill_detail_service import (
    GetSkillDetailService,
    SkillDetailResult,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


@pytest.mark.asyncio
async def test_get_skill_detail_success() -> None:
    """Test successful skill detail retrieval with user metadata."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    # Setup loader response
    summary = SkillSummary(
        name="test-skill",
        description="Test skill",
        plugin_name="test-plugin",
        folder="test-folder",
        state="published",
    )
    details = SkillDetails(summary=summary, content="# Test Content")
    mock_loader.get_skill_details_for_user.return_value = details

    # Setup store responses
    mock_store.list_resource_names.return_value = ["resource1.txt", "resource2.md"]
    mock_store.list_script_names.return_value = ["script1.py", "script2.sh"]
    mock_store.get_skill_details.return_value = SkillDetails(
        summary=SkillSummary(
            name="test-skill",
            description="Test skill",
            plugin_name="test-plugin",
            folder="user-folder",
            state="draft",
        ),
        content="",
    )

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)
    result = await service.execute(user_id="user123", plugin_name="test-plugin", skill_name="test-skill")

    assert isinstance(result, SkillDetailResult)
    assert result.content == "# Test Content"
    assert result.resources == ["resource1.txt", "resource2.md"]
    assert result.scripts == ["script1.py", "script2.sh"]
    assert result.folder == "user-folder"
    assert result.state == "draft"

    mock_loader.get_skill_details_for_user.assert_called_once_with(
        user_id="user123", plugin_name="test-plugin", skill_name="test-skill"
    )
    mock_store.list_resource_names.assert_called_once_with(
        author="user123", plugin_name="test-plugin", skill_name="test-skill"
    )
    mock_store.list_script_names.assert_called_once_with(
        author="user123", plugin_name="test-plugin", skill_name="test-skill"
    )
    mock_store.get_skill_details.assert_called_once_with(
        author="user123", plugin_name="test-plugin", skill_name="test-skill"
    )


@pytest.mark.asyncio
async def test_get_skill_detail_fallback_to_system_author() -> None:
    """Test fallback to system author when user metadata not found."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    # Setup loader response
    summary = SkillSummary(
        name="test-skill",
        description="Test skill",
        plugin_name="test-plugin",
    )
    details = SkillDetails(summary=summary, content="# Test Content")
    mock_loader.get_skill_details_for_user.return_value = details

    # Setup store responses
    mock_store.list_resource_names.return_value = []
    mock_store.list_script_names.return_value = []

    # User lookup fails, system succeeds
    async def get_skill_details_side_effect(*, author: str, **kwargs):  # type: ignore[no-untyped-def]
        if author == "user123":
            raise SkillNotFoundError("Not found")
        return SkillDetails(
            summary=SkillSummary(
                name="test-skill",
                description="Test skill",
                plugin_name="test-plugin",
                folder="system-folder",
                state="published",
            ),
            content="",
        )

    mock_store.get_skill_details.side_effect = get_skill_details_side_effect

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)
    result = await service.execute(user_id="user123", plugin_name="test-plugin", skill_name="test-skill")

    assert result.content == "# Test Content"
    assert result.folder == "system-folder"
    assert result.state == "published"

    # Should have called get_skill_details twice (user then system)
    assert mock_store.get_skill_details.call_count == 2


@pytest.mark.asyncio
async def test_get_skill_detail_defaults_when_both_lookups_fail() -> None:
    """Test default values when both user and system metadata lookups fail."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    # Setup loader response
    summary = SkillSummary(
        name="test-skill",
        description="Test skill",
        plugin_name="test-plugin",
    )
    details = SkillDetails(summary=summary, content="# Test Content")
    mock_loader.get_skill_details_for_user.return_value = details

    # Setup store responses
    mock_store.list_resource_names.return_value = []
    mock_store.list_script_names.return_value = []
    mock_store.get_skill_details.side_effect = SkillNotFoundError("Not found")

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)
    result = await service.execute(user_id="user123", plugin_name="test-plugin", skill_name="test-skill")

    assert result.content == "# Test Content"
    assert result.folder is None
    assert result.state == "draft"

    # Should have called get_skill_details twice (user then system)
    assert mock_store.get_skill_details.call_count == 2


@pytest.mark.asyncio
async def test_get_skill_detail_skill_not_found() -> None:
    """Test SkillOperationError when skill content not found."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    mock_loader.get_skill_details_for_user.side_effect = SkillNotFoundError("Skill not found")

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="Skill not found"):
        await service.execute(user_id="user123", plugin_name="test-plugin", skill_name="test-skill")


@pytest.mark.asyncio
async def test_get_skill_detail_empty_user_id() -> None:
    """Test that empty user_id raises SkillOperationError."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="user_id is required"):
        await service.execute(user_id="", plugin_name="test-plugin", skill_name="test-skill")


@pytest.mark.asyncio
async def test_get_skill_detail_empty_plugin_name() -> None:
    """Test that empty plugin_name raises SkillOperationError."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="plugin_name must be"):
        await service.execute(user_id="user123", plugin_name="", skill_name="test-skill")


@pytest.mark.asyncio
async def test_get_skill_detail_empty_skill_name() -> None:
    """Test that empty skill_name raises SkillOperationError."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    mock_store = AsyncMock(spec=PluginSkillStore)

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=mock_store)

    with pytest.raises(SkillOperationError, match="skill_name must be"):
        await service.execute(user_id="user123", plugin_name="test-plugin", skill_name="")


@pytest.mark.asyncio
async def test_get_skill_detail_no_store() -> None:
    """Test that service works without a store (uses loader metadata)."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)

    # Setup loader response with metadata
    summary = SkillSummary(
        name="test-skill",
        description="Test skill",
        plugin_name="test-plugin",
        folder="loader-folder",
        state="published",
    )
    details = SkillDetails(summary=summary, content="# Test Content")
    mock_loader.get_skill_details_for_user.return_value = details

    service = GetSkillDetailService(skill_loader=mock_loader, mongo_skill_loader=None)
    result = await service.execute(user_id="user123", plugin_name="test-plugin", skill_name="test-skill")

    assert result.content == "# Test Content"
    assert result.resources == []
    assert result.scripts == []
    assert result.folder == "loader-folder"
    assert result.state == "published"
