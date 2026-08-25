from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.skills_model import SkillSummary
from langchain_ai_skills_framework.services.list_skills_service import (
    ListSkillsService,
)


@pytest.mark.asyncio
async def test_list_skills_excludes_specified_states() -> None:
    """Test that exclude_states filters out skills with matching states."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)

    # Create summaries with different states
    summaries = [
        SkillSummary(
            name="skill1",
            description="Published skill",
            plugin_name="plugin1",
            state="published",
        ),
        SkillSummary(
            name="skill2",
            description="Draft skill",
            plugin_name="plugin1",
            state="draft",
        ),
        SkillSummary(
            name="skill3",
            description="Archived skill",
            plugin_name="plugin1",
            state="archived",
        ),
        SkillSummary(
            name="skill4",
            description="Another published skill",
            plugin_name="plugin1",
            state="published",
        ),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(
        user_id="user123",
        exclude_states={"draft", "archived"},
    )

    # Should only return published skills
    assert len(result) == 2
    assert all(skill.state == "published" for skill in result)
    assert {skill.name for skill in result} == {"skill1", "skill4"}


@pytest.mark.asyncio
async def test_list_skills_no_exclusion_when_none() -> None:
    """Test that no skills are filtered when exclude_states is None."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)

    summaries = [
        SkillSummary(
            name="skill1",
            description="Published skill",
            plugin_name="plugin1",
            state="published",
        ),
        SkillSummary(
            name="skill2",
            description="Draft skill",
            plugin_name="plugin1",
            state="draft",
        ),
        SkillSummary(
            name="skill3",
            description="Archived skill",
            plugin_name="plugin1",
            state="archived",
        ),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(user_id="user123", exclude_states=None)

    # Should return all skills
    assert len(result) == 3
    assert {skill.name for skill in result} == {"skill1", "skill2", "skill3"}


@pytest.mark.asyncio
async def test_list_skills_exclude_states_with_other_filters() -> None:
    """Test that exclude_states works together with other filters."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)

    summaries = [
        SkillSummary(
            name="skill1",
            description="Published skill in folder1",
            plugin_name="plugin1",
            folder="folder1",
            state="published",
        ),
        SkillSummary(
            name="skill2",
            description="Draft skill in folder1",
            plugin_name="plugin1",
            folder="folder1",
            state="draft",
        ),
        SkillSummary(
            name="skill3",
            description="Published skill in folder2",
            plugin_name="plugin1",
            folder="folder2",
            state="published",
        ),
        SkillSummary(
            name="skill4",
            description="Published skill in folder1 (plugin2)",
            plugin_name="plugin2",
            folder="folder1",
            state="published",
        ),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(
        user_id="user123",
        plugin_name="plugin1",
        folder="folder1",
        exclude_states={"draft"},
    )

    # Should return only plugin1, folder1, non-draft skills
    assert len(result) == 1
    assert result[0].name == "skill1"
    assert result[0].plugin_name == "plugin1"
    assert result[0].folder == "folder1"
    assert result[0].state == "published"


@pytest.mark.asyncio
async def test_list_skills_exclude_empty_set() -> None:
    """Test that an empty exclude_states set doesn't filter anything."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)

    summaries = [
        SkillSummary(
            name="skill1",
            description="Published skill",
            plugin_name="plugin1",
            state="published",
        ),
        SkillSummary(
            name="skill2",
            description="Draft skill",
            plugin_name="plugin1",
            state="draft",
        ),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(user_id="user123", exclude_states=set())

    # Empty set is falsy, so no filtering
    assert len(result) == 2


@pytest.mark.asyncio
async def test_allow_states_keeps_only_listed_states() -> None:
    """Test that allow_states is a whitelist filter — only listed states pass through."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    summaries = [
        SkillSummary(name="pub1", description="", plugin_name="p1", state="published"),
        SkillSummary(name="draft1", description="", plugin_name="p1", state="draft"),
        SkillSummary(name="stage1", description="", plugin_name="p1", state="staging"),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(user_id="u", allow_states={"published"})

    assert {s.name for s in result} == {"pub1"}


@pytest.mark.asyncio
async def test_allow_states_takes_precedence_over_exclude_states() -> None:
    """When both are set, allow_states wins (positive whitelist beats deny-list)."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)
    summaries = [
        SkillSummary(name="pub1", description="", plugin_name="p1", state="published"),
        SkillSummary(name="draft1", description="", plugin_name="p1", state="draft"),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(
        user_id="u",
        allow_states={"draft"},
        exclude_states={"draft"},
    )

    assert {s.name for s in result} == {"draft1"}


@pytest.mark.asyncio
async def test_list_skills_basic_functionality() -> None:
    """Test basic list_skills without filters (regression test)."""
    mock_loader = AsyncMock(spec=SkillLoaderProtocol)

    date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    summaries = [
        SkillSummary(
            name="zebra-skill",
            description="Last alphabetically",
            plugin_name="plugin1",
            state="published",
            date_modified=date,
        ),
        SkillSummary(
            name="alpha-skill",
            description="First alphabetically",
            plugin_name="plugin1",
            state="draft",
            folder="test-folder",
        ),
    ]
    mock_loader.list_all_summaries.return_value = summaries

    service = ListSkillsService(skill_loader=mock_loader)
    result = await service.execute(user_id="user123")

    # Should be sorted by name
    assert len(result) == 2
    assert result[0].name == "alpha-skill"
    assert result[0].folder == "test-folder"
    assert result[1].name == "zebra-skill"
    assert result[1].date_modified == date
