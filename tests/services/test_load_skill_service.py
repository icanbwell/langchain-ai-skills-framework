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
from langchain_ai_skills_framework.services.load_skill_service import LoadSkillService


def _make_skill_details(
    *,
    name: str = "assess-stress",
    plugin_name: str = "bailey",
    content: str = "# Stress assessment\nSkill body",
    author: str | None = None,
) -> SkillDetails:
    metadata: dict[str, object] = {}
    if author is not None:
        metadata["user_id"] = author
    summary = SkillSummary(
        name=name,
        description="desc",
        plugin_name=plugin_name,
        folder=None,
        state="staging",
        metadata=metadata,
    )
    return SkillDetails(summary=summary, content=content)


class TestLoadSkillServiceOmittedPluginName:
    """When the LLM cannot reliably know ``plugin_name``, omitting it must succeed.

    The MCP tool was previously requiring ``plugin_name`` as a required string,
    which forced the model to guess (often producing nonsense like the skill
    name doubling as the plugin). Making it optional and resolving by
    ``(user_id, skill_name)`` removes the guess.
    """

    @pytest.mark.asyncio
    async def test_resolves_skill_when_plugin_name_omitted(self) -> None:
        loader = AsyncMock(spec=SkillLoaderProtocol)
        details = _make_skill_details(name="assess-stress", plugin_name="bailey")
        loader.get_skill_details_for_user.return_value = details

        service = LoadSkillService(skill_loader=loader)

        content = await service.execute(user_id="tester-subject-id", skill_name="assess-stress")

        assert content == "# Stress assessment\nSkill body"
        loader.get_skill_details_for_user.assert_awaited_once_with(
            user_id="tester-subject-id",
            plugin_name=None,
            skill_name="assess-stress",
        )

    @pytest.mark.asyncio
    async def test_still_accepts_explicit_plugin_name(self) -> None:
        loader = AsyncMock(spec=SkillLoaderProtocol)
        details = _make_skill_details(name="assess-stress", plugin_name="bailey")
        loader.get_skill_details_for_user.return_value = details

        service = LoadSkillService(skill_loader=loader)

        await service.execute(
            user_id="tester-subject-id",
            plugin_name="bailey",
            skill_name="assess-stress",
        )

        loader.get_skill_details_for_user.assert_awaited_once_with(
            user_id="tester-subject-id",
            plugin_name="bailey",
            skill_name="assess-stress",
        )


class TestLoadSkillServiceUsageRecording:
    """Usage is recorded against the *resolved* plugin, not the caller-supplied one.

    Even when the caller omits ``plugin_name``, the loader returns the canonical
    plugin via ``summary.plugin_name`` — that's what gets recorded.
    """

    @pytest.mark.asyncio
    async def test_records_usage_with_resolved_plugin_name(self) -> None:
        loader = AsyncMock(spec=SkillLoaderProtocol)
        store = AsyncMock(spec=PluginSkillStore)
        details = _make_skill_details(name="assess-stress", plugin_name="bailey")
        loader.get_skill_details_for_user.return_value = details

        service = LoadSkillService(skill_loader=loader, user_skill_store=store)

        await service.execute(user_id="tester-subject-id", skill_name="assess-stress")

        store.record_skill_usage.assert_awaited_once_with(
            plugin_name="bailey",
            skill_name="assess-stress",
            user_id="tester-subject-id",
        )

    @pytest.mark.asyncio
    async def test_skips_usage_recording_when_skill_not_found(self) -> None:
        loader = AsyncMock(spec=SkillLoaderProtocol)
        store = AsyncMock(spec=PluginSkillStore)
        loader.get_skill_details_for_user.side_effect = SkillNotFoundError("no such skill")
        loader.list_all_summaries.return_value = []

        service = LoadSkillService(skill_loader=loader, user_skill_store=store)

        content = await service.execute(user_id="tester-subject-id", skill_name="missing")

        assert "not found" in content
        store.record_skill_usage.assert_not_called()


class TestLoadSkillServiceAuthorPrefix:
    @pytest.mark.asyncio
    async def test_prefixes_content_with_author_when_metadata_present(self) -> None:
        loader = AsyncMock(spec=SkillLoaderProtocol)
        details = _make_skill_details(
            name="assess-stress",
            plugin_name="bailey",
            content="# body",
            author="tester-subject-id",
        )
        loader.get_skill_details_for_user.return_value = details

        service = LoadSkillService(skill_loader=loader)

        content = await service.execute(user_id="tester-subject-id", skill_name="assess-stress")

        assert content == "Author: tester-subject-id\n\n# body"

    @pytest.mark.asyncio
    async def test_no_author_prefix_when_metadata_absent(self) -> None:
        loader = AsyncMock(spec=SkillLoaderProtocol)
        details = _make_skill_details(name="phq-9", plugin_name="bailey", content="# body", author=None)
        loader.get_skill_details_for_user.return_value = details

        service = LoadSkillService(skill_loader=loader)

        content = await service.execute(user_id="tester-subject-id", skill_name="phq-9")

        assert content == "# body"
