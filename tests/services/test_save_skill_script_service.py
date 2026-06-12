from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginScriptDocument,
)
from langchain_ai_skills_framework.services.post_save_script_hook import (
    PostSaveScriptHook,
)
from langchain_ai_skills_framework.services.save_skill_script_service import (
    SaveSkillScriptResult,
    SaveSkillScriptService,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


def _make_script_doc(*, script_name: str = "test.sh") -> MongoPluginScriptDocument:
    return MongoPluginScriptDocument(
        plugin_name="test-plugin",
        author="user-1",
        skill_name="test-skill",
        script_name=script_name,
        content="#!/bin/bash\necho test",
        modified_by="user-1",
        date_created=datetime.now(timezone.utc),
        date_modified=datetime.now(timezone.utc),
    )


class TestSaveSkillScriptService:
    @pytest.mark.asyncio
    async def test_saves_script_successfully(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.save_script.return_value = _make_script_doc()
        service = SaveSkillScriptService(mongo_skill_loader=store)

        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
            content="#!/bin/bash\necho test",
        )

        assert isinstance(result, SaveSkillScriptResult)
        assert result.ok is True
        assert "Script 'test.sh' saved for skill 'test-skill'" in result.message
        store.save_script.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
            content="#!/bin/bash\necho test",
            modified_by="user-1",
            folder=None,
            path=None,
        )

    @pytest.mark.asyncio
    async def test_invokes_hook_after_save(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.save_script.return_value = _make_script_doc()
        hook = AsyncMock(spec=PostSaveScriptHook)
        service = SaveSkillScriptService(mongo_skill_loader=store, post_save_hook=hook)

        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
            content="#!/bin/bash\necho test",
        )

        assert "Script 'test.sh' saved" in result.message
        hook.on_script_saved.assert_awaited_once_with(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
        )

    @pytest.mark.asyncio
    async def test_works_without_hook(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.save_script.return_value = _make_script_doc()
        service = SaveSkillScriptService(mongo_skill_loader=store, post_save_hook=None)

        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
            content="#!/bin/bash\necho test",
        )

        assert "Script 'test.sh' saved" in result.message

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_break_save(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.save_script.return_value = _make_script_doc()
        hook = AsyncMock(spec=PostSaveScriptHook)
        hook.on_script_saved.side_effect = RuntimeError("Hook failed")
        service = SaveSkillScriptService(mongo_skill_loader=store, post_save_hook=hook)

        # Should succeed despite hook failure
        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            script_name="test.sh",
            content="#!/bin/bash\necho test",
        )

        assert "Script 'test.sh' saved" in result.message
        hook.on_script_saved.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        service = SaveSkillScriptService(mongo_skill_loader=AsyncMock())

        with pytest.raises(SkillOperationError, match="user_id is required"):
            await service.execute(
                user_id="",
                plugin_name="test-plugin",
                skill_name="test-skill",
                script_name="test.sh",
                content="echo test",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        service = SaveSkillScriptService(mongo_skill_loader=AsyncMock())

        with pytest.raises(SkillOperationError, match="skill_name must be a non-empty"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="  ",
                script_name="test.sh",
                content="echo test",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_script_name(self) -> None:
        service = SaveSkillScriptService(mongo_skill_loader=AsyncMock())

        with pytest.raises(SkillOperationError, match="script_name must be a non-empty"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                script_name="  ",
                content="echo test",
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_content(self) -> None:
        service = SaveSkillScriptService(mongo_skill_loader=AsyncMock())

        with pytest.raises(SkillOperationError, match="content must be a non-empty"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                script_name="test.sh",
                content="  ",
            )

    @pytest.mark.asyncio
    async def test_rejects_when_store_not_configured(self) -> None:
        service = SaveSkillScriptService(mongo_skill_loader=None)

        with pytest.raises(SkillOperationError, match="mongo_skill_loader is not configured"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                script_name="test.sh",
                content="echo test",
            )

    @pytest.mark.asyncio
    async def test_wraps_unexpected_exception(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.save_script.side_effect = RuntimeError("db down")
        service = SaveSkillScriptService(mongo_skill_loader=store)

        with pytest.raises(SkillOperationError, match="Unable to save script"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                script_name="test.sh",
                content="echo test",
            )
