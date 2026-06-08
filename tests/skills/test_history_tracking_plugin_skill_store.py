"""Tests for HistoryTrackingPluginSkillStore."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.history_tracking_plugin_skill_store import (
    HistoryTrackingPluginSkillStore,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary


@pytest.fixture
def inner_store() -> AsyncMock:
    store = AsyncMock()
    store.ensure_indexes = AsyncMock()
    store.skill_exists = AsyncMock(return_value=False)
    store.resource_exists = AsyncMock(return_value=False)
    store.script_exists = AsyncMock(return_value=False)
    store.save_skill = AsyncMock(
        return_value=MongoPluginSkillDocument(
            plugin_name="my-plugin",
            skill_name="my-skill",
            path="my-plugin/skills/my-skill/SKILL.md",
            description="A test skill",
            content="# Skill content",
            author="user-1",
            modified_by="user-1",
        )
    )
    store.save_resource = AsyncMock(
        return_value=MongoPluginResourceDocument(
            plugin_name="my-plugin",
            skill_name="my-skill",
            resource_name="data.json",
            path="my-plugin/skills/my-skill/data.json",
            content='{"key": "value"}',
            author="user-1",
            modified_by="user-1",
        )
    )
    store.save_script = AsyncMock(
        return_value=MongoPluginScriptDocument(
            plugin_name="my-plugin",
            skill_name="my-skill",
            script_name="run.py",
            path="my-plugin/skills/my-skill/scripts/run.py",
            content="print('hello')",
            author="user-1",
            modified_by="user-1",
        )
    )
    store.save_plugin = AsyncMock(
        return_value=MongoPluginDefinitionDocument(
            plugin_name="my-plugin",
            description="A plugin",
            skills=["my-skill"],
            mcp_servers=[],
        )
    )
    store.set_skill_state = AsyncMock(
        return_value=MongoPluginSkillDocument(
            plugin_name="my-plugin",
            skill_name="my-skill",
            path="my-plugin/skills/my-skill/SKILL.md",
            description="A test skill",
            content="# Skill content",
            author="user-1",
            state="published",
        )
    )
    store.delete_skill = AsyncMock(return_value=True)
    store.delete_resource = AsyncMock(return_value=True)
    store.delete_script = AsyncMock(return_value=True)
    store.get_skill_details = AsyncMock(
        return_value=SkillDetails(
            summary=SkillSummary(
                name="my-skill",
                description="A test skill",
                plugin_name="my-plugin",
            ),
            content="# Skill content",
        )
    )
    store.read_resource = AsyncMock(return_value='{"key": "value"}')
    store.read_script = AsyncMock(return_value="print('hello')")
    store.load_snapshot = AsyncMock()
    store.load_shared_snapshot = AsyncMock()
    store.list_resource_names = AsyncMock(return_value=["data.json"])
    store.list_script_names = AsyncMock(return_value=["run.py"])
    store.record_skill_usage = AsyncMock()
    store.get_skill_usage_count = AsyncMock(return_value=5)
    store.get_skill_usage_counts = AsyncMock(return_value={"my-skill": 5})
    store.plugin_exists = AsyncMock(return_value=False)
    store.list_plugins = AsyncMock(return_value=[])
    store.has_plugins = AsyncMock(return_value=True)
    return store


@pytest.fixture
def history_writer() -> AsyncMock:
    writer = AsyncMock()
    writer.ensure_indexes = AsyncMock()
    writer.write_skill_history = AsyncMock()
    writer.write_script_history = AsyncMock()
    writer.write_reference_history = AsyncMock()
    writer.write_plugin_history = AsyncMock()
    writer.get_skill_history = AsyncMock(return_value=[])
    writer.get_script_history = AsyncMock(return_value=[])
    writer.get_resource_history = AsyncMock(return_value=[])
    writer.get_plugin_history = AsyncMock(return_value=[])
    return writer


@pytest.fixture
def store(*, inner_store: AsyncMock, history_writer: AsyncMock) -> HistoryTrackingPluginSkillStore:
    return HistoryTrackingPluginSkillStore(inner_store=inner_store, history_writer=history_writer)


# ------------------------------------------------------------------
# ensure_indexes
# ------------------------------------------------------------------


async def test_ensure_indexes_delegates_to_both(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    await store.ensure_indexes()

    inner_store.ensure_indexes.assert_awaited_once()
    history_writer.ensure_indexes.assert_awaited_once()


# ------------------------------------------------------------------
# save_skill
# ------------------------------------------------------------------


async def test_save_skill_new_records_created_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.skill_exists.return_value = False

    await store.save_skill(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        content="# Skill content",
        modified_by="editor-1",
    )

    history_writer.write_skill_history.assert_awaited_once()
    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.action == "created"
    assert record.changed_by == "editor-1"
    assert record.source_collection == "plugin_skills"
    assert record.user_id == "user-1"
    assert record.plugin_name == "my-plugin"
    assert record.skill_name == "my-skill"


async def test_save_skill_existing_records_updated_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.skill_exists.return_value = True

    await store.save_skill(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        content="# Updated",
        modified_by="editor-1",
    )

    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.action == "updated"


async def test_save_skill_uses_author_when_no_modified_by(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.skill_exists.return_value = False

    await store.save_skill(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        content="# Content",
    )

    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.changed_by == "user-1"


# ------------------------------------------------------------------
# set_skill_state
# ------------------------------------------------------------------


async def test_set_skill_state_published_records_state_changed(
    store: HistoryTrackingPluginSkillStore, history_writer: AsyncMock
) -> None:
    await store.set_skill_state(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        state="published",
    )

    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.action == "state_changed"
    assert record.changed_by == "user-1"


async def test_set_skill_state_personal_records_state_changed(
    store: HistoryTrackingPluginSkillStore, history_writer: AsyncMock
) -> None:
    await store.set_skill_state(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        state="draft",
    )

    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.action == "state_changed"


# ------------------------------------------------------------------
# delete_skill
# ------------------------------------------------------------------


async def test_delete_skill_records_deleted_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    result = await store.delete_skill(author="user-1", plugin_name="my-plugin", skill_name="my-skill")

    assert result is True
    inner_store.delete_skill.assert_awaited_once_with(author="user-1", plugin_name="my-plugin", skill_name="my-skill")
    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.action == "deleted"
    assert record.document_snapshot["skill_name"] == "my-skill"


async def test_delete_skill_no_history_when_not_deleted(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.delete_skill.return_value = False

    result = await store.delete_skill(author="user-1", plugin_name="my-plugin", skill_name="my-skill")

    assert result is False
    history_writer.write_skill_history.assert_not_awaited()


async def test_delete_skill_snapshot_failure_still_deletes(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.get_skill_details.side_effect = Exception("not found")

    result = await store.delete_skill(author="user-1", plugin_name="my-plugin", skill_name="my-skill")

    assert result is True
    record = history_writer.write_skill_history.call_args.kwargs["record"]
    assert record.action == "deleted"
    assert record.document_snapshot == {}


async def test_delete_skill_records_cascade_deleted_children(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.list_resource_names.return_value = ["data.json", "config.yaml"]
    inner_store.list_script_names.return_value = ["run.py"]

    await store.delete_skill(author="user-1", plugin_name="my-plugin", skill_name="my-skill")

    assert history_writer.write_reference_history.await_count == 2
    assert history_writer.write_script_history.await_count == 1
    assert history_writer.write_skill_history.await_count == 1

    ref_record_1 = history_writer.write_reference_history.call_args_list[0].kwargs["record"]
    assert ref_record_1.action == "deleted"
    assert ref_record_1.resource_name == "data.json"

    script_record = history_writer.write_script_history.call_args_list[0].kwargs["record"]
    assert script_record.action == "deleted"
    assert script_record.script_name == "run.py"


# ------------------------------------------------------------------
# save_resource
# ------------------------------------------------------------------


async def test_save_resource_new_records_created_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.resource_exists.return_value = False

    await store.save_resource(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        resource_name="data.json",
        content='{"key": "value"}',
        modified_by="editor-1",
    )

    history_writer.write_reference_history.assert_awaited_once()
    record = history_writer.write_reference_history.call_args.kwargs["record"]
    assert record.action == "created"
    assert record.source_collection == "plugin_references"
    assert record.resource_name == "data.json"


async def test_save_resource_existing_records_updated_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.resource_exists.return_value = True

    await store.save_resource(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        resource_name="data.json",
        content='{"key": "new"}',
    )

    record = history_writer.write_reference_history.call_args.kwargs["record"]
    assert record.action == "updated"


# ------------------------------------------------------------------
# delete_resource
# ------------------------------------------------------------------


async def test_delete_resource_records_deleted_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    result = await store.delete_resource(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", resource_name="data.json"
    )

    assert result is True
    record = history_writer.write_reference_history.call_args.kwargs["record"]
    assert record.action == "deleted"
    assert record.document_snapshot["resource_name"] == "data.json"


# ------------------------------------------------------------------
# save_script
# ------------------------------------------------------------------


async def test_save_script_new_records_created_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.script_exists.return_value = False

    await store.save_script(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        script_name="run.py",
        content="print('hello')",
        modified_by="editor-1",
    )

    history_writer.write_script_history.assert_awaited_once()
    record = history_writer.write_script_history.call_args.kwargs["record"]
    assert record.action == "created"
    assert record.source_collection == "plugin_scripts"
    assert record.script_name == "run.py"


async def test_save_script_existing_records_updated_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.script_exists.return_value = True

    await store.save_script(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        script_name="run.py",
        content="print('updated')",
    )

    record = history_writer.write_script_history.call_args.kwargs["record"]
    assert record.action == "updated"


# ------------------------------------------------------------------
# delete_script
# ------------------------------------------------------------------


async def test_delete_script_records_deleted_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    result = await store.delete_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )

    assert result is True
    record = history_writer.write_script_history.call_args.kwargs["record"]
    assert record.action == "deleted"
    assert record.document_snapshot["script_name"] == "run.py"


# ------------------------------------------------------------------
# save_plugin
# ------------------------------------------------------------------


async def test_save_plugin_new_records_created_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.plugin_exists.return_value = False

    await store.save_plugin(
        plugin_name="my-plugin",
        description="A plugin",
        skills=["my-skill"],
        mcp_servers=[],
    )

    history_writer.write_plugin_history.assert_awaited_once()
    record = history_writer.write_plugin_history.call_args.kwargs["record"]
    assert record.action == "created"
    assert record.source_collection == "plugins"
    assert record.plugin_name == "my-plugin"


async def test_save_plugin_existing_records_updated_action(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.plugin_exists.return_value = True

    await store.save_plugin(
        plugin_name="my-plugin",
        description="A plugin",
        skills=["my-skill"],
        mcp_servers=[],
    )

    record = history_writer.write_plugin_history.call_args.kwargs["record"]
    assert record.action == "updated"


# ------------------------------------------------------------------
# Read pass-through (no history written)
# ------------------------------------------------------------------


async def test_load_snapshot_passes_through(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    await store.load_snapshot(author="user-1", plugin_name="my-plugin")

    inner_store.load_snapshot.assert_awaited_once_with(author="user-1", plugin_name="my-plugin", include_staging=False)
    history_writer.write_skill_history.assert_not_awaited()


async def test_skill_exists_passes_through(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    inner_store.skill_exists.return_value = True

    result = await store.skill_exists(author="user-1", plugin_name="my-plugin", skill_name="my-skill")

    assert result is True
    history_writer.write_skill_history.assert_not_awaited()


async def test_read_resource_passes_through(
    store: HistoryTrackingPluginSkillStore, inner_store: AsyncMock, history_writer: AsyncMock
) -> None:
    result = await store.read_resource(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", resource_name="data.json"
    )

    assert result == '{"key": "value"}'
    history_writer.write_reference_history.assert_not_awaited()


# ------------------------------------------------------------------
# History query pass-through
# ------------------------------------------------------------------


async def test_get_skill_history_delegates_to_writer(
    store: HistoryTrackingPluginSkillStore, history_writer: AsyncMock
) -> None:
    await store.get_skill_history(user_id="user-1", plugin_name="my-plugin", skill_name="my-skill")

    history_writer.get_skill_history.assert_awaited_once_with(
        user_id="user-1", plugin_name="my-plugin", skill_name="my-skill", limit=50, offset=0
    )


# ------------------------------------------------------------------
# Error recording
# ------------------------------------------------------------------


async def test_save_skill_failure_records_error(inner_store: AsyncMock, history_writer: AsyncMock) -> None:
    error_writer = AsyncMock()
    error_writer.write_error = AsyncMock()
    store_with_errors = HistoryTrackingPluginSkillStore(
        inner_store=inner_store, history_writer=history_writer, error_writer=error_writer
    )
    inner_store.save_skill.side_effect = RuntimeError("connection lost")

    with pytest.raises(RuntimeError, match="connection lost"):
        await store_with_errors.save_skill(
            author="user-1",
            plugin_name="my-plugin",
            skill_name="my-skill",
            content="# Content",
        )

    error_writer.write_error.assert_awaited_once()
    error_record = error_writer.write_error.call_args.kwargs["record"]
    assert error_record.operation == "save"
    assert error_record.error_type == "RuntimeError"
    assert "connection lost" in error_record.error_message


async def test_set_skill_state_failure_records_error(inner_store: AsyncMock, history_writer: AsyncMock) -> None:
    error_writer = AsyncMock()
    error_writer.write_error = AsyncMock()
    store_with_errors = HistoryTrackingPluginSkillStore(
        inner_store=inner_store, history_writer=history_writer, error_writer=error_writer
    )
    inner_store.set_skill_state.side_effect = RuntimeError("state change failed")

    with pytest.raises(RuntimeError, match="state change failed"):
        await store_with_errors.set_skill_state(
            author="user-1",
            plugin_name="my-plugin",
            skill_name="my-skill",
            state="published",
        )

    error_record = error_writer.write_error.call_args.kwargs["record"]
    assert error_record.operation == "state_change"
