"""Tests for HistoryWriter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.models.history_record import HistoryRecord
from langchain_ai_skills_framework.persistence.history_writer import HistoryWriter


@pytest.fixture
def history_writer(mock_mongo_database: MagicMock) -> HistoryWriter:
    return HistoryWriter(database=mock_mongo_database)


def _make_record(
    *,
    action: Literal["created", "updated", "deleted", "published", "unpublished"] = "created",
    source: str = "plugin_skills",
    skill_name: str = "my-skill",
    resource_name: str | None = None,
    script_name: str | None = None,
) -> HistoryRecord:
    return HistoryRecord(
        action=action,
        document_snapshot={"content": "test"},
        changed_by="user-1",
        source_collection=source,
        user_id="user-1",
        plugin_name="my-plugin",
        skill_name=skill_name,
        resource_name=resource_name,
        script_name=script_name,
    )


@pytest.mark.parametrize(
    ("write_method", "collection_name", "record"),
    [
        ("write_skill_history", "plugin_skills_history", _make_record()),
        (
            "write_script_history",
            "plugin_scripts_history",
            _make_record(action="updated", source="plugin_scripts", script_name="run.py"),
        ),
        (
            "write_reference_history",
            "plugin_references_history",
            _make_record(action="deleted", source="plugin_references", resource_name="data.json"),
        ),
        ("write_plugin_history", "plugins_history", _make_record(action="updated", source="plugins")),
    ],
)
async def test_write_methods_insert_document(
    history_writer: HistoryWriter,
    mock_mongo_database: MagicMock,
    write_method: str,
    collection_name: str,
    record: HistoryRecord,
) -> None:
    await getattr(history_writer, write_method)(record=record)

    collection = mock_mongo_database[collection_name]
    collection.insert_one.assert_called_once()


async def test_get_skill_history_returns_records(history_writer: HistoryWriter, mock_mongo_database: MagicMock) -> None:
    now = datetime.now(UTC)
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.skip.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "action": "created",
                "document_snapshot": {"content": "v1"},
                "changed_by": "user-1",
                "timestamp": now,
                "source_collection": "plugin_skills",
                "user_id": "user-1",
                "plugin_name": "p",
                "skill_name": "s",
            }
        ]
    )
    collection = mock_mongo_database["plugin_skills_history"]
    collection.find = MagicMock(return_value=mock_cursor)

    results = await history_writer.get_skill_history(user_id="user-1", plugin_name="p", skill_name="s")

    assert len(results) == 1
    assert results[0].action == "created"


async def test_write_skill_history_logs_warning_on_failure(
    history_writer: HistoryWriter, mock_mongo_database: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    record = _make_record()
    collection = mock_mongo_database["plugin_skills_history"]
    collection.insert_one.side_effect = Exception("connection lost")

    with caplog.at_level("WARNING"):
        await history_writer.write_skill_history(record=record)

    assert "Failed to write skill history" in caplog.text
