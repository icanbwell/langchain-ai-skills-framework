"""Tests for HistoryWriter."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.models.history_record import HistoryRecord
from langchain_ai_skills_framework.persistence.history_writer import HistoryWriter


@pytest.fixture
def mock_database() -> MagicMock:
    db = MagicMock()
    collections: dict[str, AsyncMock] = {}

    def get_collection(name: str) -> AsyncMock:
        if name not in collections:
            collections[name] = AsyncMock()
        return collections[name]

    db.__getitem__ = MagicMock(side_effect=get_collection)
    return db


@pytest.fixture
def history_writer(mock_database: MagicMock) -> HistoryWriter:
    return HistoryWriter(
        database=mock_database,
        skills_history_collection_name="plugin_skills_history",
        scripts_history_collection_name="plugin_scripts_history",
        references_history_collection_name="plugin_references_history",
        plugins_history_collection_name="plugins_history",
    )


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


async def test_write_skill_history_inserts_document(history_writer: HistoryWriter, mock_database: MagicMock) -> None:
    record = _make_record()
    await history_writer.write_skill_history(record=record)

    collection = mock_database["plugin_skills_history"]
    collection.insert_one.assert_called_once()
    inserted = collection.insert_one.call_args[0][0]
    assert inserted["action"] == "created"
    assert inserted["skill_name"] == "my-skill"


async def test_write_script_history_inserts_document(history_writer: HistoryWriter, mock_database: MagicMock) -> None:
    record = _make_record(action="updated", source="plugin_scripts", script_name="run.py")
    await history_writer.write_script_history(record=record)

    collection = mock_database["plugin_scripts_history"]
    collection.insert_one.assert_called_once()


async def test_write_reference_history_inserts_document(
    history_writer: HistoryWriter, mock_database: MagicMock
) -> None:
    record = _make_record(action="deleted", source="plugin_references", resource_name="data.json")
    await history_writer.write_reference_history(record=record)

    collection = mock_database["plugin_references_history"]
    collection.insert_one.assert_called_once()


async def test_write_plugin_history_inserts_document(history_writer: HistoryWriter, mock_database: MagicMock) -> None:
    record = _make_record(action="updated", source="plugins")
    await history_writer.write_plugin_history(record=record)

    collection = mock_database["plugins_history"]
    collection.insert_one.assert_called_once()


async def test_get_skill_history_returns_records(history_writer: HistoryWriter, mock_database: MagicMock) -> None:
    now = datetime.now(timezone.utc)
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
    collection = mock_database["plugin_skills_history"]
    collection.find = MagicMock(return_value=mock_cursor)

    results = await history_writer.get_skill_history(user_id="user-1", plugin_name="p", skill_name="s")

    assert len(results) == 1
    assert results[0].action == "created"


async def test_write_skill_history_logs_warning_on_failure(
    history_writer: HistoryWriter, mock_database: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    record = _make_record()
    collection = mock_database["plugin_skills_history"]
    collection.insert_one.side_effect = Exception("connection lost")

    with caplog.at_level("WARNING"):
        await history_writer.write_skill_history(record=record)

    assert "Failed to write skill history" in caplog.text
