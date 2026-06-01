"""Tests for ErrorWriter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.models.error_record import ErrorRecord
from langchain_ai_skills_framework.persistence.error_writer import ErrorWriter


@pytest.fixture
def mock_database() -> MagicMock:
    db = MagicMock()
    collection = AsyncMock()
    collection.create_index = AsyncMock()
    collection.insert_one = AsyncMock()
    cursor = AsyncMock()
    cursor.sort = MagicMock(return_value=cursor)
    cursor.skip = MagicMock(return_value=cursor)
    cursor.limit = MagicMock(return_value=cursor)
    cursor.to_list = AsyncMock(return_value=[])
    collection.find = MagicMock(return_value=cursor)
    db.__getitem__ = MagicMock(return_value=collection)
    return db


@pytest.fixture
def writer(*, mock_database: MagicMock) -> ErrorWriter:
    return ErrorWriter(database=mock_database, errors_collection_name="test_errors")


async def test_ensure_indexes_creates_indexes(writer: ErrorWriter, mock_database: MagicMock) -> None:
    collection = mock_database["test_errors"]
    await writer.ensure_indexes()

    assert collection.create_index.await_count == 2


async def test_write_error_inserts_record(writer: ErrorWriter, mock_database: MagicMock) -> None:
    collection = mock_database["test_errors"]
    record = ErrorRecord(
        operation="save",
        error_type="RuntimeError",
        error_message="connection failed",
        user_id="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
    )

    await writer.write_error(record=record)

    collection.insert_one.assert_awaited_once()
    inserted = collection.insert_one.call_args[0][0]
    assert inserted["operation"] == "save"
    assert inserted["error_type"] == "RuntimeError"
    assert inserted["plugin_name"] == "my-plugin"


async def test_write_error_does_not_raise_on_failure(writer: ErrorWriter, mock_database: MagicMock) -> None:
    collection = mock_database["test_errors"]
    collection.insert_one.side_effect = Exception("db down")

    record = ErrorRecord(
        operation="retrieve",
        error_type="TimeoutError",
        error_message="timed out",
    )

    await writer.write_error(record=record)


async def test_get_errors_queries_with_filters(writer: ErrorWriter, mock_database: MagicMock) -> None:
    result = await writer.get_errors(
        user_id="user-1",
        plugin_name="my-plugin",
        operation="save",
    )

    assert result == []
    collection = mock_database["test_errors"]
    collection.find.assert_called_once()
    query = collection.find.call_args[0][0]
    assert query["user_id"] == "user-1"
    assert query["plugin_name"] == "my-plugin"
    assert query["operation"] == "save"
