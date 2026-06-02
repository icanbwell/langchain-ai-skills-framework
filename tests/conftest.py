"""Shared test fixtures for the langchain-ai-skills-framework test suite."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_mongo_database() -> MagicMock:
    """Mock AsyncIOMotorDatabase that lazily creates collections."""
    db = MagicMock()
    collections: dict[str, AsyncMock] = {}

    def get_collection(name: str) -> AsyncMock:
        if name not in collections:
            col = AsyncMock()
            col.create_index = AsyncMock()
            col.insert_one = AsyncMock()
            cursor = MagicMock()
            cursor.sort = MagicMock(return_value=cursor)
            cursor.skip = MagicMock(return_value=cursor)
            cursor.limit = MagicMock(return_value=cursor)
            cursor.to_list = AsyncMock(return_value=[])
            col.find = MagicMock(return_value=cursor)
            collections[name] = col
        return collections[name]

    db.__getitem__ = MagicMock(side_effect=get_collection)
    return db
