"""Shared fixtures for LangChain tool tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tool_runtime() -> MagicMock:
    """Create a mock ToolRuntime with user-1 in context."""
    runtime = MagicMock()
    runtime.context = {"user_id": "user-1"}
    return runtime


def make_runtime(user_id: str = "user-1") -> MagicMock:
    """Create a mock ToolRuntime with the given user_id in context."""
    runtime = MagicMock()
    runtime.context = {"user_id": user_id}
    return runtime
