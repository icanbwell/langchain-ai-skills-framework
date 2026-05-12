"""Framework-agnostic error raised by skill service operations.

LangChain tools translate this into ``ToolException``; other tool
frameworks can map it to their own error type.
"""

from __future__ import annotations

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore


class SkillOperationError(Exception):
    """A skill operation failed with a user-facing message."""


def require_user_id(user_id: str, operation: str) -> None:
    """Raise if *user_id* is blank or missing."""
    if not user_id:
        raise SkillOperationError(f"user_id is required for {operation}")


def require_non_empty(value: str | None, label: str) -> None:
    """Raise if *value* is None, empty, or whitespace-only."""
    if not value or not value.strip():
        raise SkillOperationError(f"{label} must be a non-empty string.")


def require_store(store: PluginSkillStore | None) -> PluginSkillStore:
    """Raise if *store* is not configured; return it narrowed to non-None."""
    if store is None:
        raise SkillOperationError("mongo_skill_loader is not configured.")
    return store
