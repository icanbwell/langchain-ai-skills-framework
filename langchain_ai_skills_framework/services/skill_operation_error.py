"""Framework-agnostic error raised by skill service operations.

LangChain tools translate this into ``ToolException``; other tool
frameworks can map it to their own error type.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore


class SkillOperationError(Exception):
    """A skill operation failed with a user-facing message."""


def require_user_id(*, user_id: str, operation: str) -> None:
    """Raise if *user_id* is blank or missing."""
    if not user_id:
        raise SkillOperationError(f"user_id is required for {operation}")


def require_non_empty(*, value: str | None, label: str) -> None:
    """Raise if *value* is None, empty, or whitespace-only."""
    if not value or not value.strip():
        raise SkillOperationError(f"{label} must be a non-empty string.")


def require_store(*, store: PluginSkillStore | None) -> PluginSkillStore:
    """Raise if *store* is not configured; return it narrowed to non-None."""
    if store is None:
        raise SkillOperationError("mongo_skill_loader is not configured.")
    return store


@dataclass(frozen=True, slots=True)
class OwnerAndActor:
    """Whose skill is acted on (``owner``) and who is acting (``actor``).

    A pure value object. Services pass ``actor`` to the store as ``modified_by`` and to the
    marketplace publisher as ``acting_user_id``; those names exist only at those boundaries.

    The framework performs no authorization: callers (the consuming service) must
    verify that ``actor`` may act on ``owner``'s skills before passing a different
    ``author`` or ``modified_by``.
    """

    owner: str
    actor: str

    @property
    def acts_for_another_user(self) -> bool:
        """True when the acting user is not the skill's owner."""
        return self.actor != self.owner


def resolve_owner_and_actor(*, user_id: str, author: str | None, modified_by: str | None) -> OwnerAndActor:
    """Resolve ``author``/``modified_by`` (default ``user_id`` only when omitted, never when empty).

    ``author`` is whose skill is acted on (the owner) and ``modified_by`` is the acting user recorded in
    history (and, for publish, the marketplace PR). This function and the services performing the operations
    do no authorization: a caller passing an ``author`` or ``modified_by`` different from ``user_id`` must
    first verify that user may do so.

    Raises:
        SkillOperationError: If an explicitly supplied ``author`` or ``modified_by`` is blank.
    """
    owner = author if author is not None else user_id
    actor = modified_by if modified_by is not None else user_id
    require_non_empty(value=owner, label="author")
    require_non_empty(value=actor, label="modified_by")
    return OwnerAndActor(owner=owner, actor=actor)


def acting_kwargs(owner_and_actor: OwnerAndActor, *, key: str) -> dict[str, str]:
    """``{key: actor}`` only when acting for another user, else empty.

    The single place for the rule "pass ``modified_by`` / ``acting_user_id`` only when the actor differs from the
    owner", so stores and publishers that predate acting-user support keep receiving their original call shape.
    """
    return {key: owner_and_actor.actor} if owner_and_actor.acts_for_another_user else {}
