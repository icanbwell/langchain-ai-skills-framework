"""Capability interface for publishing skills to a shared marketplace."""

from __future__ import annotations

from typing import Protocol


class MarketplacePublisher(Protocol):
    """Publishes or removes a skill in the shared marketplace.

    ``user_id`` is the skill's author. ``acting_user_id`` is the user who triggered the operation; it is the
    publisher-boundary name for the acting user (the store boundary calls it ``modified_by``).
    ``GitHubMarketplacePublisher`` is the shipped implementation.
    """

    @property
    def use_branch(self) -> bool:
        """True when changes go through a branch and pull request rather than a direct commit."""
        ...

    async def publish_skill(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        skill_content: str,
        resources: dict[str, str],
        scripts: dict[str, str],
        user_id: str,
        branch_name: str | None = None,
        skill_path: str | None = None,
        resource_paths: dict[str, str] | None = None,
        script_paths: dict[str, str] | None = None,
        acting_user_id: str | None = None,
    ) -> str: ...

    async def unpublish_skill(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
        skill_dir: str | None = None,
        acting_user_id: str | None = None,
    ) -> str | None: ...
