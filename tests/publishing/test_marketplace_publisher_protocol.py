"""``GitHubMarketplacePublisher`` satisfies the ``MarketplacePublisher`` capability interface (checked by mypy)."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_ai_skills_framework.publishing.github_marketplace_publisher import GitHubMarketplacePublisher
from langchain_ai_skills_framework.publishing.marketplace_publisher import MarketplacePublisher


def test_github_publisher_is_a_marketplace_publisher() -> None:
    publisher: MarketplacePublisher = GitHubMarketplacePublisher(token_provider=MagicMock(), repo="org/repo")

    assert publisher.use_branch is True
