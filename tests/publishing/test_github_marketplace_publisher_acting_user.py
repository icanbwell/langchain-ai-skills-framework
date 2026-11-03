"""PR body names the acting user and the author when they differ (BAI-1500)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    MAX_PR_BODY_ID_LENGTH,
    GitHubMarketplacePublisher,
)

OWNER = "author-a@example.com"
ADMIN = "admin@example.com"

BodyFn = Callable[[GitHubMarketplacePublisher, str | None], Awaitable[str]]


def _publisher() -> GitHubMarketplacePublisher:
    return GitHubMarketplacePublisher(token_provider=MagicMock(), repo="org/repo", use_branch=True)


async def _publish_body(publisher: GitHubMarketplacePublisher, acting: str | None) -> str:
    with patch.object(
        publisher, "_create_or_update_pr", new_callable=AsyncMock, return_value="https://example.test/pr/1"
    ) as create_pr:
        await publisher.publish_skill(
            plugin_name="p",
            skill_name="s",
            skill_content="c",
            resources={},
            scripts={},
            user_id=OWNER,
            acting_user_id=acting,
        )
    return str(create_pr.call_args.kwargs["pr_body"])


async def _unpublish_body(publisher: GitHubMarketplacePublisher, acting: str | None) -> str:
    with patch.object(
        publisher, "_create_or_update_removal_pr", new_callable=AsyncMock, return_value="https://example.test/pr/2"
    ) as remove_pr:
        await publisher.unpublish_skill(plugin_name="p", skill_name="s", user_id=OWNER, acting_user_id=acting)
    return str(remove_pr.call_args.kwargs["pr_body"])


BODY_FNS = (
    pytest.param(_publish_body, "Published by", id="publish"),
    pytest.param(_unpublish_body, "Unpublished by", id="unpublish"),
)


@pytest.mark.parametrize(("body_fn", "label"), BODY_FNS)
async def test_body_names_actor_and_author_when_they_differ(body_fn: BodyFn, label: str) -> None:
    body = await body_fn(_publisher(), ADMIN)

    assert f"- **{label}**: {ADMIN}\n" in body
    assert f"- **Author**: {OWNER}\n" in body


@pytest.mark.parametrize("acting", [None, OWNER])
@pytest.mark.parametrize(("body_fn", "label"), BODY_FNS)
async def test_body_unchanged_when_actor_matches_or_is_omitted(body_fn: BodyFn, label: str, acting: str | None) -> None:
    body = await body_fn(_publisher(), acting)

    assert f"- **{label}**: {OWNER}\n" in body
    assert "**Author**" not in body


@pytest.mark.parametrize(
    ("hostile", "expected"),
    [
        ("admin@example.com\n## Approved by security", "admin@example.com ## Approved by security"),
        ("admin\r\n- **Author**: ceo@example.com", "admin - **Author**: ceo@example.com"),
        ("<img src=x onerror=alert(1)>", "&lt;img src=x onerror=alert(1)&gt;"),
        ("[click](https://evil.example)", "\\[click\\](https://evil.example)"),
    ],
)
@pytest.mark.parametrize(("body_fn", "label"), BODY_FNS)
async def test_hostile_acting_user_cannot_inject_lines_or_markup(
    body_fn: BodyFn, label: str, hostile: str, expected: str
) -> None:
    body = await body_fn(_publisher(), hostile)

    assert f"- **{label}**: {expected}\n- **Author**: {OWNER}\n" in body
    assert "\n## " not in body  # no heading smuggled in after the template's own first line


async def test_overlong_acting_user_is_truncated() -> None:
    body = await _publish_body(_publisher(), "a" * (MAX_PR_BODY_ID_LENGTH * 5))

    assert "a" * MAX_PR_BODY_ID_LENGTH in body
    assert "a" * (MAX_PR_BODY_ID_LENGTH + 1) not in body
