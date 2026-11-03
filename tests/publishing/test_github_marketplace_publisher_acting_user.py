"""PR body names the acting user and the author when they differ (BAI-1500)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.publishing.github_marketplace_publisher import GitHubMarketplacePublisher

OWNER = "author-a@example.com"
ADMIN = "admin@example.com"


def _publisher() -> GitHubMarketplacePublisher:
    publisher = GitHubMarketplacePublisher(token_provider=MagicMock(), repo="org/repo", use_branch=True)
    publisher._create_or_update_pr = AsyncMock(return_value="https://example.test/pr/1")  # type: ignore[method-assign]
    publisher._create_or_update_removal_pr = AsyncMock(return_value="https://example.test/pr/2")  # type: ignore[method-assign]
    return publisher


async def _publish_body(publisher: GitHubMarketplacePublisher, acting: str | None) -> str:
    await publisher.publish_skill(
        plugin_name="p",
        skill_name="s",
        skill_content="c",
        resources={},
        scripts={},
        user_id=OWNER,
        acting_user_id=acting,
    )
    return str(publisher._create_or_update_pr.call_args.kwargs["pr_body"])  # type: ignore[attr-defined]


async def _unpublish_body(publisher: GitHubMarketplacePublisher, acting: str | None) -> str:
    await publisher.unpublish_skill(plugin_name="p", skill_name="s", user_id=OWNER, acting_user_id=acting)
    return str(publisher._create_or_update_removal_pr.call_args.kwargs["pr_body"])  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("body_fn", "label"),
    [(_publish_body, "Published by"), (_unpublish_body, "Unpublished by")],
)
async def test_body_names_actor_and_author_when_they_differ(body_fn: object, label: str) -> None:
    body = await body_fn(_publisher(), ADMIN)  # type: ignore[operator]

    assert f"- **{label}**: {ADMIN}\n" in body
    assert f"- **Author**: {OWNER}\n" in body


@pytest.mark.parametrize("acting", [None, OWNER])
@pytest.mark.parametrize(
    ("body_fn", "label"),
    [(_publish_body, "Published by"), (_unpublish_body, "Unpublished by")],
)
async def test_body_unchanged_when_actor_matches_or_is_omitted(body_fn: object, label: str, acting: str | None) -> None:
    body = await body_fn(_publisher(), acting)  # type: ignore[operator]

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
@pytest.mark.parametrize("body_fn", [_publish_body, _unpublish_body])
async def test_hostile_acting_user_cannot_inject_lines_or_markup(body_fn: object, hostile: str, expected: str) -> None:
    body = await body_fn(_publisher(), hostile)  # type: ignore[operator]

    assert f": {expected}\n- **Author**: {OWNER}\n" in body
    assert "\n## " not in body  # no heading smuggled in after the template's own first line


async def test_overlong_acting_user_is_truncated() -> None:
    body = await _publish_body(_publisher(), "a" * 5000)

    assert "a" * 201 not in body
