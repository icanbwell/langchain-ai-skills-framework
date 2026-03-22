from pathlib import Path

import pytest

from langchain_ai_skills_framework.cache.skill_cache import (
    SkillCache,
    SkillCacheSnapshot,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)


class _FakeSkillCacheEnvironmentVariables:
    def __init__(self, *, skills_cache_timeout_seconds: int) -> None:
        self._skills_cache_timeout_seconds = skills_cache_timeout_seconds

    @property
    def skills_cache_timeout_seconds(self) -> int:
        return self._skills_cache_timeout_seconds


def _snapshot() -> SkillCacheSnapshot:
    summary = SkillSummary(
        name="alpha-skill",
        description="Example description",
        source_path=Path("skills/alpha-skill/SKILL.md"),
    )
    details = SkillDetails(
        summary=summary,
        content="# Body",
        source_path=summary.source_path,
    )
    return SkillCacheSnapshot(
        details_by_name={summary.name: details},
        ordered_summaries=(summary,),
    )


def test_skill_cache_reads_ttl_from_environment_variables() -> None:
    cache = SkillCache(
        environment_variables=_FakeSkillCacheEnvironmentVariables(
            skills_cache_timeout_seconds=60
        )
    )

    cache.set(_snapshot())

    assert cache.is_valid() is True


def test_skill_cache_prefers_explicit_ttl_over_environment_variables() -> None:
    cache = SkillCache(
        ttl_seconds=0,
        environment_variables=_FakeSkillCacheEnvironmentVariables(
            skills_cache_timeout_seconds=60
        ),
    )

    cache.set(_snapshot())

    assert cache.is_valid() is True


def test_skill_cache_ttl_expiration_with_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_time = 100.0

    def _fake_time() -> float:
        return current_time

    monkeypatch.setattr(
        "langchain_ai_skills_framework.cache.skill_cache.time.time",
        _fake_time,
    )

    cache = SkillCache(
        environment_variables=_FakeSkillCacheEnvironmentVariables(
            skills_cache_timeout_seconds=1
        )
    )
    cache.set(_snapshot())

    assert cache.is_valid() is True

    current_time = 102.0
    assert cache.is_valid() is False
