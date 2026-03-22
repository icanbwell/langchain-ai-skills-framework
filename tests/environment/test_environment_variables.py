import pytest

from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, 3600),
        ("120", 120),
        ("abc", 3600),
        ("0", 3600),
        ("-10", 3600),
    ],
)
def test_skills_cache_timeout_seconds(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str | None,
    expected: int,
) -> None:
    env_vars = LangchainAISkillsFrameworkEnvironmentVariables()

    if raw_value is None:
        monkeypatch.delenv("SKILLS_CACHE_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SKILLS_CACHE_TIMEOUT_SECONDS", raw_value)

    assert env_vars.skills_cache_timeout_seconds == expected
