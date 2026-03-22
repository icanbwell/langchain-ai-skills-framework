from __future__ import annotations

from pathlib import Path
from typing import Mapping, cast

import pytest
from pydantic_ai_skills.types import Skill

from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderEnvironmentVariables,
    SkillNotFoundError,
    SkillValidationError,
)


def _write_skill(
    root: Path,
    directory: str,
    *,
    name: str | None = None,
    body: str | None = None,
) -> None:
    skill_name = name or directory
    skill_dir = root / directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    body_text = body or f"Details for {skill_name}."
    (skill_dir / "SKILL.md").write_text(
        """---
name: {name}
description: Example description for {name}.
license: Apache-2.0
---
# Body

{body}
""".format(name=skill_name, body=body_text),
        encoding="utf-8",
    )


class FakeEnvironmentVariables(SkillLoaderEnvironmentVariables):
    def __init__(
        self,
        *,
        skills_directory: str,
        excluded_skills: set[str] | None = None,
        excluded_skill_groups: set[str] | None = None,
        github_token: str | None = None,
        skills_cache_timeout_seconds: int = 3600,
    ) -> None:
        self._skills_directory = skills_directory
        self._excluded_skills = set(excluded_skills or set())
        self._excluded_skill_groups = set(excluded_skill_groups or set())
        self._github_token = github_token
        self._skills_cache_timeout_seconds = skills_cache_timeout_seconds

    @property
    def skills_directory(self) -> str:
        return self._skills_directory

    @skills_directory.setter
    def skills_directory(self, value: str) -> None:
        self._skills_directory = value

    @property
    def excluded_skills(self) -> set[str]:
        return set(self._excluded_skills)

    @excluded_skills.setter
    def excluded_skills(self, values: set[str]) -> None:
        self._excluded_skills = set(values)

    @property
    def excluded_skill_groups(self) -> set[str]:
        return set(self._excluded_skill_groups)

    @excluded_skill_groups.setter
    def excluded_skill_groups(self, values: set[str]) -> None:
        self._excluded_skill_groups = set(values)

    def set_exclusions(self, values: set[str]) -> None:
        self.excluded_skills = values

    def set_group_exclusions(self, values: set[str]) -> None:
        self.excluded_skill_groups = values

    @property
    def skills_github_token(self) -> str | None:
        return self._github_token

    @property
    def skills_cache_timeout_seconds(self) -> int:
        return self._skills_cache_timeout_seconds


def _create_environment_variables(skills_directory: Path) -> FakeEnvironmentVariables:
    return FakeEnvironmentVariables(skills_directory=str(skills_directory))


def test_skill_loader_reads_metadata_and_content(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill"]

    details = loader.get_skill_details("alpha-skill")
    assert details.content.strip().startswith("# Body")
    assert details.source_path.name == "SKILL.md"


def test_skill_loader_accepts_non_string_metadata_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(environment_variables=environment_variables)

    class _FakeToolset:
        def __init__(self) -> None:
            self.skills: Mapping[str, Skill] = {
                "alpha-skill": Skill(
                    name="alpha-skill",
                    description="Alpha",
                    content="Alpha content",
                    uri=str(tmp_path / "alpha-skill"),
                    metadata={
                        "metadata": {
                            "priority": 1,
                            "enabled": True,
                            "tags": ["intake", "triage"],
                        }
                    },
                )
            }

    monkeypatch.setattr(loader, "_create_toolset", lambda: _FakeToolset())

    details = loader.get_skill_details("alpha-skill")
    assert details.summary.metadata == {
        "priority": 1,
        "enabled": True,
        "tags": ["intake", "triage"],
    }


@pytest.mark.parametrize(
    "raw_metadata",
    [
        cast(dict[str, object], {1: "owner"}),
        cast(dict[str, object], {"owner": "team", 2: "bad"}),
        cast(dict[str, object], {None: "bad"}),
    ],
)
def test_skill_loader_rejects_non_string_metadata_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_metadata: dict[str, object],
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(environment_variables=environment_variables)

    class _FakeToolset:
        def __init__(self) -> None:
            self.skills: Mapping[str, Skill] = {
                "alpha-skill": Skill(
                    name="alpha-skill",
                    description="Alpha",
                    content="Alpha content",
                    uri=str(tmp_path / "alpha-skill"),
                    metadata={"metadata": raw_metadata},
                )
            }

    monkeypatch.setattr(loader, "_create_toolset", lambda: _FakeToolset())

    with pytest.raises(SkillValidationError, match="metadata keys must be strings"):
        loader.list_skill_summaries()


def test_skill_loader_reads_nested_skills(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "category/alpha-skill", name="alpha-skill")
    _write_skill(tmp_path, "beta-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill", "beta-skill"]

    details = loader.get_skill_details("alpha-skill")
    assert details.source_path.parent.name == "alpha-skill"


def test_skill_loader_reads_skills_from_github_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    github_uri = "github://icanbwell/skill-repo/skills?ref=main"
    skills_root = tmp_path / "repo" / "skills"
    alpha_dir = skills_root / "group-one" / "alpha-skill"
    beta_dir = skills_root / "beta-skill"
    alpha_dir.mkdir(parents=True)
    beta_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeGitSkillsRegistry:
        def __init__(
            self,
            repo_url: str,
            *,
            path: str,
            target_dir: Path,
            token: str | None,
            clone_options: object,
            **_: object,
        ) -> None:
            captured["repo_url"] = repo_url
            captured["path"] = path
            captured["target_dir"] = target_dir
            captured["token"] = token
            captured["clone_options"] = clone_options

        def _skills_root(self) -> Path:
            return skills_root

    class _FakeSkillsToolset:
        def __init__(
            self, *, registries: list[object] | None = None, **_: object
        ) -> None:
            assert registries and len(registries) == 1
            self.skills = {
                "alpha-skill": Skill(
                    name="alpha-skill",
                    description="Alpha",
                    content="Alpha content",
                    uri=str(alpha_dir),
                ),
                "beta-skill": Skill(
                    name="beta-skill",
                    description="Beta",
                    content="Beta content",
                    uri=str(beta_dir),
                ),
            }

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.skill_directory_loader.GitSkillsRegistry",
        _FakeGitSkillsRegistry,
    )
    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.skill_directory_loader.SkillsToolset",
        _FakeSkillsToolset,
    )

    environment_variables = FakeEnvironmentVariables(
        skills_directory=github_uri,
        github_token="token-123",
    )
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()

    assert [summary.name for summary in summaries] == ["alpha-skill", "beta-skill"]
    assert captured["repo_url"] == "https://github.com/icanbwell/skill-repo.git"
    assert captured["path"] == "skills"
    assert captured["token"] == "token-123"
    assert (
        loader.get_skill_details("alpha-skill").source_path.parent.name == "alpha-skill"
    )


def test_skill_loader_rejects_github_uri_without_owner() -> None:
    github_uri = "github:///skill-repo/skills?ref=main"

    environment_variables = FakeEnvironmentVariables(
        skills_directory=github_uri,
        github_token="token-123",
    )
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_rejects_unsupported_github_uri_query_parameter() -> None:
    github_uri = "github://icanbwell/skill-repo/skills?branch=main"

    environment_variables = FakeEnvironmentVariables(
        skills_directory=github_uri,
        github_token="token-123",
    )
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_skips_excluded_skills(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    _write_skill(tmp_path, "beta-skill")
    environment_variables = FakeEnvironmentVariables(
        skills_directory=str(tmp_path),
        excluded_skills={"beta_skill"},
    )
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill"]

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_details("beta-skill")


def test_skill_loader_reads_exclusions_from_environment_variables(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    _write_skill(tmp_path, "beta-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill", "beta-skill"]

    environment_variables.set_exclusions({"beta-skill"})
    loader.refresh()

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill"]


def test_skill_loader_skips_excluded_skill_groups(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "group_one/alpha-skill", name="alpha-skill")
    _write_skill(tmp_path, "group-two/beta-skill", name="beta-skill")
    _write_skill(tmp_path, "gamma-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == [
        "alpha-skill",
        "beta-skill",
        "gamma-skill",
    ]

    environment_variables.set_group_exclusions({"group-one"})
    loader.refresh()

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["beta-skill", "gamma-skill"]


def test_skill_loader_raises_for_missing_skill(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_details("beta")


def test_skill_loader_reloads_toolset_after_ttl_expires(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill", body="Version 1 content")
    environment_variables = FakeEnvironmentVariables(
        skills_directory=str(tmp_path),
        skills_cache_timeout_seconds=1,
    )
    loader = SkillDirectoryLoader(environment_variables=environment_variables)

    class _FakeToolset:
        def __init__(self) -> None:
            self.reload_calls: list[bool] = []
            self.skills: Mapping[str, Skill] = {
                "alpha-skill": Skill(
                    name="alpha-skill",
                    description="Alpha",
                    content="Version 1 content",
                    uri=str(tmp_path / "alpha-skill"),
                )
            }

        def reload(self, *, include_registries: bool = False) -> None:
            self.reload_calls.append(include_registries)
            self.skills = {
                "alpha-skill": Skill(
                    name="alpha-skill",
                    description="Alpha",
                    content="Version 2 content",
                    uri=str(tmp_path / "alpha-skill"),
                )
            }

    fake_toolset = _FakeToolset()
    monkeypatch.setattr(loader, "_create_toolset", lambda: fake_toolset)

    assert "Version 1" in loader.get_skill_details("alpha-skill").content

    loader._snapshot_loaded_at = 0.0
    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.skill_directory_loader.time.monotonic",
        lambda: 5.0,
    )

    assert "Version 2" in loader.get_skill_details("alpha-skill").content
    assert fake_toolset.reload_calls == [True]


def test_skill_loader_returns_empty_when_directory_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing"
    environment_variables = _create_environment_variables(missing_path)
    loader = SkillDirectoryLoader(
        environment_variables=environment_variables,
    )

    summaries = loader.list_skill_summaries()

    assert summaries == ()
