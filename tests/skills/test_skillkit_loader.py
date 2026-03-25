from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest
import yaml

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.github_skill_downloader import (
    GithubSkillDownloader,
)
from langchain_ai_skills_framework.loaders.skill_loader_environment_variables import (
    SkillLoaderEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.skillkit_directory_loader import (
    SkillkitDirectoryLoader,
)
import langchain_ai_skills_framework.loaders.skillkit_directory_loader as skillkit_directory_loader_module


def _write_skill(
    root: Path,
    directory: str,
    *,
    name: str | None = None,
    description: str | None = None,
    extra_frontmatter: Mapping[str, object] | None = None,
    body: str | None = None,
) -> None:
    skill_name = name or Path(directory).name
    skill_description = description or f"Example description for {skill_name}."
    skill_dir = root / directory
    skill_dir.mkdir(parents=True, exist_ok=True)

    frontmatter: dict[str, object] = {
        "name": skill_name,
        "description": skill_description,
    }
    if extra_frontmatter:
        frontmatter.update(dict(extra_frontmatter))

    frontmatter_text = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    body_text = body or f"Body for {skill_name}."
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter_text}\n---\n# Body\n\n{body_text}\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class _FakeSkillMetadata:
    name: str
    description: str
    skill_path: Path
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakeSkill:
    metadata: _FakeSkillMetadata
    content: str


class _FakeSkillManager:
    def __init__(self, skills_root: Path) -> None:
        self._skills_root = skills_root
        self._metadata_by_name: dict[str, _FakeSkillMetadata] = {}
        self.discover_calls = 0

    def discover(self) -> None:
        self.discover_calls += 1
        self._metadata_by_name = {}
        for skill_file in sorted(self._skills_root.rglob("SKILL.md")):
            content = skill_file.read_text(encoding="utf-8")
            frontmatter = self._parse_frontmatter(content)
            if not isinstance(frontmatter, Mapping):
                continue
            name = frontmatter.get("name")
            description = frontmatter.get("description")
            if not isinstance(name, str) or not isinstance(description, str):
                continue
            allowed_tools = self._parse_allowed_tools(frontmatter.get("allowed-tools"))
            self._metadata_by_name[name] = _FakeSkillMetadata(
                name=name,
                description=description,
                skill_path=skill_file,
                allowed_tools=allowed_tools,
            )

    def list_skills(
        self, include_qualified: bool = False
    ) -> tuple[_FakeSkillMetadata, ...]:
        del include_qualified
        metadata_values = tuple(self._metadata_by_name.values())
        return metadata_values

    def load_skill(self, name: str) -> _FakeSkill:
        metadata = self._metadata_by_name[name]
        return _FakeSkill(
            metadata=metadata,
            content=metadata.skill_path.read_text(encoding="utf-8"),
        )

    @staticmethod
    def _parse_frontmatter(content: str) -> object:
        if not content.startswith("---\n"):
            return None
        _, _, rest = content.partition("---\n")
        frontmatter_text, _, _ = rest.partition("\n---\n")
        return yaml.safe_load(frontmatter_text)

    @staticmethod
    def _parse_allowed_tools(value: object) -> tuple[str, ...]:
        if not isinstance(value, str):
            return ()
        return tuple(item for item in value.split() if item)


class FakeEnvironmentVariables(SkillLoaderEnvironmentVariables):
    def __init__(
        self,
        *,
        skills_directory: str,
        excluded_skills: set[str] | None = None,
        excluded_skill_groups: set[str] | None = None,
        skills_cache_timeout_seconds: int = 3600,
    ) -> None:
        self._skills_directory = skills_directory
        self._excluded_skills = set(excluded_skills or set())
        self._excluded_skill_groups = set(excluded_skill_groups or set())
        self._skills_cache_timeout_seconds = skills_cache_timeout_seconds

    @property
    def skills_directory(self) -> str:
        return self._skills_directory

    @property
    def excluded_skills(self) -> set[str]:
        return set(self._excluded_skills)

    @property
    def excluded_skill_groups(self) -> set[str]:
        return set(self._excluded_skill_groups)

    @property
    def skills_github_token(self) -> str | None:
        return None

    @property
    def skills_cache_timeout_seconds(self) -> int:
        return self._skills_cache_timeout_seconds


def _build_loader(
    monkeypatch: pytest.MonkeyPatch,
    skills_root: Path,
    *,
    excluded_skills: set[str] | None = None,
    excluded_skill_groups: set[str] | None = None,
) -> tuple[SkillkitDirectoryLoader, _FakeSkillManager]:
    manager = _FakeSkillManager(skills_root)
    monkeypatch.setattr(
        skillkit_directory_loader_module,
        "SkillMetadata",
        _FakeSkillMetadata,
    )
    monkeypatch.setattr(
        SkillkitDirectoryLoader, "_create_manager", lambda self: manager
    )

    loader = SkillkitDirectoryLoader(
        environment_variables=FakeEnvironmentVariables(
            skills_directory=str(skills_root),
            excluded_skills=excluded_skills,
            excluded_skill_groups=excluded_skill_groups,
        ),
        github_skill_downloader=GithubSkillDownloader(),
    )
    return loader, manager


@pytest.mark.asyncio
async def test_skillkit_loader_reads_metadata_content_and_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path,
        "alpha-skill",
        extra_frontmatter={
            "license": "Apache-2.0",
            "compatibility": "v1",
            "allowed-tools": "search_tool load_skill",
            "metadata": {"owner": "platform", "priority": 1},
        },
    )
    loader, _ = _build_loader(monkeypatch, tmp_path)

    summaries = loader.list_skill_summaries(allowed_skills=set())

    assert [summary.name for summary in summaries] == ["alpha-skill"]
    assert summaries[0].allowed_tools == ("search_tool", "load_skill")
    assert isinstance(summaries[0].metadata, Mapping)
    assert summaries[0].license is None
    assert summaries[0].compatibility is None

    details = loader.get_skill_details(skill_name="alpha-skill")
    assert details.content == ""

    instructions = await loader.get_instructions()
    assert "<available_skills>" in instructions
    assert "alpha-skill" in instructions
    assert (
        "Use `read_skill_resource` to read files referenced by the skill"
        in instructions
    )
    assert "Use `run_skill_script` to run scripts provided by the skill" in instructions
    assert (
        "Use `run_inline_skill_script` to run inline script content in a skill context"
        in instructions
    )


def test_skillkit_loader_ignores_non_string_allowed_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(
        tmp_path,
        "alpha-skill",
        extra_frontmatter={"allowed-tools": ["search_tool"]},
    )
    loader, _ = _build_loader(monkeypatch, tmp_path)

    summaries = loader.list_skill_summaries(allowed_skills=set())

    assert [summary.name for summary in summaries] == ["alpha-skill"]
    assert summaries[0].allowed_tools == ()


def test_skillkit_loader_applies_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "group_one/alpha-skill", name="alpha-skill")
    _write_skill(tmp_path, "group-two/beta-skill", name="beta-skill")
    _write_skill(tmp_path, "gamma-skill")

    loader, _ = _build_loader(
        monkeypatch,
        tmp_path,
        excluded_skills={"gamma-skill"},
        excluded_skill_groups={"group-one"},
    )

    summaries = loader.list_skill_summaries(allowed_skills=set())

    assert [summary.name for summary in summaries] == ["beta-skill"]


def test_skillkit_loader_refresh_and_missing_skill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    loader, manager = _build_loader(monkeypatch, tmp_path)

    loader.list_skill_summaries(allowed_skills=set())
    assert manager.discover_calls == 1

    loader.refresh()
    assert manager.discover_calls == 2

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_details(skill_name="missing")


def test_skillkit_loader_registers_inline_script_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    loader, _ = _build_loader(monkeypatch, tmp_path)

    tool_names = {tool.name for tool in loader.get_tools()}

    assert "run_inline_skill_script" in tool_names
