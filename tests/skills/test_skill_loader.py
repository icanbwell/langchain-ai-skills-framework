from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import pytest

from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillLoaderEnvironmentVariables,
    SkillNotFoundError,
    SkillValidationError,
)
from langchain_ai_skills_framework.cache.skill_cache import SkillCache


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


def _write_skill_raw(root: Path, directory: str, *, content: str) -> None:
    skill_dir = root / directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def _write_skill_frontmatter(
    root: Path,
    directory: str,
    *,
    frontmatter: str,
    body: str = "# Body\n\nDetails.",
) -> None:
    content = f"---\n{frontmatter}\n---\n{body}\n"
    _write_skill_raw(root, directory, content=content)


class FakeEnvironmentVariables(SkillLoaderEnvironmentVariables):
    def __init__(
        self,
        *,
        skills_directory: str,
        excluded_skills: set[str] | None = None,
        excluded_skill_groups: set[str] | None = None,
    ) -> None:
        self._skills_directory = skills_directory
        self._excluded_skills = set(excluded_skills or set())
        self._excluded_skill_groups = set(excluded_skill_groups or set())

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


class _StubFsspecFileSystem:
    def __init__(self, files: dict[str, str]) -> None:
        self._files = dict(files)
        self._directories = self._build_directories(tuple(self._files.keys()))

    @staticmethod
    def _build_directories(files: tuple[str, ...]) -> set[str]:
        directories: set[str] = set()
        for file_path in files:
            current = file_path.rsplit("/", 1)[0]
            while current:
                directories.add(current)
                if "/" not in current:
                    break
                current = current.rsplit("/", 1)[0]
        return directories

    def exists(self, path: str) -> bool:
        return path in self._directories or path in self._files

    def isdir(self, path: str) -> bool:
        return path in self._directories

    def isfile(self, path: str) -> bool:
        return path in self._files

    def ls(self, path: str, detail: bool = True) -> list[dict[str, str]]:
        _ = detail
        prefix = f"{path.rstrip('/')}/"
        children: set[str] = set()
        for directory in self._directories:
            if directory.startswith(prefix):
                remainder = directory[len(prefix) :]
                if remainder and "/" not in remainder:
                    children.add(directory)
        for file_path in self._files:
            if file_path.startswith(prefix):
                remainder = file_path[len(prefix) :]
                if remainder and "/" not in remainder:
                    children.add(file_path)
        entries: list[dict[str, str]] = []
        for child in sorted(children):
            entries.append(
                {
                    "name": child,
                    "type": "directory" if child in self._directories else "file",
                }
            )
        return entries

    def open(self, path: str, mode: str = "rb") -> BinaryIO:
        if mode != "rb":
            raise ValueError(f"Unsupported mode: {mode}")
        return io.BytesIO(self._files[path].encode("utf-8"))


def _create_environment_variables(skills_directory: Path) -> FakeEnvironmentVariables:
    return FakeEnvironmentVariables(skills_directory=str(skills_directory))


def test_skill_loader_reads_metadata_and_content(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    cache = SkillCache()
    loader = SkillDirectoryLoader(
        cache=cache,
        environment_variables=_create_environment_variables(tmp_path),
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill"]

    details = loader.get_skill_details("alpha-skill")
    assert details.content.strip().startswith("# Body")
    assert details.source_path.name == "SKILL.md"


def test_skill_loader_reads_nested_skills(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "category/alpha-skill", name="alpha-skill")
    _write_skill(tmp_path, "beta-skill")
    cache = SkillCache()
    loader = SkillDirectoryLoader(
        cache=cache,
        environment_variables=_create_environment_variables(tmp_path),
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill", "beta-skill"]

    details = loader.get_skill_details("alpha-skill")
    assert details.source_path.parent.name == "alpha-skill"


def test_skill_loader_reads_skills_from_github_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_uri = "github://icanbwell:skill-repo@main/skills"
    github_root = "icanbwell/skill-repo/main/skills"
    filesystem = _StubFsspecFileSystem(
        {
            f"{github_root}/group-one/alpha-skill/SKILL.md": (
                "---\n"
                "name: alpha-skill\n"
                "description: Example description for alpha-skill.\n"
                "---\n"
                "# Body\n\n"
                "Alpha content\n"
            ),
            f"{github_root}/beta-skill/SKILL.md": (
                "---\n"
                "name: beta-skill\n"
                "description: Example description for beta-skill.\n"
                "---\n"
                "# Body\n\n"
                "Beta content\n"
            ),
        }
    )

    def _fake_url_to_fs(uri: str) -> tuple[_StubFsspecFileSystem, str]:
        if uri != github_uri:
            raise AssertionError(f"Unexpected URI: {uri}")
        return filesystem, github_root

    monkeypatch.setattr(
        "langchain_ai_skills_framework.loaders.skill_directory_loader.fsspec.core.url_to_fs",
        _fake_url_to_fs,
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=FakeEnvironmentVariables(skills_directory=github_uri),
    )

    summaries = loader.list_skill_summaries()

    assert [summary.name for summary in summaries] == ["alpha-skill", "beta-skill"]
    assert (
        loader.get_skill_details("alpha-skill").source_path.parent.name == "alpha-skill"
    )


def test_skill_loader_skips_excluded_skills(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    _write_skill(tmp_path, "beta-skill")
    cache = SkillCache()
    environment_variables = FakeEnvironmentVariables(
        skills_directory=str(tmp_path),
        excluded_skills={"beta_skill"},
    )
    loader = SkillDirectoryLoader(
        cache=cache,
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
    cache = SkillCache()
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        cache=cache,
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
    cache = SkillCache()
    environment_variables = _create_environment_variables(tmp_path)
    loader = SkillDirectoryLoader(
        cache=cache,
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
    cache = SkillCache()
    loader = SkillDirectoryLoader(
        cache=cache,
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_details("beta")


def test_skill_loader_allows_directory_name_mismatch(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill-dir", name="alpha-skill")

    cache = SkillCache()
    loader = SkillDirectoryLoader(
        cache=cache,
        environment_variables=_create_environment_variables(tmp_path),
    )

    summaries = loader.list_skill_summaries()

    assert [summary.name for summary in summaries] == ["alpha-skill"]

    details = loader.get_skill_details("alpha-skill")
    assert details.source_path.name == "SKILL.md"


def test_skill_loader_reuses_shared_cache_until_refresh(
    tmp_path: Path,
) -> None:
    shared_cache = SkillCache()
    _write_skill(tmp_path, "alpha-skill", body="Version 1 content")

    loader_a = SkillDirectoryLoader(
        cache=shared_cache,
        environment_variables=_create_environment_variables(tmp_path),
    )
    assert "Version 1" in loader_a.get_skill_details("alpha-skill").content

    _write_skill(tmp_path, "alpha-skill", body="Version 2 content")
    loader_b = SkillDirectoryLoader(
        cache=shared_cache,
        environment_variables=_create_environment_variables(tmp_path),
    )
    # Snapshot should still reflect the cached data
    assert "Version 1" in loader_b.get_skill_details("alpha-skill").content

    loader_b.refresh()
    assert "Version 2" in loader_b.get_skill_details("alpha-skill").content


def test_skill_loader_returns_empty_when_directory_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing"
    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(missing_path),
    )

    summaries = loader.list_skill_summaries()

    assert summaries == ()


def test_skill_loader_rejects_non_directory_path(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "skills.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(file_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


@pytest.mark.parametrize(
    ("content", "case"),
    [
        ("name: alpha\n", "missing-header"),
        ("---\nname: alpha\n", "missing-terminator"),
        ("---\n:bad\n---\nBody", "invalid-yaml"),
        ("---\n- item\n---\nBody", "non-mapping"),
    ],
)
def test_skill_loader_rejects_invalid_frontmatter(
    tmp_path: Path, content: str, case: str
) -> None:
    _write_skill_raw(tmp_path, f"skill-{case}", content=content)

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_validates_required_fields(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter="description: Missing name",
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_rejects_empty_description(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter="name: alpha-skill\ndescription: ''",
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_rejects_invalid_metadata_and_tools(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter=(
            "name: alpha-skill\n"
            "description: Valid description\n"
            "metadata: {1: value}\n"
            "allowed-tools: [tool-a]\n"
        ),
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_rejects_invalid_license_and_compatibility(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter=(
            "name: alpha-skill\n"
            "description: Valid description\n"
            "license: [bad]\n"
            "compatibility: [bad]\n"
        ),
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_parses_allowed_tools(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter=(
            "name: alpha-skill\n"
            "description: Valid description\n"
            "allowed-tools: tool-a tool-b\n"
        ),
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    details = loader.get_skill_details("alpha-skill")

    assert details.summary.allowed_tools == ("tool-a", "tool-b")


def test_skill_loader_rejects_duplicate_normalized_names(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter="name: alpha-skill\ndescription: Primary\n",
    )
    _write_skill_frontmatter(
        tmp_path,
        "alpha_skill",
        frontmatter="name: alpha-skill\ndescription: Duplicate\n",
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_rejects_duplicate_across_nested_and_root(
    tmp_path: Path,
) -> None:
    _write_skill_frontmatter(
        tmp_path,
        "alpha-skill",
        frontmatter="name: alpha-skill\ndescription: Root\n",
    )
    _write_skill_frontmatter(
        tmp_path,
        "category/alpha-skill",
        frontmatter="name: alpha-skill\ndescription: Nested\n",
    )

    loader = SkillDirectoryLoader(
        cache=SkillCache(),
        environment_variables=_create_environment_variables(tmp_path),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()
