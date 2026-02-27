from __future__ import annotations

from pathlib import Path

import pytest

from aiskills.skills.loaders.skill_loader import (
    SkillDirectoryLoader,
    SkillNotFoundError,
    SkillValidationError,
)
from aiskills.utilities.cache.skill_cache import SkillCache


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


def test_skill_loader_reads_metadata_and_content(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    cache = SkillCache()
    loader = SkillDirectoryLoader(
        skills_directory=str(tmp_path),
        cache=cache,
    )

    summaries = loader.list_skill_summaries()
    assert [summary.name for summary in summaries] == ["alpha-skill"]

    details = loader.get_skill_details("alpha-skill")
    assert details.content.strip().startswith("# Body")
    assert details.source_path.name == "SKILL.md"


def test_skill_loader_raises_for_missing_skill(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha-skill")
    cache = SkillCache()
    loader = SkillDirectoryLoader(
        skills_directory=str(tmp_path),
        cache=cache,
    )

    with pytest.raises(SkillNotFoundError):
        loader.get_skill_details("beta")


def test_skill_loader_validates_directory_name(
    tmp_path: Path,
) -> None:
    _write_skill(tmp_path, "alpha", name="Alpha")

    cache = SkillCache()
    loader = SkillDirectoryLoader(
        skills_directory=str(tmp_path),
        cache=cache,
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()


def test_skill_loader_reuses_shared_cache_until_refresh(
    tmp_path: Path,
) -> None:
    shared_cache = SkillCache()
    _write_skill(tmp_path, "alpha-skill", body="Version 1 content")

    loader_a = SkillDirectoryLoader(
        skills_directory=str(tmp_path),
        cache=shared_cache,
    )
    assert "Version 1" in loader_a.get_skill_details("alpha-skill").content

    _write_skill(tmp_path, "alpha-skill", body="Version 2 content")
    loader_b = SkillDirectoryLoader(
        skills_directory=str(tmp_path),
        cache=shared_cache,
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
        skills_directory=str(missing_path),
        cache=SkillCache(),
    )

    summaries = loader.list_skill_summaries()

    assert summaries == ()


def test_skill_loader_rejects_non_directory_path(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "skills.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    loader = SkillDirectoryLoader(
        skills_directory=str(file_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
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
        skills_directory=str(tmp_path),
        cache=SkillCache(),
    )

    with pytest.raises(SkillValidationError):
        loader.list_skill_summaries()
