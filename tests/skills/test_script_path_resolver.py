"""Tests for resolve_script_file_path."""

from __future__ import annotations

from pathlib import Path

from langchain_ai_skills_framework.loaders.script_path_resolver import resolve_script_file_path


def test_resolves_py_script_in_scripts_dir(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('hi')")

    result = resolve_script_file_path(skill_dir=tmp_path, script_name="run")

    assert result == scripts_dir / "run.py"


def test_resolves_sh_script_when_py_absent(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.sh").write_text("echo hi")

    result = resolve_script_file_path(skill_dir=tmp_path, script_name="run")

    assert result == scripts_dir / "run.sh"


def test_falls_back_to_legacy_skill_root_layout(tmp_path: Path) -> None:
    (tmp_path / "run.py").write_text("print('hi')")

    result = resolve_script_file_path(skill_dir=tmp_path, script_name="run")

    assert result == tmp_path / "run.py"


def test_returns_none_when_not_found(tmp_path: Path) -> None:
    result = resolve_script_file_path(skill_dir=tmp_path, script_name="missing")

    assert result is None
