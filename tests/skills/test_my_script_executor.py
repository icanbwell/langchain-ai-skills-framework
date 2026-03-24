from __future__ import annotations

from pathlib import Path

import pytest

from langchain_ai_skills_framework.executors.my_script_executor import (
    MyScriptExecutor,
    PathSecurityError,
)


@pytest.mark.parametrize("as_absolute", [False, True])
def test_validate_path_accepts_paths_within_skill_directory(
    tmp_path: Path,
    as_absolute: bool,
) -> None:
    skill_base_dir = tmp_path / "skill"
    scripts_dir = skill_base_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    script_file = scripts_dir / "extract.py"
    script_file.write_text("print('ok')\n", encoding="utf-8")

    executor = MyScriptExecutor()
    script_path = script_file if as_absolute else Path("scripts/extract.py")

    resolved = executor._validate_path(
        script_path=script_path, skill_base_dir=skill_base_dir
    )

    assert resolved == script_file.resolve()


def test_validate_path_rejects_relative_traversal_outside_skill_directory(
    tmp_path: Path,
) -> None:
    skill_base_dir = tmp_path / "skill"
    scripts_dir = skill_base_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "extract.py").write_text("print('ok')\n", encoding="utf-8")

    outside_script = tmp_path / "outside.py"
    outside_script.write_text("print('outside')\n", encoding="utf-8")

    executor = MyScriptExecutor()

    with pytest.raises(PathSecurityError, match="outside skill directory"):
        executor._validate_path(
            script_path=Path("../outside.py"),
            skill_base_dir=skill_base_dir,
        )
