from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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


@dataclass
class _FakeProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@pytest.mark.asyncio
async def test_execute_passes_arguments_as_json_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill_base_dir = tmp_path / "skill"
    scripts_dir = skill_base_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    script_file = scripts_dir / "extract.py"
    script_file.write_text("print('ok')\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    async def _fake_run_process(
        cmd: list[str],
        *,
        check: bool,
        cwd: str,
        input: bytes | None,
        env: dict[str, str],
    ) -> _FakeProcessResult:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        captured["input"] = input
        captured["env"] = env
        return _FakeProcessResult(returncode=0, stdout=b"done\n", stderr=b"")

    monkeypatch.setattr(
        "langchain_ai_skills_framework.executors.my_script_executor.anyio.run_process",
        _fake_run_process,
    )

    executor = MyScriptExecutor()
    arguments = {"file_path": "document.pdf", "page_range": "all"}

    result = await executor.execute_script_from_path(
        script_name="extract.py",
        script_path=Path("scripts/extract.py"),
        arguments=arguments,
        skill_base_dir=skill_base_dir,
        skill_metadata=cast(Any, object()),
        use_uv=False,
    )

    assert result.success is True
    assert result.stdout == "done"
    assert captured["cmd"] == [str(script_file.resolve())]
    assert captured["input"] is not None
    assert json.loads(captured["input"].decode("utf-8")) == arguments


@pytest.mark.asyncio
async def test_execute_rejects_invalid_argument_keys(tmp_path: Path) -> None:
    skill_base_dir = tmp_path / "skill"
    scripts_dir = skill_base_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    script_file = scripts_dir / "extract.py"
    script_file.write_text("print('ok')\n", encoding="utf-8")

    executor = MyScriptExecutor()

    with pytest.raises(ValueError, match="Invalid argument key"):
        await executor.execute_script_from_path(
            script_name="extract.py",
            script_path=Path("scripts/extract.py"),
            arguments={"bad key": "value"},
            skill_base_dir=skill_base_dir,
            skill_metadata=cast(Any, object()),
            use_uv=False,
        )


@pytest.mark.asyncio
async def test_execute_script_runs_inline_script_and_cleans_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skill_base_dir = tmp_path / "skill"
    skill_base_dir.mkdir(parents=True)

    captured: dict[str, Any] = {}

    async def _fake_run_process(
        cmd: list[str],
        *,
        check: bool,
        cwd: str,
        input: bytes | None,
        env: dict[str, str],
    ) -> _FakeProcessResult:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = cwd
        captured["input"] = input
        captured["env"] = env
        return _FakeProcessResult(returncode=0, stdout=b"done\n", stderr=b"")

    monkeypatch.setattr(
        "langchain_ai_skills_framework.executors.my_script_executor.anyio.run_process",
        _fake_run_process,
    )

    executor = MyScriptExecutor()
    arguments = {"file_path": "document.pdf"}

    result = await executor.execute_inline_script(
        script_name="inline.py",
        script="print('ok')\n",
        arguments=arguments,
        skill_base_dir=skill_base_dir,
        skill_metadata=cast(Any, object()),
        use_uv=False,
    )

    assert result.success is True
    assert result.stdout == "done"
    assert captured["input"] is not None
    assert json.loads(captured["input"].decode("utf-8")) == arguments
    assert captured["cmd"]

    temp_script_path = Path(captured["cmd"][0])
    assert temp_script_path.name.startswith(".tmp_skill_script_")
    assert temp_script_path.exists() is False


@pytest.mark.asyncio
async def test_execute_script_rejects_empty_script(tmp_path: Path) -> None:
    skill_base_dir = tmp_path / "skill"
    skill_base_dir.mkdir(parents=True)

    executor = MyScriptExecutor()

    with pytest.raises(ValueError, match="cannot be empty"):
        await executor.execute_inline_script(
            script_name="inline.py",
            script="   ",
            arguments={},
            skill_base_dir=skill_base_dir,
            skill_metadata=cast(Any, object()),
            use_uv=False,
        )
