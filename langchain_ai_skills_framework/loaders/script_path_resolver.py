"""Shared script-file resolution logic (used by SkillSync and MarketplaceDirectoryLoader)."""

from __future__ import annotations

from pathlib import Path


def resolve_script_file_path(*, skill_dir: Path, script_name: str) -> Path | None:
    """Locate a skill script file on disk, trying conventional locations in order.

    Tries ``scripts/<name>.py``, then ``scripts/<name>.sh``, then two legacy
    fallbacks directly in the skill directory root (``<name>.py``, bare ``<name>``).
    """
    candidates = [
        skill_dir / "scripts" / f"{script_name}.py",
        skill_dir / "scripts" / f"{script_name}.sh",
        skill_dir / f"{script_name}.py",
        skill_dir / script_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
