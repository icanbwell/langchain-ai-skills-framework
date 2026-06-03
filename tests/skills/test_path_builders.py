"""Tests for materialized path builder functions."""

from __future__ import annotations

import pytest

from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    build_resource_path,
    build_script_path,
    build_skill_path,
)


class TestBuildSkillPath:
    @pytest.mark.parametrize(
        ("plugin_name", "skill_name", "folder", "expected"),
        [
            ("my-plugin", "my-skill", None, "my-plugin/skills/my-skill/SKILL.md"),
            ("my-plugin", "my-skill", "sub", "my-plugin/skills/sub/my-skill/SKILL.md"),
            ("my-plugin", "my-skill", "a/b", "my-plugin/skills/a/b/my-skill/SKILL.md"),
        ],
    )
    def test_build_skill_path(self, *, plugin_name: str, skill_name: str, folder: str | None, expected: str) -> None:
        assert build_skill_path(plugin_name=plugin_name, skill_name=skill_name, folder=folder) == expected


class TestBuildResourcePath:
    @pytest.mark.parametrize(
        ("plugin_name", "skill_name", "resource_name", "folder", "expected"),
        [
            ("p", "s", "data.json", None, "p/skills/s/data.json"),
            ("p", "s", "data.json", "sub", "p/skills/sub/s/data.json"),
        ],
    )
    def test_build_resource_path(
        self, *, plugin_name: str, skill_name: str, resource_name: str, folder: str | None, expected: str
    ) -> None:
        assert (
            build_resource_path(
                plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name, folder=folder
            )
            == expected
        )


class TestBuildScriptPath:
    @pytest.mark.parametrize(
        ("plugin_name", "skill_name", "script_name", "folder", "expected"),
        [
            ("p", "s", "run.py", None, "p/skills/s/scripts/run.py"),
            ("p", "s", "run.py", "sub", "p/skills/sub/s/scripts/run.py"),
        ],
    )
    def test_build_script_path(
        self, *, plugin_name: str, skill_name: str, script_name: str, folder: str | None, expected: str
    ) -> None:
        assert (
            build_script_path(plugin_name=plugin_name, skill_name=skill_name, script_name=script_name, folder=folder)
            == expected
        )
