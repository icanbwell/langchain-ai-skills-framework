"""Unit tests for GitHubMarketplacePublisher._build_file_map and _sanitize_override_path.

These cover the path-override behavior introduced to honor custom folder layouts
when publishing skills to the marketplace.
"""

from __future__ import annotations

import pytest

from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)


class TestSanitizeOverridePath:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_returns_none_for_unset(self, *, value: str | None) -> None:
        assert GitHubMarketplacePublisher._sanitize_override_path(value=value, label="x") is None

    def test_strips_surrounding_whitespace(self) -> None:
        assert (
            GitHubMarketplacePublisher._sanitize_override_path(value="  plugins/p/skills/s/SKILL.md  ", label="x")
            == "plugins/p/skills/s/SKILL.md"
        )

    def test_rejects_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            GitHubMarketplacePublisher._sanitize_override_path(value="/etc/passwd", label="skill_path")

    def test_rejects_parent_segment(self) -> None:
        with pytest.raises(ValueError, match=r"'\.\.'"):
            GitHubMarketplacePublisher._sanitize_override_path(value="plugins/../etc/passwd", label="skill_path")


class TestBuildFileMap:
    def test_default_layout_when_no_overrides(self) -> None:
        files = GitHubMarketplacePublisher._build_file_map(
            plugin_name="p",
            skill_name="s",
            skill_content="skill",
            resources={"REF.md": "ref"},
            scripts={"run.py": "code"},
        )
        assert files == {
            "plugins/p/skills/s/SKILL.md": "skill",
            "plugins/p/skills/s/references/REF.md": "ref",
            "plugins/p/skills/s/scripts/run.py": "code",
        }

    def test_overrides_replace_default_layout(self) -> None:
        files = GitHubMarketplacePublisher._build_file_map(
            plugin_name="p",
            skill_name="s",
            skill_content="skill",
            resources={"REF.md": "ref"},
            scripts={"run.py": "code"},
            skill_path="plugins/p/skills/folder/s/SKILL.md",
            resource_paths={"REF.md": "plugins/p/skills/folder/s/references/REF.md"},
            script_paths={"run.py": "plugins/p/skills/folder/s/scripts/run.py"},
        )
        assert files == {
            "plugins/p/skills/folder/s/SKILL.md": "skill",
            "plugins/p/skills/folder/s/references/REF.md": "ref",
            "plugins/p/skills/folder/s/scripts/run.py": "code",
        }

    def test_partial_overrides_mix_with_defaults(self) -> None:
        # Only the skill is overridden; resources/scripts use default layout.
        files = GitHubMarketplacePublisher._build_file_map(
            plugin_name="p",
            skill_name="s",
            skill_content="skill",
            resources={"REF.md": "ref"},
            scripts={"run.py": "code"},
            skill_path="custom/SKILL.md",
        )
        assert files == {
            "custom/SKILL.md": "skill",
            "plugins/p/skills/s/references/REF.md": "ref",
            "plugins/p/skills/s/scripts/run.py": "code",
        }

    def test_override_skips_name_segment_validation(self) -> None:
        # Resource names with slashes are rejected by the default branch,
        # but accepted when an explicit override path is provided.
        files = GitHubMarketplacePublisher._build_file_map(
            plugin_name="p",
            skill_name="s",
            skill_content="skill",
            resources={"sub/REF.md": "ref"},
            scripts={},
            resource_paths={"sub/REF.md": "plugins/p/skills/s/references/sub/REF.md"},
        )
        assert "plugins/p/skills/s/references/sub/REF.md" in files

    def test_invalid_override_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            GitHubMarketplacePublisher._build_file_map(
                plugin_name="p",
                skill_name="s",
                skill_content="x",
                resources={},
                scripts={},
                skill_path="/etc/passwd",
            )

    def test_rejects_resource_name_with_slash_when_no_override(self) -> None:
        with pytest.raises(ValueError, match=r"'\.\.'"):
            GitHubMarketplacePublisher._build_file_map(
                plugin_name="p",
                skill_name="s",
                skill_content="x",
                resources={"../escape.md": "y"},
                scripts={},
            )
