from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from langchain_ai_skills_framework.executors.my_shell_executor import MyShellExecutor
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_validation_error import (
    SkillValidationError,
)
from langchain_ai_skills_framework.loaders.marketplace_directory_loader import (
    MarketplaceDirectoryLoader,
)


@dataclass(frozen=True)
class FakeEnvVars:
    """Minimal implementation of SkillLoaderEnvironmentVariables for tests."""

    plugins_marketplace: str | None = "/var/data/marketplace"  # nosec B108
    plugins_marketplace_cache_folder: str | None = None
    skills_github_token: str | None = None
    skills_cache_timeout_seconds: int = 3600
    excluded_skills: set[str] = field(default_factory=set)
    excluded_skill_groups: set[str] = field(default_factory=set)
    plugins_marketplace_include: set[str] | None = None
    plugins_marketplace_exclude: set[str] = field(default_factory=set)
    plugins_marketplace_publish_enabled: bool = False
    plugins_marketplace_publish_branch: str = "main"
    plugins_marketplace_publish_use_branch: bool = True
    plugins_marketplace_publish_path: str | None = None
    snapshot_cache_plugins_collection: str | None = None
    plugins_collection: str | None = "plugins"
    plugin_skills_collection: str | None = "plugin_skills"
    plugin_references_collection: str | None = "plugin_references"
    plugin_scripts_collection: str | None = "plugin_scripts"


def _write_marketplace_skill(
    root: Path,
    plugin_name: str,
    skill_name: str,
    *,
    description: str | None = None,
    body: str = "Skill body.",
) -> Path:
    """Create a marketplace skill file at plugins/<plugin>/skills/<skill>/SKILL.md."""
    skill_dir = root / "plugins" / plugin_name / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(
        {"name": skill_name, "description": description or f"Description for {skill_name}"},
        sort_keys=False,
    ).strip()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\n{frontmatter}\n---\n{body}\n", encoding="utf-8")
    return skill_file


def _write_marketplace_json(
    root: Path,
    plugins: list[dict[str, str]],
    *,
    name: str = "test-marketplace",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Create .claude-plugin/marketplace.json at the marketplace root."""
    manifest_dir = root / ".claude-plugin"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "name": name,
        "owner": {"name": "Test Owner"},
        "plugins": plugins,
    }
    if metadata:
        manifest["metadata"] = metadata
    manifest_path = manifest_dir / "marketplace.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


class TestMarketplaceDirectoryLoaderInit:
    def test_missing_marketplace_uri_returns_empty(self) -> None:
        env = FakeEnvVars(plugins_marketplace=None)
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )
        assert loader.list_skill_summaries(allowed_skills=set()) == ()

    def test_blank_marketplace_uri_returns_empty(self) -> None:
        env = FakeEnvVars(plugins_marketplace="   ")
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )
        assert loader.list_skill_summaries(allowed_skills=set()) == ()


class TestLocalMarketplaceDiscovery:
    def test_discovers_skills_from_plugins_structure(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "plugin-a", "skill-one")
        _write_marketplace_skill(tmp_path, "plugin-b", "skill-two")

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "skill-one" in names
        assert "skill-two" in names

    def test_excludes_skills_by_name(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "plugin-a", "keep-me")
        _write_marketplace_skill(tmp_path, "plugin-a", "exclude-me")

        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            excluded_skills={"exclude-me"},
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "keep-me" in names
        assert "exclude-me" not in names

    def test_excludes_plugin_groups(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "good-plugin", "good-skill")
        _write_marketplace_skill(tmp_path, "bad-plugin", "bad-skill")

        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            excluded_skill_groups={"bad-plugin"},
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "good-skill" in names
        assert "bad-skill" not in names

    def test_get_skill_details_returns_content(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "plugin-a", "my-skill", body="Custom body.")

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        details = loader.get_skill_details("my-skill")
        assert "Custom body." in details.content

    def test_get_skill_details_raises_not_found(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "plugin-a", "exists")

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        with pytest.raises(SkillNotFoundError):
            loader.get_skill_details("does-not-exist")

    def test_nonexistent_local_path_raises(self) -> None:
        env = FakeEnvVars(plugins_marketplace="/nonexistent/path/xyz")
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        with pytest.raises(SkillValidationError, match="does not exist"):
            loader.list_skill_summaries(allowed_skills=set())

    def test_include_list_filters_plugins(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "glass-health", "glass-skill")
        _write_marketplace_skill(tmp_path, "other-plugin", "other-skill")

        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            plugins_marketplace_include={"glass-health"},
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "glass-skill" in names
        assert "other-skill" not in names

    def test_exclude_list_filters_plugins(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "good-plugin", "good-skill")
        _write_marketplace_skill(tmp_path, "unwanted", "unwanted-skill")

        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            plugins_marketplace_exclude={"unwanted"},
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "good-skill" in names
        assert "unwanted-skill" not in names

    def test_include_and_exclude_combined(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "alpha", "alpha-skill")
        _write_marketplace_skill(tmp_path, "beta", "beta-skill")
        _write_marketplace_skill(tmp_path, "gamma", "gamma-skill")

        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            plugins_marketplace_include={"alpha", "beta"},
            plugins_marketplace_exclude={"beta"},
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "alpha-skill" in names
        assert "beta-skill" not in names
        assert "gamma-skill" not in names


class TestCacheRefreshSemantics:
    def test_refresh_forces_reload(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "plugin-a", "original")

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        # Initial load
        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1

        # Add new skill on disk
        _write_marketplace_skill(tmp_path, "plugin-a", "new-skill")

        # Without refresh, TTL prevents reload
        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1

        # After refresh, new skill is visible
        loader.refresh()
        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]
        assert "new-skill" in names

    def test_snapshot_expires_after_ttl(self, tmp_path: Path) -> None:
        _write_marketplace_skill(tmp_path, "plugin-a", "skill-a")

        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            skills_cache_timeout_seconds=1,  # 1 second TTL
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        # Initial load
        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1

        # Add new skill
        _write_marketplace_skill(tmp_path, "plugin-a", "skill-b")

        # Wait for TTL to expire
        time.sleep(1.1)

        # After TTL, should pick up new skill
        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]
        assert "skill-b" in names


class TestGithubCacheTTL:
    def test_passes_ttl_to_downloader(self, tmp_path: Path) -> None:
        """Regression: previously used a bare existence check that never re-downloaded."""
        env = FakeEnvVars(
            plugins_marketplace="github://org/repo/plugins?ref=main",
            skills_cache_timeout_seconds=600,
        )
        downloader = MagicMock()
        captured_calls: list[dict[str, int]] = []

        def fake_download(
            *, source_uri: str, github_token: str | None, cache_path: Path, cache_ttl_seconds: int = 0
        ) -> Path:
            captured_calls.append({"cache_ttl_seconds": cache_ttl_seconds})
            plugins_dir = tmp_path / "plugins" / "test-plugin" / "skills" / "test-skill"
            plugins_dir.mkdir(parents=True, exist_ok=True)
            frontmatter = "---\nname: test-skill\ndescription: A test skill\n---\nBody.\n"
            (plugins_dir / "SKILL.md").write_text(frontmatter, encoding="utf-8")
            return tmp_path

        downloader.download = fake_download

        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=downloader,
        )
        loader.list_skill_summaries(allowed_skills=set())

        assert len(captured_calls) == 1
        # Should pass the TTL value (600), not 0
        assert captured_calls[0]["cache_ttl_seconds"] == 600

    def test_force_passes_zero_ttl(self, tmp_path: Path) -> None:
        """refresh() should force a re-download by passing cache_ttl_seconds=0."""
        env = FakeEnvVars(
            plugins_marketplace="github://org/repo/plugins?ref=main",
            skills_cache_timeout_seconds=3600,
        )
        downloader = MagicMock()
        captured_calls: list[dict[str, int]] = []

        def fake_download(
            *, source_uri: str, github_token: str | None, cache_path: Path, cache_ttl_seconds: int = 0
        ) -> Path:
            captured_calls.append({"cache_ttl_seconds": cache_ttl_seconds})
            plugins_dir = tmp_path / "plugins" / "test-plugin" / "skills" / "test-skill"
            plugins_dir.mkdir(parents=True, exist_ok=True)
            frontmatter = "---\nname: test-skill\ndescription: A test skill\n---\nBody.\n"
            (plugins_dir / "SKILL.md").write_text(frontmatter, encoding="utf-8")
            return tmp_path

        downloader.download = fake_download

        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=downloader,
        )
        # Initial load uses TTL
        loader.list_skill_summaries(allowed_skills=set())
        # refresh() should force download (cache_ttl_seconds=0)
        loader.refresh()

        assert len(captured_calls) == 2
        assert captured_calls[0]["cache_ttl_seconds"] == 3600
        assert captured_calls[1]["cache_ttl_seconds"] == 0


class TestShellScriptSecurity:
    @pytest.fixture(autouse=True)
    def _require_sh(self) -> None:
        import shutil

        if shutil.which("sh") is None:
            pytest.skip("sh not available in this environment")

    @pytest.mark.asyncio
    async def test_shell_script_uses_restricted_environment(self, tmp_path: Path) -> None:
        """Regression: previously passed full os.environ to shell scripts."""
        script = tmp_path / "dump_env.sh"
        script.write_text("#!/bin/sh\nenv\n", encoding="utf-8")
        script.chmod(0o755)

        executor = MyShellExecutor()
        result = await executor.execute(
            script_path=script,
            skill_base_dir=tmp_path,
            arguments={"my_key": "my_value"},
        )

        assert result.success
        assert result.stdout is not None
        env_output = result.stdout

        # Should contain only the restricted set
        assert "SKILL_NAME=" in env_output
        assert "SKILL_BASE_DIR=" in env_output

        # User arguments should NOT appear as env vars (they're on stdin now)
        assert "MY_KEY=my_value" not in env_output

    @pytest.mark.asyncio
    async def test_shell_script_receives_arguments_on_stdin(self, tmp_path: Path) -> None:
        """Arguments should be passed as JSON on stdin, not as env vars."""
        script = tmp_path / "read_stdin.sh"
        script.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
        script.chmod(0o755)

        executor = MyShellExecutor()
        result = await executor.execute(
            script_path=script,
            skill_base_dir=tmp_path,
            arguments={"key": "value", "number": "42"},
        )

        assert result.success
        assert result.stdout is not None
        import json

        parsed = json.loads(result.stdout)
        assert parsed == {"key": "value", "number": "42"}

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """Scripts outside the skill base directory should be rejected."""
        from langchain_ai_skills_framework.executors.my_script_executor import PathSecurityError

        outer_dir = tmp_path / "outer"
        outer_dir.mkdir()
        script = outer_dir / "evil.sh"
        script.write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
        script.chmod(0o755)

        inner_dir = tmp_path / "inner"
        inner_dir.mkdir()

        executor = MyShellExecutor()
        with pytest.raises(PathSecurityError):
            await executor.execute(
                script_path=script,
                skill_base_dir=inner_dir,
            )

    @pytest.mark.asyncio
    async def test_rejects_invalid_argument_keys(self, tmp_path: Path) -> None:
        """Argument keys must match safe identifier pattern."""
        script = tmp_path / "noop.sh"
        script.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
        script.chmod(0o755)

        executor = MyShellExecutor()
        with pytest.raises(ValueError, match="Invalid argument key"):
            await executor.execute(
                script_path=script,
                skill_base_dir=tmp_path,
                arguments={"../../etc/passwd": "bad"},  # pragma: allowlist secret
            )

    @pytest.mark.asyncio
    async def test_rejects_world_writable_script(self, tmp_path: Path) -> None:
        """World-writable scripts should be rejected."""
        from langchain_ai_skills_framework.executors.my_script_executor import ScriptPermissionError

        script = tmp_path / "writable.sh"
        script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
        script.chmod(0o777)  # world-writable

        executor = MyShellExecutor()
        with pytest.raises(ScriptPermissionError):
            await executor.execute(
                script_path=script,
                skill_base_dir=tmp_path,
            )


class TestMarketplaceJsonDiscovery:
    """Tests for .claude-plugin/marketplace.json based plugin discovery."""

    def test_discovers_plugins_from_marketplace_json(self, tmp_path: Path) -> None:
        """Plugins listed in marketplace.json are discovered by name and source path."""
        _write_marketplace_skill(tmp_path, "plugin-a", "skill-one")
        _write_marketplace_skill(tmp_path, "plugin-b", "skill-two")
        _write_marketplace_json(
            tmp_path,
            [
                {"name": "plugin-a", "source": "./plugins/plugin-a", "description": "First plugin"},
                {"name": "plugin-b", "source": "./plugins/plugin-b", "description": "Second plugin"},
            ],
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]

        assert "skill-one" in names
        assert "skill-two" in names

    def test_marketplace_json_name_used_for_filtering(self, tmp_path: Path) -> None:
        """The plugin 'name' from marketplace.json is used for include/exclude, not the directory name."""
        # Directory is called "my-plugin-dir" but marketplace.json names it "custom-name"
        skill_dir = tmp_path / "my-plugin-dir" / "skills" / "the-skill"
        skill_dir.mkdir(parents=True)
        frontmatter = yaml.safe_dump(
            {"name": "the-skill", "description": "A skill"},
            sort_keys=False,
        ).strip()
        (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nBody.\n")

        _write_marketplace_json(
            tmp_path,
            [{"name": "custom-name", "source": "./my-plugin-dir"}],
        )

        # Include filter uses the marketplace.json name, not the directory name
        env = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            plugins_marketplace_include={"custom-name"},
        )
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )
        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1
        assert summaries[0].name == "the-skill"

        # Excluding by directory name should NOT filter it (wrong name)
        env_wrong = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            plugins_marketplace_exclude={"my-plugin-dir"},
        )
        loader_wrong = MarketplaceDirectoryLoader(
            environment_variables=env_wrong,
            github_directory_downloader=MagicMock(),
        )
        summaries_wrong = loader_wrong.list_skill_summaries(allowed_skills=set())
        assert len(summaries_wrong) == 1  # NOT excluded because name doesn't match

        # Excluding by marketplace.json name DOES filter it
        env_correct = FakeEnvVars(
            plugins_marketplace=str(tmp_path),
            plugins_marketplace_exclude={"custom-name"},
        )
        loader_correct = MarketplaceDirectoryLoader(
            environment_variables=env_correct,
            github_directory_downloader=MagicMock(),
        )
        summaries_correct = loader_correct.list_skill_summaries(allowed_skills=set())
        assert len(summaries_correct) == 0

    def test_plugin_root_resolves_bare_relative_paths(self, tmp_path: Path) -> None:
        """metadata.pluginRoot is prepended to bare relative paths (no ./ prefix)."""
        # Skill lives at <root>/custom-root/plugin-a/skills/deep-skill/SKILL.md
        deep_root = tmp_path / "custom-root"
        skill_dir = deep_root / "plugin-a" / "skills" / "deep-skill"
        skill_dir.mkdir(parents=True)
        frontmatter = yaml.safe_dump(
            {"name": "deep-skill", "description": "Deeply nested"},
            sort_keys=False,
        ).strip()
        (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nBody.\n")

        _write_marketplace_json(
            tmp_path,
            [{"name": "plugin-a", "source": "plugin-a"}],
            metadata={"pluginRoot": "custom-root"},
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1
        assert summaries[0].name == "deep-skill"

    def test_dot_slash_source_ignores_plugin_root(self, tmp_path: Path) -> None:
        """Explicit ./ paths resolve from marketplace root, ignoring pluginRoot."""
        # Skill lives at <root>/plugins/plugin-a/skills/deep-skill/SKILL.md
        skill_dir = tmp_path / "plugins" / "plugin-a" / "skills" / "deep-skill"
        skill_dir.mkdir(parents=True)
        frontmatter = yaml.safe_dump(
            {"name": "deep-skill", "description": "Deeply nested"},
            sort_keys=False,
        ).strip()
        (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nBody.\n")

        _write_marketplace_json(
            tmp_path,
            [{"name": "plugin-a", "source": "./plugins/plugin-a"}],
            metadata={"pluginRoot": "should-not-be-used"},
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1
        assert summaries[0].name == "deep-skill"

    def test_bare_relative_paths_resolve(self, tmp_path: Path) -> None:
        """Source paths without ./ prefix are treated as relative to marketplace root."""
        _write_marketplace_skill(tmp_path, "plugin-a", "bare-skill")
        _write_marketplace_json(
            tmp_path,
            [{"name": "plugin-a", "source": "plugins/plugin-a"}],
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1
        assert summaries[0].name == "bare-skill"

    def test_missing_source_plugin_skipped(self, tmp_path: Path) -> None:
        """Plugins whose source path doesn't exist are skipped gracefully."""
        _write_marketplace_skill(tmp_path, "real-plugin", "real-skill")
        _write_marketplace_json(
            tmp_path,
            [
                {"name": "real-plugin", "source": "./plugins/real-plugin"},
                {"name": "ghost-plugin", "source": "./plugins/nonexistent"},
            ],
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]
        assert "real-skill" in names
        assert len(summaries) == 1

    def test_invalid_marketplace_json_falls_back_to_directory_scan(self, tmp_path: Path) -> None:
        """Malformed marketplace.json triggers fallback to directory scanning."""
        _write_marketplace_skill(tmp_path, "plugin-a", "fallback-skill")

        manifest_dir = tmp_path / ".claude-plugin"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "marketplace.json").write_text("not valid json{{{", encoding="utf-8")

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]
        assert "fallback-skill" in names

    def test_no_marketplace_json_falls_back_to_directory_scan(self, tmp_path: Path) -> None:
        """Without marketplace.json, legacy directory scanning is used."""
        _write_marketplace_skill(tmp_path, "legacy-plugin", "legacy-skill")

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        names = [s.name for s in summaries]
        assert "legacy-skill" in names

    def test_empty_plugins_array_returns_no_plugins(self, tmp_path: Path) -> None:
        """An empty plugins array in marketplace.json results in no plugins loaded."""
        _write_marketplace_json(tmp_path, [])

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 0

    def test_plugin_entry_missing_name_skipped(self, tmp_path: Path) -> None:
        """Plugin entries without a 'name' field are skipped."""
        _write_marketplace_skill(tmp_path, "good-plugin", "good-skill")
        _write_marketplace_json(
            tmp_path,
            [
                {"name": "good-plugin", "source": "./plugins/good-plugin"},
                {"source": "./plugins/nameless"},  # missing name
            ],
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1
        assert summaries[0].name == "good-skill"

    def test_plugin_entry_missing_source_skipped(self, tmp_path: Path) -> None:
        """Plugin entries without a 'source' field are skipped."""
        _write_marketplace_skill(tmp_path, "valid-plugin", "valid-skill")
        _write_marketplace_json(
            tmp_path,
            [
                {"name": "valid-plugin", "source": "./plugins/valid-plugin"},
                {"name": "sourceless"},  # missing source
            ],
        )

        env = FakeEnvVars(plugins_marketplace=str(tmp_path))
        loader = MarketplaceDirectoryLoader(
            environment_variables=env,
            github_directory_downloader=MagicMock(),
        )

        summaries = loader.list_skill_summaries(allowed_skills=set())
        assert len(summaries) == 1
        assert summaries[0].name == "valid-skill"
