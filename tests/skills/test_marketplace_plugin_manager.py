"""Unit tests for MarketplacePluginManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langchain_ai_skills_framework.loaders.marketplace_plugin_manager import (
    MarketplacePluginManager,
    PluginEntry,
)


@pytest.fixture
def manager() -> MarketplacePluginManager:
    return MarketplacePluginManager()


class TestDiscoverPlugins:
    """Tests for plugin discovery logic."""

    def test_discovers_from_marketplace_json(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        # Create plugin directory
        plugin_dir = tmp_path / "plugins" / "my-plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "skills").mkdir()

        # Write marketplace.json
        manifest_dir = tmp_path / ".claude-plugin"
        manifest_dir.mkdir()
        manifest = {
            "name": "test-marketplace",
            "plugins": [{"name": "my-plugin", "source": "./plugins/my-plugin", "description": "A test plugin"}],
        }
        (manifest_dir / "marketplace.json").write_text(json.dumps(manifest))

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "my-plugin"
        assert entries[0].path == plugin_dir.resolve()
        assert entries[0].description == "A test plugin"

    def test_falls_back_to_directory_scan(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        # No marketplace.json — uses directory fallback
        plugin_dir = tmp_path / "plugins" / "fallback-plugin"
        plugin_dir.mkdir(parents=True)

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "fallback-plugin"

    def test_directory_scan_requires_skills_dir_when_no_plugins_subdir(
        self, tmp_path: Path, manager: MarketplacePluginManager
    ) -> None:
        # Direct layout: only dirs with skills/ are included
        plugin_with_skills = tmp_path / "good-plugin"
        plugin_with_skills.mkdir()
        (plugin_with_skills / "skills").mkdir()

        plugin_without = tmp_path / "no-skills"
        plugin_without.mkdir()

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "good-plugin"

    def test_include_filter(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        for name in ("alpha", "beta", "gamma"):
            d = tmp_path / "plugins" / name
            d.mkdir(parents=True)

        entries = manager.discover_plugins(tmp_path, include_filter=frozenset({"alpha", "gamma"}))
        names = {e.name for e in entries}
        assert names == {"alpha", "gamma"}

    def test_exclude_filter(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        for name in ("alpha", "beta", "gamma"):
            d = tmp_path / "plugins" / name
            d.mkdir(parents=True)

        entries = manager.discover_plugins(tmp_path, exclude_filter=frozenset({"beta"}))
        names = {e.name for e in entries}
        assert "beta" not in names
        assert "alpha" in names

    def test_plugin_root_in_metadata_bare_path(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        # pluginRoot applies to bare relative paths (no ./ prefix)
        plugin_dir = tmp_path / "packages" / "my-plugin"
        plugin_dir.mkdir(parents=True)

        manifest_dir = tmp_path / ".claude-plugin"
        manifest_dir.mkdir()
        manifest = {
            "plugins": [{"name": "my-plugin", "source": "my-plugin"}],
            "metadata": {"pluginRoot": "packages"},
        }
        (manifest_dir / "marketplace.json").write_text(json.dumps(manifest))

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].path == plugin_dir.resolve()

    def test_dot_slash_source_ignores_plugin_root(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        # Explicit ./ paths resolve from marketplace root, not pluginRoot
        plugin_dir = tmp_path / "packages" / "my-plugin"
        plugin_dir.mkdir(parents=True)

        manifest_dir = tmp_path / ".claude-plugin"
        manifest_dir.mkdir()
        manifest = {
            "plugins": [{"name": "my-plugin", "source": "./packages/my-plugin"}],
            "metadata": {"pluginRoot": "should-not-be-used"},
        }
        (manifest_dir / "marketplace.json").write_text(json.dumps(manifest))

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].path == plugin_dir.resolve()

    def test_skips_hidden_directories(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        (tmp_path / "plugins" / ".hidden").mkdir(parents=True)
        (tmp_path / "plugins" / "visible").mkdir(parents=True)

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "visible"

    def test_invalid_marketplace_json_falls_back(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        manifest_dir = tmp_path / ".claude-plugin"
        manifest_dir.mkdir()
        (manifest_dir / "marketplace.json").write_text("not valid json {{{")

        # Fallback directory scan
        plugin_dir = tmp_path / "plugins" / "fallback"
        plugin_dir.mkdir(parents=True)

        entries = manager.discover_plugins(tmp_path)
        assert len(entries) == 1
        assert entries[0].name == "fallback"


class TestReadMcpConfigs:
    """Tests for .mcp.json reading."""

    def test_reads_http_server(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        mcp_config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8080/mcp",
                    "description": "Test server",
                    "displayName": "My Server",
                }
            }
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(mcp_config))

        entry = PluginEntry(name="test-plugin", path=tmp_path)
        configs = manager.read_mcp_configs(entry)

        assert len(configs) == 1
        assert configs[0].server_key == "my-server"
        assert configs[0].plugin_name == "test-plugin"
        assert configs[0].url == "http://localhost:8080/mcp"
        assert configs[0].description == "Test server"
        assert configs[0].display_name == "My Server"
        assert configs[0].is_http is True

    def test_reads_stdio_server(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        mcp_config = {
            "mcpServers": {
                "local-tool": {
                    "command": "python",
                    "args": ["-m", "my_tool"],
                }
            }
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(mcp_config))

        entry = PluginEntry(name="test-plugin", path=tmp_path)
        configs = manager.read_mcp_configs(entry)

        assert len(configs) == 1
        assert configs[0].is_http is False
        assert configs[0].command == "python"
        assert configs[0].args == ("-m", "my_tool")

    def test_substitutes_plugin_root(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        mcp_config = {
            "mcpServers": {
                "server": {
                    "url": "http://localhost:8080",
                    "command": "${CLAUDE_PLUGIN_ROOT}/bin/run",
                    "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
                    "env": {"HOME": "${CLAUDE_PLUGIN_ROOT}"},
                }
            }
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(mcp_config))

        entry = PluginEntry(name="test-plugin", path=tmp_path)
        configs = manager.read_mcp_configs(entry)

        root = str(tmp_path)
        assert configs[0].command == f"{root}/bin/run"
        assert configs[0].args == ("--config", f"{root}/config.json")
        assert configs[0].env == {"HOME": root}

    def test_returns_empty_for_missing_file(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        entry = PluginEntry(name="no-mcp", path=tmp_path)
        assert manager.read_mcp_configs(entry) == []

    def test_returns_empty_for_invalid_json(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        (tmp_path / ".mcp.json").write_text("not json")
        entry = PluginEntry(name="bad-json", path=tmp_path)
        assert manager.read_mcp_configs(entry) == []

    def test_namespaced_key(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        mcp_config = {"mcpServers": {"search": {"url": "http://localhost:9090"}}}
        (tmp_path / ".mcp.json").write_text(json.dumps(mcp_config))

        entry = PluginEntry(name="my-plugin", path=tmp_path)
        configs = manager.read_mcp_configs(entry)

        assert configs[0].namespaced_key == "my-plugin__search"

    def test_headers_and_auth(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        mcp_config = {
            "mcpServers": {
                "authed": {
                    "url": "http://localhost:8080",
                    "headers": {"X-Custom": "value"},
                    "auth": "oauth2",
                }
            }
        }
        (tmp_path / ".mcp.json").write_text(json.dumps(mcp_config))

        entry = PluginEntry(name="test", path=tmp_path)
        configs = manager.read_mcp_configs(entry)

        assert configs[0].headers == {"X-Custom": "value"}
        assert configs[0].auth == "oauth2"


class TestCollectAllMcpConfigs:
    """Tests for batch collection across multiple plugins."""

    def test_collects_from_multiple_plugins(self, tmp_path: Path, manager: MarketplacePluginManager) -> None:
        for i, name in enumerate(("plugin-a", "plugin-b")):
            plugin_dir = tmp_path / name
            plugin_dir.mkdir()
            mcp_config = {"mcpServers": {f"server-{i}": {"url": f"http://localhost:{8080 + i}"}}}
            (plugin_dir / ".mcp.json").write_text(json.dumps(mcp_config))

        entries = [
            PluginEntry(name="plugin-a", path=tmp_path / "plugin-a"),
            PluginEntry(name="plugin-b", path=tmp_path / "plugin-b"),
        ]
        configs = manager.collect_all_mcp_configs(entries)

        assert len(configs) == 2
        urls = {c.url for c in configs}
        assert "http://localhost:8080" in urls
        assert "http://localhost:8081" in urls
