"""Encapsulates marketplace plugin discovery and MCP config reading.

This module handles:
- Discovering plugins via ``.claude-plugin/marketplace.json`` (with directory-scanning fallback)
- Reading per-plugin ``.mcp.json`` files
- Resolving plugin source paths (relative, absolute, bare)
- ``${CLAUDE_PLUGIN_ROOT}`` variable substitution in MCP configs
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.utilities.skill_name_normalizer import normalize_skill_name

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)}")


@dataclass(frozen=True)
class PluginEntry:
    """A discovered plugin with its resolved filesystem path and canonical name."""

    name: str
    path: Path
    description: str | None = None


class MarketplacePluginManager:
    """Discovers plugins and reads their MCP server configurations.

    Supports two discovery modes:
    1. Manifest-based: reads ``.claude-plugin/marketplace.json``
    2. Directory-based: scans for subdirectories with ``skills/`` folders

    Per-plugin ``.mcp.json`` files are parsed to extract MCP server
    entries with ``${CLAUDE_PLUGIN_ROOT}`` substitution.
    """

    def discover_plugins(
        self,
        marketplace_root: Path,
        *,
        include_filter: frozenset[str] | None = None,
        exclude_filter: frozenset[str] | None = None,
    ) -> list[PluginEntry]:
        """Discover plugins from a marketplace root directory.

        Args:
            marketplace_root: Path to the downloaded marketplace.
            include_filter: If set, only plugins whose normalized names
                appear in this set are returned.
            exclude_filter: Plugin names to exclude.

        Returns:
            Sorted list of PluginEntry objects for valid, non-excluded plugins.
        """
        entries = self._discover_raw(marketplace_root)

        if include_filter or exclude_filter:
            entries = self._apply_filters(
                entries,
                include_filter=include_filter,
                exclude_filter=exclude_filter,
            )

        return entries

    def read_mcp_configs(self, entry: PluginEntry) -> list[PluginMcpServerEntry]:
        """Read .mcp.json from a plugin directory and return server entries.

        Substitutes ``${CLAUDE_PLUGIN_ROOT}`` in url, command, args, and env
        values with the plugin's resolved path.
        """
        mcp_json_path = entry.path / ".mcp.json"
        if not mcp_json_path.is_file():
            return []

        try:
            raw = mcp_json_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read .mcp.json for plugin '%s' at %s: %s",
                entry.name,
                mcp_json_path,
                exc,
            )
            return []

        if not isinstance(data, dict):
            return []

        servers_dict = data.get("mcpServers", {})
        if not isinstance(servers_dict, dict):
            return []

        plugin_root_str = str(entry.path)
        entries: list[PluginMcpServerEntry] = []

        for server_key, server_config in servers_dict.items():
            if not isinstance(server_key, str) or not isinstance(server_config, dict):
                continue

            def _sub(value: str, _root: str = plugin_root_str) -> str:
                """Substitute ${CLAUDE_PLUGIN_ROOT} and ${ENV_VAR} references."""
                value = value.replace("${CLAUDE_PLUGIN_ROOT}", _root)
                return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)

            url = server_config.get("url")
            command = server_config.get("command")
            args_raw = server_config.get("args", [])
            env_raw = server_config.get("env", {})
            headers_raw = server_config.get("headers", {})
            oauth_raw = server_config.get("oauth")

            mcp_entry = PluginMcpServerEntry(
                server_key=server_key,
                plugin_name=entry.name,
                plugin_root=entry.path,
                url=_sub(url) if isinstance(url, str) else None,
                command=_sub(command) if isinstance(command, str) else None,
                args=tuple(_sub(a) for a in args_raw if isinstance(a, str)),
                env={k: _sub(v) for k, v in env_raw.items() if isinstance(k, str) and isinstance(v, str)},
                headers={k: _sub(v) for k, v in headers_raw.items() if isinstance(k, str) and isinstance(v, str)},
                description=server_config.get("description")
                if isinstance(server_config.get("description"), str)
                else None,
                display_name=server_config.get("displayName")
                if isinstance(server_config.get("displayName"), str)
                else None,
                auth=server_config.get("auth") if isinstance(server_config.get("auth"), str) else None,
                oauth=oauth_raw if isinstance(oauth_raw, dict) else None,
            )

            if not mcp_entry.is_http:
                logger.debug(
                    "Plugin '%s' MCP server '%s' has no url (stdio-only); skipping for server-side use",
                    entry.name,
                    server_key,
                )

            entries.append(mcp_entry)

        if entries:
            logger.info(
                "Plugin '%s': discovered %d MCP server(s) from .mcp.json",
                entry.name,
                len(entries),
            )

        return entries

    def collect_all_mcp_configs(self, entries: Sequence[PluginEntry]) -> list[PluginMcpServerEntry]:
        """Read MCP configs from all plugin entries and return a flat list."""
        result: list[PluginMcpServerEntry] = []
        for entry in entries:
            result.extend(self.read_mcp_configs(entry))
        return result

    # --- Private implementation ------------------------------------------------

    def _discover_raw(self, marketplace_root: Path) -> list[PluginEntry]:
        """Discover plugins via marketplace.json or directory scanning fallback."""
        manifest_path = marketplace_root / ".claude-plugin" / "marketplace.json"
        if manifest_path.is_file():
            return self._parse_marketplace_json(manifest_path, marketplace_root)

        return self._discover_from_directories(marketplace_root)

    def _parse_marketplace_json(self, manifest_path: Path, marketplace_root: Path) -> list[PluginEntry]:
        """Parse .claude-plugin/marketplace.json and resolve plugin paths."""
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read marketplace.json at %s: %s; falling back to directory scan",
                manifest_path,
                exc,
            )
            return self._discover_from_directories(marketplace_root)

        if not isinstance(manifest, dict):
            logger.warning(
                "marketplace.json at %s is not a JSON object; falling back to directory scan",
                manifest_path,
            )
            return self._discover_from_directories(marketplace_root)

        plugins_list = manifest.get("plugins", [])
        if not isinstance(plugins_list, list):
            logger.warning("marketplace.json 'plugins' is not an array; falling back to directory scan")
            return self._discover_from_directories(marketplace_root)

        # metadata.pluginRoot is a base directory prepended to relative source paths
        metadata = manifest.get("metadata") or {}
        plugin_root_str = metadata.get("pluginRoot", "")

        entries: list[PluginEntry] = []
        for plugin_spec in plugins_list:
            if not isinstance(plugin_spec, dict):
                continue

            name = plugin_spec.get("name")
            source = plugin_spec.get("source")
            if not name or not isinstance(name, str):
                logger.warning("Skipping marketplace plugin entry with missing/invalid 'name'")
                continue
            if not source or not isinstance(source, str):
                logger.warning(
                    "Skipping marketplace plugin '%s' with missing/invalid 'source'",
                    name,
                )
                continue

            resolved_path = self._resolve_plugin_source(
                source=source,
                marketplace_root=marketplace_root,
                plugin_root=plugin_root_str,
            )
            if resolved_path is None:
                logger.warning(
                    "Skipping marketplace plugin '%s': unsupported source type '%s'",
                    name,
                    source,
                )
                continue

            if not resolved_path.is_dir():
                logger.warning(
                    "Skipping marketplace plugin '%s': resolved path does not exist: %s",
                    name,
                    resolved_path,
                )
                continue

            description = plugin_spec.get("description")
            entries.append(
                PluginEntry(
                    name=name.strip(),
                    path=resolved_path,
                    description=description if isinstance(description, str) else None,
                )
            )

        logger.info(
            "marketplace.json: discovered %d plugins from %s",
            len(entries),
            manifest_path,
        )
        return sorted(entries, key=lambda e: e.name)

    @staticmethod
    def _resolve_plugin_source(*, source: str, marketplace_root: Path, plugin_root: str) -> Path | None:
        """Resolve a plugin source path from marketplace.json.

        Supports relative paths (starting with ``./`` or ``../``), absolute
        paths, and bare relative paths.  Other source types (git URLs, npm
        packages) are not yet supported and return None.

        Explicit relative paths (``./``, ``../``) resolve from ``marketplace_root``
        directly — they already carry their own directory context.
        ``pluginRoot`` is only prepended for bare relative paths (no leading dot).
        """
        if source.startswith("./") or source.startswith("../"):
            return (marketplace_root / source).resolve()

        # Absolute local path
        if source.startswith("/"):
            return Path(source)

        # Bare relative path (no ./ prefix)
        base = marketplace_root
        if plugin_root:
            base = marketplace_root / plugin_root
        return (base / source).resolve()

    @staticmethod
    def _discover_from_directories(cache_path: Path) -> list[PluginEntry]:
        """Legacy fallback: discover plugins by scanning directories.

        Supports two layouts:
        1. Nested: cache_path/plugins/ contains plugin subdirectories
        2. Direct: cache_path contains plugin subdirectories with skills/ folders
        """
        plugins_subdir = cache_path / "plugins"
        if plugins_subdir.is_dir():
            return sorted(
                (
                    PluginEntry(name=d.name, path=d)
                    for d in plugins_subdir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ),
                key=lambda e: e.name,
            )

        return sorted(
            (
                PluginEntry(name=d.name, path=d)
                for d in cache_path.iterdir()
                if d.is_dir() and not d.name.startswith(".") and (d / "skills").is_dir()
            ),
            key=lambda e: e.name,
        )

    @staticmethod
    def _apply_filters(
        entries: list[PluginEntry],
        *,
        include_filter: frozenset[str] | None,
        exclude_filter: frozenset[str] | None,
    ) -> list[PluginEntry]:
        """Apply include/exclude filters to a list of plugin entries."""
        result: list[PluginEntry] = []
        for entry in entries:
            normalized = normalize_skill_name(entry.name)
            if include_filter and normalized not in include_filter:
                logger.debug(
                    "Marketplace: skipping plugin '%s' (not in include list)",
                    entry.name,
                )
                continue
            if exclude_filter and normalized in exclude_filter:
                logger.info("Marketplace: skipping excluded plugin '%s'", entry.name)
                continue
            result.append(entry)
        return result
