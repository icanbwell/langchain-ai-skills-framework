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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_ai_skills_framework.models.plugin_mcp_config import (
    PluginMcpServerEntry,
    SkippedMcpServer,
    coerce_mcp_visibility,
)
from langchain_ai_skills_framework.utilities.skill_name_normalizer import normalize_skill_name

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)}")


@dataclass(frozen=True)
class McpConfigReadResult:
    """Result of reading one plugin's ``.mcp.json``: the servers that
    resolved cleanly, plus any that were skipped for referencing an unset
    ``${ENV_VAR}`` (see :class:`SkippedMcpServer`)."""

    entries: tuple[PluginMcpServerEntry, ...] = ()
    skipped: tuple[SkippedMcpServer, ...] = ()


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

    ``${ENV_VAR}`` in a server's ``url``, ``headers``, ``command``, ``args`` and ``env`` is expanded from the process
    environment. ``env_expansion_plugins`` limits that to the named plugins (matched like
    ``PLUGINS_MARKETPLACE_INCLUDE``: case-insensitive, ``_`` and spaces equal ``-``): a plugin outside the set keeps the placeholder verbatim, so a plugin
    author cannot read a service secret (a database password, a GitHub App key) by naming it in
    ``.mcp.json``. ``None`` (the default) expands for every plugin, as before.

    ``env_expansion_variables`` narrows that per variable. It is read on every lookup, so a mapping the host fills
    in later is honoured. It maps a plugin name (normalized the same way) to the exact
    variable names that plugin may expand. A plugin in ``env_expansion_plugins`` is fully trusted and may expand any
    variable. Any other plugin with an entry in the map expands only the listed variables and keeps every other
    ``${VAR}`` verbatim (a warning logs the plugin and the variable name, never a value). A plugin in neither expands
    nothing when ``env_expansion_plugins`` is set; when it is ``None`` such a plugin expands everything, as before.
    """

    def __init__(
        self,
        *,
        env_expansion_plugins: frozenset[str] | None = None,
        env_expansion_variables: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._env_expansion_plugins = (
            None
            if env_expansion_plugins is None
            else frozenset(normalize_skill_name(value=name) for name in env_expansion_plugins)
        )
        # Held by reference, not copied: the host may build the loader before it knows the variables and fill (or
        # replace the contents of) this mapping later.
        self._env_expansion_variables = env_expansion_variables

    def _variables_for(self, normalized_plugin: str) -> frozenset[str] | None:
        """Union of the variable sets whose key normalizes to ``normalized_plugin``, read from the live mapping."""
        if self._env_expansion_variables is None:
            return None
        matched: frozenset[str] | None = None
        for name, variables in self._env_expansion_variables.items():
            if normalize_skill_name(value=name) == normalized_plugin:
                matched = frozenset(variables) if matched is None else matched | frozenset(variables)
        return matched

    def _allowed_env_vars(self, plugin_name: str) -> frozenset[str] | None:
        """The variable names a plugin may expand: ``None`` means any, an empty set means none."""
        normalized = normalize_skill_name(value=plugin_name)
        if self._env_expansion_plugins is not None and normalized in self._env_expansion_plugins:
            return None
        per_variable = self._variables_for(normalized)
        if per_variable is not None:
            return per_variable
        if self._env_expansion_plugins is None:
            return None
        return frozenset()

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

    def read_mcp_configs(self, entry: PluginEntry) -> McpConfigReadResult:
        """Read .mcp.json from a plugin directory and return server entries.

        Substitutes ``${CLAUDE_PLUGIN_ROOT}`` in url, command, args, and env
        values with the plugin's resolved path.
        """
        mcp_json_path = entry.path / ".mcp.json"
        if not mcp_json_path.is_file():
            return McpConfigReadResult()

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
            return McpConfigReadResult()

        if not isinstance(data, dict):
            return McpConfigReadResult()

        servers_dict = data.get("mcpServers", {})
        if not isinstance(servers_dict, dict):
            return McpConfigReadResult()

        plugin_root_str = str(entry.path)
        allowed_env = self._allowed_env_vars(entry.name)
        entries: list[PluginMcpServerEntry] = []
        skipped: list[SkippedMcpServer] = []

        for server_key, server_config in servers_dict.items():
            if not isinstance(server_key, str) or not isinstance(server_config, dict):
                continue

            missing_env_vars: list[str] = []
            unexpanded_env_vars: list[str] = []
            unexpanded_other_vars: list[str] = []

            def _sub_strict(
                value: str,
                _root: str = plugin_root_str,
                _missing: list[str] = missing_env_vars,
                _unexpanded: list[str] = unexpanded_env_vars,
            ) -> str:
                """Substitute ${CLAUDE_PLUGIN_ROOT} and ${ENV_VAR} in url/headers.

                Records any ${ENV_VAR} whose variable is unset in _missing
                instead of silently leaving the literal "${ENV_VAR}" text in
                the result -- an unresolved placeholder is not a valid url/
                header value, and passing it through as if it were one just
                relocates the failure somewhere far less diagnosable
                downstream (e.g. an SSRF check rejecting a URL that's
                literally "${MCP_SERVER_GATEWAY_URL}/skills-library/").
                Only url/headers use this -- see is_http's docstring: this
                loader never acts on command/args/env server-side, so an
                unresolved placeholder there is inert, not a latent bug, and
                forcing it through this same check would drop an otherwise-
                valid HTTP entry over an irrelevant stdio-only field.
                """
                value = value.replace("${CLAUDE_PLUGIN_ROOT}", _root)

                def _replace(m: re.Match[str]) -> str:
                    name = m.group(1)
                    if allowed_env is not None and name not in allowed_env:
                        # Not allowed for this plugin: the placeholder stays as written, and an unset
                        # variable is not an error because nothing is read.
                        _unexpanded.append(name)
                        return m.group(0)
                    env_value = os.environ.get(name)
                    if env_value is None:
                        _missing.append(name)
                        return m.group(0)
                    return env_value

                return _ENV_VAR_RE.sub(_replace, value)

            def _sub_lenient(
                value: str,
                _root: str = plugin_root_str,
                _unexpanded: list[str] = unexpanded_other_vars,
            ) -> str:
                """Substitute ${CLAUDE_PLUGIN_ROOT} and ${ENV_VAR} in command/args/env.

                Best-effort only, no missing-var tracking -- see _sub_strict's
                docstring for why these stdio-only fields are exempt. A variable the
                plugin may not expand stays verbatim here too: these values are
                persisted by the sync, so expanding them would leak the same
                secrets the allowlist guards.
                """
                value = value.replace("${CLAUDE_PLUGIN_ROOT}", _root)

                def _replace(m: re.Match[str]) -> str:
                    name = m.group(1)
                    if allowed_env is not None and name not in allowed_env:
                        _unexpanded.append(name)
                        return m.group(0)
                    return os.environ.get(name, m.group(0))

                return _ENV_VAR_RE.sub(_replace, value)

            url = server_config.get("url")
            command = server_config.get("command")
            args_raw = server_config.get("args", [])
            env_raw = server_config.get("env", {})
            headers_raw = server_config.get("headers", {})
            oauth_raw = server_config.get("oauth")

            resolved_url = _sub_strict(url) if isinstance(url, str) else None
            resolved_headers = {
                k: _sub_strict(v) for k, v in headers_raw.items() if isinstance(k, str) and isinstance(v, str)
            }

            if unexpanded_env_vars:
                logger.warning(
                    "Plugin '%s' MCP server '%s': leaving environment placeholder(s) %s unexpanded "
                    "in url/headers (the plugin is not allowed to expand them). "
                    "Only the variable names are logged.",
                    entry.name,
                    server_key,
                    sorted(set(unexpanded_env_vars)),
                )

            if missing_env_vars:
                logger.error(
                    "Plugin '%s' MCP server '%s': skipping -- .mcp.json references "
                    "undefined environment variable(s) in url/headers: %s",
                    entry.name,
                    server_key,
                    sorted(set(missing_env_vars)),
                )
                skipped.append(
                    SkippedMcpServer(
                        server_key=server_key,
                        plugin_name=entry.name,
                        missing_env_vars=tuple(sorted(set(missing_env_vars))),
                    )
                )
                continue

            resolved_command = _sub_lenient(command) if isinstance(command, str) else None
            resolved_args = tuple(_sub_lenient(a) for a in args_raw if isinstance(a, str))
            resolved_env = {k: _sub_lenient(v) for k, v in env_raw.items() if isinstance(k, str) and isinstance(v, str)}

            if unexpanded_other_vars:
                logger.warning(
                    "Plugin '%s' MCP server '%s': leaving environment placeholder(s) %s unexpanded "
                    "in command/args/env (the plugin is not allowed to expand them). "
                    "Only the variable names are logged.",
                    entry.name,
                    server_key,
                    sorted(set(unexpanded_other_vars)),
                )

            mcp_entry = PluginMcpServerEntry(
                server_key=server_key,
                plugin_name=entry.name,
                plugin_root=entry.path,
                url=resolved_url,
                command=resolved_command,
                args=resolved_args,
                env=resolved_env,
                headers=resolved_headers,
                description=server_config.get("description")
                if isinstance(server_config.get("description"), str)
                else None,
                display_name=server_config.get("displayName")
                if isinstance(server_config.get("displayName"), str)
                else None,
                auth=server_config.get("auth") if isinstance(server_config.get("auth"), str) else None,
                oauth=oauth_raw if isinstance(oauth_raw, dict) else None,
                visibility=coerce_mcp_visibility(server_config.get("visibility")),
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

        return McpConfigReadResult(entries=tuple(entries), skipped=tuple(skipped))

    def collect_all_mcp_configs(self, entries: Sequence[PluginEntry]) -> list[PluginMcpServerEntry]:
        """Read MCP configs from all plugin entries and return a flat list.

        Skipped servers (see :class:`SkippedMcpServer`) are dropped here --
        this helper has no plugin-scoped destination to attach them to.
        Callers that need skip visibility (``MarketplaceDirectoryLoader``)
        call ``read_mcp_configs`` directly instead.
        """
        result: list[PluginMcpServerEntry] = []
        for entry in entries:
            result.extend(self.read_mcp_configs(entry).entries)
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
        if source.startswith(("./", "../")):
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
            normalized = normalize_skill_name(value=entry.name)
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
