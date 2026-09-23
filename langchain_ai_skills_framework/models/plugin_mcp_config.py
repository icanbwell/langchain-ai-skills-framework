from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Trust tier of an MCP server declared in a plugin's .mcp.json.
#
# "internal" (the default) means a b.well-owned server such as
# "fhir-server". "external" marks a partner-owned server so different rules
# can apply to it (e.g. withholding it from config discovery unless a skill
# explicitly declares it needs the server). Purely additive — every
# existing .mcp.json has no "visibility" key and keeps resolving to
# "internal" unchanged.
McpServerVisibility = Literal["internal", "external"]


def coerce_mcp_visibility(value: object) -> McpServerVisibility:
    """Coerce an arbitrary value (e.g. from JSON or MongoDB) to a known visibility.

    Forward-compatible: anything other than the literal string ``"external"``
    (missing, ``None``, or an unrecognized future value) resolves to
    ``"internal"`` rather than raising, so old callers and unknown values
    never break.
    """
    return "external" if value == "external" else "internal"


@dataclass(frozen=True)
class PluginMcpServerEntry:
    """An MCP server declared in a plugin's .mcp.json.

    Each entry represents a single MCP server from a marketplace plugin,
    with its connection details and owning plugin metadata for namespacing.
    """

    server_key: str
    """Key from the plugin's mcpServers dict (e.g., "plugin-database")."""

    plugin_name: str
    """Owning plugin name, used for namespacing (e.g., "my-plugin")."""

    plugin_root: Path
    """Resolved filesystem path to the plugin directory."""

    url: str | None = None
    """HTTP endpoint for the MCP server. Required for server-side use."""

    command: str | None = None
    """Command to launch a stdio-based MCP server (not supported server-side)."""

    args: tuple[str, ...] = ()
    """Arguments passed to command when launching a stdio server."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables for the server process."""

    headers: dict[str, str] = field(default_factory=dict)
    """HTTP headers sent with every request."""

    description: str | None = None
    """Description of the server's capabilities."""

    display_name: str | None = None
    """Human-readable name for UI display."""

    auth: str | None = None
    """Authentication mode (e.g., "oauth", "jwt_token", "headers")."""

    oauth: dict[str, Any] | None = None
    """OAuth configuration dict (clientId, authServerMetadataUrl, etc.)."""

    visibility: McpServerVisibility = "internal"
    """Trust tier: "internal" (default, e.g. fhir-server) or "external"
    (partner-owned). See :data:`McpServerVisibility`."""

    @property
    def namespaced_key(self) -> str:
        """Server key namespaced by plugin name to avoid collisions."""
        return f"{self.plugin_name}__{self.server_key}"

    @property
    def is_http(self) -> bool:
        """Whether this server uses HTTP transport (has a url)."""
        return self.url is not None


@dataclass(frozen=True)
class SkippedMcpServer:
    """An MCP server declared in a plugin's ``.mcp.json`` that was dropped
    during discovery because its ``url``/``headers`` referenced an
    environment variable that isn't set in this process (BAI-859).

    Kept as a first-class, queryable record — rather than only a log line —
    so a misconfiguration that silently drops a server from the tool catalog
    (as opposed to a hard failure) is still visible without log archaeology:
    it's threaded through ``PluginDefinition``, persisted alongside the
    plugin document, and surfaced in ``reload_plugins``'s own summary.
    """

    server_key: str
    """Key from the plugin's mcpServers dict (e.g., "plugin-marketplace")."""

    plugin_name: str
    """Owning plugin name."""

    missing_env_vars: tuple[str, ...]
    """Names of the ${ENV_VAR} placeholders that were unset."""
