from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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

    @property
    def namespaced_key(self) -> str:
        """Server key namespaced by plugin name to avoid collisions."""
        return f"{self.plugin_name}__{self.server_key}"

    @property
    def is_http(self) -> bool:
        """Whether this server uses HTTP transport (has a url)."""
        return self.url is not None