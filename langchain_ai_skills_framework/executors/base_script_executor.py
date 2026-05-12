from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class PathSecurityError(Exception):
    """Raised when path validation fails."""


class ScriptPermissionError(Exception):
    """Raised when script has dangerous permissions."""


class BaseScriptExecutor:
    """Shared security controls for script executors.

    Provides path validation (traversal prevention, permission checks),
    argument key validation, and output size limiting.  Subclasses
    implement actual execution via ``_execute_validated`` or similar.
    """

    _ARGUMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        allowed_base_dirs: list[Path] | None = None,
        max_timeout: int = 300,
        max_output_size: int = 10 * 1024 * 1024,  # 10MB
    ) -> None:
        self.allowed_base_dirs = allowed_base_dirs or []
        self.max_timeout = max_timeout
        self.max_output_size = max_output_size

    def _validate_path(self, script_path: Path, skill_base_dir: Path) -> Path:
        """Validate that the script path is safe.

        Raises:
            PathSecurityError: If path validation fails.
            ScriptPermissionError: If script has dangerous permissions.
        """
        try:
            resolved_base = skill_base_dir.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathSecurityError(f"Cannot resolve skill base directory: {e}") from e

        candidate = script_path if script_path.is_absolute() else resolved_base / script_path

        try:
            resolved_path = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathSecurityError(f"Cannot resolve script path: {e}") from e

        if not resolved_path.exists():
            raise PathSecurityError(f"Script does not exist: {resolved_path}")

        if not resolved_path.is_file():
            raise PathSecurityError(f"Script path is not a file: {resolved_path}")

        # Prevent directory traversal
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise PathSecurityError(f"Script path {resolved_path} is outside skill directory {resolved_base}") from None

        # Check allowed_base_dirs constraint
        if self.allowed_base_dirs:
            in_allowed = False
            for base_dir in self.allowed_base_dirs:
                try:
                    resolved_path.relative_to(base_dir.resolve(strict=True))
                    in_allowed = True
                    break
                except (ValueError, OSError, RuntimeError):
                    continue
            if not in_allowed:
                raise PathSecurityError(f"Script path {resolved_path} is not in allowed directories")

        # Check file permissions (Unix-like systems)
        try:
            stat_info = resolved_path.stat()
            if stat_info.st_mode & 0o002:
                raise ScriptPermissionError(f"Script {resolved_path} is world-writable (insecure)")
        except OSError:
            pass  # Permission check not available on this system

        return resolved_path

    def _validate_argument_keys(self, arguments: dict[str, Any]) -> None:
        """Reject argument keys that don't match safe identifier pattern."""
        for key in arguments:
            if not self._ARGUMENT_KEY_PATTERN.fullmatch(key):
                raise ValueError(f"Invalid argument key: {key}")

    def _check_output_size(self, output: bytes) -> None:
        """Prevent memory exhaustion from oversized script output."""
        if len(output) > self.max_output_size:
            raise Exception(f"Script output too large: {len(output)} bytes (max {self.max_output_size})")
