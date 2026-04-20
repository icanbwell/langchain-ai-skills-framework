import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import (
    PathSecurityError,
    ScriptPermissionError,
)

logger = logging.getLogger(__name__)


class MyShellExecutor:
    """Security-hardened executor for shell scripts.

    Mirrors MyScriptExecutor's security controls:
    - Path validation (traversal prevention, permission checks)
    - Restricted environment (no inherited secrets)
    - Argument key validation
    - Output size limiting
    - Timeout enforcement
    """

    _ARGUMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        allowed_base_dirs: list[Path] | None = None,
        max_timeout: int = 30,
        max_output_size: int = 10 * 1024 * 1024,  # 10MB
    ) -> None:
        self.allowed_base_dirs = allowed_base_dirs or []
        self.max_timeout = max_timeout
        self.max_output_size = max_output_size

    async def execute(
        self,
        *,
        script_path: Path,
        skill_base_dir: Path,
        arguments: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> MyScriptExecutionResult:
        """Execute a shell script with full security controls.

        Args:
            script_path: Path to the .sh script (relative or absolute).
            skill_base_dir: Base directory the script must reside within.
            arguments: Arguments passed as JSON on stdin.
            timeout: Timeout in seconds (capped by max_timeout).

        Returns:
            MyScriptExecutionResult with execution details.

        Raises:
            PathSecurityError: If path validation fails.
            ScriptPermissionError: If script has dangerous permissions.
            ValueError: If argument keys are invalid.
        """
        normalized_arguments = {k.lower(): v for k, v in (arguments or {}).items()}
        self._validate_argument_keys(normalized_arguments)
        validated_path = self._validate_path(script_path, skill_base_dir)

        effective_timeout = min(timeout or self.max_timeout, self.max_timeout)

        return await self._execute_validated(
            script_path=validated_path,
            skill_base_dir=skill_base_dir,
            arguments=normalized_arguments,
            timeout=effective_timeout,
        )

    async def _execute_validated(
        self,
        *,
        script_path: Path,
        skill_base_dir: Path,
        arguments: dict[str, Any],
        timeout: int,
    ) -> MyScriptExecutionResult:
        """Execute a validated shell script with a restricted environment."""
        cmd = ["sh", str(script_path)]

        restricted_env: dict[str, str] = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "SKILL_NAME": script_path.stem,
            "SKILL_BASE_DIR": str(skill_base_dir.resolve()),
        }

        stdin_data = json.dumps(arguments).encode("utf-8")
        start_time = time.monotonic()

        logger.info(
            "Shell script execution requested: path=%s, timeout=%ds",
            script_path,
            timeout,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=restricted_env,
                cwd=str(skill_base_dir.resolve()),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(input=stdin_data), timeout=timeout)
        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Shell script '%s' timed out after %d seconds",
                script_path.name,
                timeout,
            )
            return MyScriptExecutionResult(
                stdout="",
                stderr=f"Script execution timed out after {timeout} seconds",
                exit_code=1,
                execution_time_ms=elapsed_ms,
                success=False,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error("Shell script '%s' failed: %s", script_path.name, exc)
            return MyScriptExecutionResult(
                stdout="",
                stderr=str(exc),
                exit_code=1,
                execution_time_ms=elapsed_ms,
                success=False,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        self._check_output_size(stdout_bytes)
        if stderr_bytes:
            self._check_output_size(stderr_bytes)

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        logger.info(
            "Shell script execution completed: path=%s, exit_code=%d, duration_ms=%.2f",
            script_path.name,
            process.returncode or 0,
            elapsed_ms,
        )

        return MyScriptExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode or 0,
            execution_time_ms=elapsed_ms,
            success=process.returncode == 0,
        )

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

        # Check file permissions
        try:
            stat_info = resolved_path.stat()
            if stat_info.st_mode & 0o002:
                raise ScriptPermissionError(f"Script {resolved_path} is world-writable (insecure)")
        except OSError:
            pass

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
