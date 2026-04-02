import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import anyio
from skillkit import SkillMetadata

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)

logger = logging.getLogger(__name__)


class PathSecurityError(Exception):
    """Raised when path validation fails"""

    pass


class ScriptPermissionError(Exception):
    """Raised when script has dangerous permissions"""

    pass


class MyScriptExecutor:
    _ARGUMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        allowed_base_dirs: list[Path] | None = None,
        max_timeout: int = 300,  # 5 minutes max
        max_output_size: int = 10 * 1024 * 1024,  # 10MB max output
    ) -> None:
        """
        Initialize executor with security constraints.

        Args:
            allowed_base_dirs: List of directories where scripts are allowed to run
            max_timeout: Maximum allowed timeout in seconds
            max_output_size: Maximum allowed output size in bytes
        """
        self.allowed_base_dirs = allowed_base_dirs or []
        self.max_timeout = max_timeout
        self.max_output_size = max_output_size

    def _validate_path(self, script_path: Path, skill_base_dir: Path) -> Path:
        """
        Validate that the script path is safe.

        Raises:
            PathSecurityError: If path validation fails
        """
        # Resolve the skill directory first, then anchor relative script paths to it.
        try:
            resolved_skill_base_dir = skill_base_dir.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathSecurityError(f"Cannot resolve skill base directory: {e}") from e

        candidate_script_path = (
            script_path
            if script_path.is_absolute()
            else resolved_skill_base_dir.joinpath(script_path)
        )

        # Resolve to absolute path to prevent directory traversal
        try:
            resolved_path = candidate_script_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise PathSecurityError(f"Cannot resolve script path: {e}") from e

        # Check if script exists
        if not resolved_path.exists():
            raise PathSecurityError(f"Script does not exist: {resolved_path}")

        # Check if it's a file (not a directory or symlink to something dangerous)
        if not resolved_path.is_file():
            raise PathSecurityError(f"Script path is not a file: {resolved_path}")

        # Prevent directory traversal - ensure script is within skill_base_dir
        try:
            resolved_path.relative_to(resolved_skill_base_dir)
        except ValueError:
            raise PathSecurityError(
                f"Script path {resolved_path} is outside skill directory {skill_base_dir}"
            ) from None

        # Check if allowed_base_dirs is set and validate
        if self.allowed_base_dirs:
            is_in_allowed_dir = False
            for base_dir in self.allowed_base_dirs:
                try:
                    resolved_path.relative_to(base_dir.resolve(strict=True))
                    is_in_allowed_dir = True
                    break
                except (ValueError, OSError, RuntimeError):
                    continue
            if not is_in_allowed_dir:
                raise PathSecurityError(
                    f"Script path {resolved_path} is not in allowed directories"
                )

        # Check file permissions (Unix-like systems)
        try:
            stat_info = resolved_path.stat()
            # Check if file is world-writable (dangerous)
            if stat_info.st_mode & 0o002:
                raise ScriptPermissionError(
                    f"Script {resolved_path} is world-writable (insecure)"
                )
        except OSError:
            pass  # Permission check not available on this system

        return resolved_path

    def _check_output_size(self, output: bytes) -> None:
        """Prevent memory exhaustion from large outputs"""
        if len(output) > self.max_output_size:
            raise Exception(
                f"Script output too large: {len(output)} bytes "
                f"(max {self.max_output_size})"
            )

    def _validate_argument_keys(self, arguments: dict[str, Any]) -> None:
        for key in arguments:
            if not self._ARGUMENT_KEY_PATTERN.fullmatch(key):
                raise ValueError(f"Invalid argument key: {key}")

    async def _execute_validated_script(
        self,
        *,
        script_name: str,
        validated_script_path: Path,
        arguments: dict[str, Any],
        skill_base_dir: Path | None,
        timeout: int,
        use_uv: bool,
    ) -> MyScriptExecutionResult:
        """Execute a validated script path with constrained runtime settings."""
        # Enforce maximum timeout
        timeout = min(timeout, self.max_timeout)

        # Log execution attempt
        logger.info(
            f"Script execution requested: "
            f"script={script_name}, "
            f"path={validated_script_path}, "
            f"timeout={timeout}s"
        )

        # Start timing
        start_time = time.perf_counter()

        cmd: list[str]

        if use_uv:
            # https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies
            # https://docs.astral.sh/uv/reference/cli/#uv-run
            # Add isolation flags for maximum security
            # uv - The main command, invoking the uv tool
            # run - Subcommand that runs a Python script or command in an isolated environment
            # --isolated - Don't discover project configuration
            # Prevents uv from searching for and using project-level configuration files (like pyproject.toml or uv.toml)
            # Ensures the command runs independently of any project settings that might exist in parent directories
            # --no-project - Don't use project environment
            # Prevents uv from using the project's virtual environment or dependencies
            # Forces uv to create a completely separate, temporary environment for this execution
            # Useful when you want to run something without interference from the current project's setup
            # -v - Verbose output
            # Enables verbose logging to see detailed information about what's happening
            # Shows dependency installation progress, resolution steps, and other diagnostic information
            # Helpful for debugging or understanding what packages are being installed
            cmd = [
                "uv",
                "run",
                # "--isolated",  # Don't discover project config
                "--no-project",  # Don't use project environment
                "--no-config",  # Don't use any config files at all (uv.toml, pyproject.toml, etc.)
                "--no-progress",  # Don't show progress bars (cleaner output)
                # "-v",  # Verbose to see dependency installation
                str(validated_script_path),
            ]
        else:
            cmd = [str(validated_script_path)]

        stdin_payload = arguments if arguments else {}
        stdin_data: bytes = json.dumps(stdin_payload).encode("utf-8")
        cwd = str(skill_base_dir.resolve()) if skill_base_dir else None

        # Create maximally restricted environment
        env: dict[str, str] = {
            # Minimal required environment
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            # UV isolation settings
            "UV_SYSTEM_PYTHON": "0",  # Don't use system packages
            # Python isolation settings
            "PYTHONNOUSERSITE": "1",  # Ignore user site-packages
            "PYTHONPATH": "",  # Clear PYTHONPATH
            # Skill-specific safe env vars
            "SKILL_NAME": script_name,
        }

        if cwd is not None:
            env["SKILL_BASE_DIR"] = cwd

        try:
            result = None
            with anyio.move_on_after(timeout) as scope:
                result = await anyio.run_process(
                    cmd,
                    check=False,
                    cwd=cwd,
                    input=stdin_data,
                    env=env,  # Use restricted environment
                )

            if scope.cancelled_caught or result is None:
                logger.error(
                    f"Script '{script_name}' timed out after {timeout} seconds"
                )
                raise Exception(
                    f"Script '{script_name}' timed out after {timeout} seconds"
                )

            # Check output size before decoding
            self._check_output_size(result.stdout)
            if result.stderr:
                self._check_output_size(result.stderr)

            output: str = result.stdout.decode("utf-8", errors="replace")
            stderr = None
            if result.stderr:
                stderr = result.stderr.decode("utf-8", errors="replace")

                # Log dependency installation info if present
                if use_uv and stderr:
                    # uv outputs dependency info to stderr
                    if (
                        "Resolved" in stderr
                        or "Installed" in stderr
                        or "dependencies" in stderr.lower()
                    ):
                        logger.info(
                            f"[UV] Dependency installation info for {script_name}:"
                        )
                        logger.debug(stderr)

            if result.returncode != 0:
                output += f"\n\nScript exited with code {result.returncode}"

            stdout = output.strip()

        except PermissionError as e:
            logger.error(
                f"Permission denied executing script '{script_name}' at "
                f"{validated_script_path}: {e}"
            )
            raise Exception(
                f"Permission denied executing script '{script_name}' at "
                f"{validated_script_path}. Ensure the script and uv binary have "
                f"execute permissions: {e}"
            ) from e
        except FileNotFoundError as e:
            logger.error(f"Command not found for script '{script_name}': {e}")
            raise Exception(
                f"Command not found. Ensure 'uv' is installed and in PATH: {e}"
            ) from e
        except OSError as e:
            logger.error(f"Failed to execute script '{script_name}': {e}")
            raise Exception(f"Failed to execute script '{script_name}': {e}") from e

        # Calculate execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Log execution completion
        logger.info(
            f"Script execution completed: "
            f"script={script_name}, "
            f"exit_code={result.returncode}, "
            f"duration_ms={execution_time_ms:.2f}"
        )

        # Return result
        return MyScriptExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
            execution_time_ms=execution_time_ms,
            success=result.returncode == 0,
        )

    # noinspection PyMethodMayBeStatic
    async def execute_script_from_path(
        self,
        *,
        script_name: str,
        script_path: Path,
        arguments: dict[str, Any],
        skill_base_dir: Path,
        skill_metadata: SkillMetadata,
        timeout: int = 30,
        use_uv: bool = True,
    ) -> MyScriptExecutionResult:
        """Execute a script with security controls.

        Args:
            script_name (str): Name of the script
            script_path: Path to the script (relative or absolute)
            arguments: Arguments to pass as JSON on stdin
            skill_base_dir: Base directory of the skill
            skill_metadata: SkillMetadata instance
            timeout: Timeout in seconds
            use_uv: Use UV for isolated execution

        Returns:
            ScriptExecutionResult with execution details

        Raises:
            PathSecurityError: If path validation fails
            ScriptPermissionError: If script has dangerous permissions
            ValueError: If arguments are invalid
            Exception: If script execution fails

        """
        # Security validations
        self._validate_argument_keys(arguments)
        validated_script_path = self._validate_path(script_path, skill_base_dir)

        return await self._execute_validated_script(
            script_name=script_name,
            validated_script_path=validated_script_path,
            arguments=arguments,
            skill_base_dir=skill_base_dir,
            timeout=timeout,
            use_uv=use_uv,
        )

    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
        use_uv: bool = True,
    ) -> MyScriptExecutionResult:
        """Execute inline script content with the same controls as execute()."""

        if not script.strip():
            raise ValueError("Script content cannot be empty")

        self._validate_argument_keys(arguments)

        temp_script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".py",
                prefix=".tmp_skill_script_",
                delete=False,
            ) as temp_file:
                temp_file.write(script)
                temp_script_path = Path(temp_file.name)

            if not temp_script_path:
                raise ValueError("Script content cannot be empty")

            try:
                temp_script_path.chmod(0o700)
            except OSError as e:
                logger.warning(
                    f"Failed to set permissions for temporary script file: {temp_script_path} {e}"
                )
                pass

            return await self._execute_validated_script(
                script_name=script_name,
                validated_script_path=temp_script_path,
                arguments=arguments,
                skill_base_dir=None,
                timeout=timeout,
                use_uv=use_uv,
            )
        finally:
            if temp_script_path is not None:
                try:
                    temp_script_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        f"Failed to clean up temporary script file: {temp_script_path}"
                    )
