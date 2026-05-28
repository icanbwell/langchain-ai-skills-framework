import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import anyio
from skillkit import SkillMetadata

from langchain_ai_skills_framework.executors.base_script_executor import (
    BaseScriptExecutor,
    PathSecurityError,
    ScriptPermissionError,
)
from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)

logger = logging.getLogger(__name__)

# Re-export so existing ``from my_script_executor import PathSecurityError``
# imports continue to work without changes.
__all__ = [
    "MyScriptExecutor",
    "PathSecurityError",
    "ScriptPermissionError",
]


class MyScriptExecutor(BaseScriptExecutor):
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
        super().__init__(
            allowed_base_dirs=allowed_base_dirs,
            max_timeout=max_timeout,
            max_output_size=max_output_size,
        )

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
            f"Script execution requested: script={script_name}, path={validated_script_path}, timeout={timeout}s"
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
            # --no-config - Don't use any config files at all (uv.toml, pyproject.toml, etc.)
            # --no-progress - Don't show progress bars (cleaner output)
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
                logger.error(f"Script '{script_name}' timed out after {timeout} seconds")
                raise Exception(f"Script '{script_name}' timed out after {timeout} seconds")

            # Check output size before decoding
            self._check_output_size(output=result.stdout)
            if result.stderr:
                self._check_output_size(output=result.stderr)

            output: str = result.stdout.decode("utf-8", errors="replace")
            stderr = None
            if result.stderr:
                stderr = result.stderr.decode("utf-8", errors="replace")

                # Log dependency installation info if present
                if use_uv and stderr:
                    # uv outputs dependency info to stderr
                    if "Resolved" in stderr or "Installed" in stderr or "dependencies" in stderr.lower():
                        logger.info(f"[UV] Dependency installation info for {script_name}:")
                        logger.debug(stderr)

            if result.returncode != 0:
                output += f"\n\nScript exited with code {result.returncode}"

            stdout = output.strip()

        except PermissionError as e:
            logger.error(f"Permission denied executing script '{script_name}' at {validated_script_path}: {e}")
            raise Exception(
                f"Permission denied executing script '{script_name}' at "
                f"{validated_script_path}. Ensure the script and uv binary have "
                f"execute permissions: {e}"
            ) from e
        except FileNotFoundError as e:
            logger.error(f"Command not found for script '{script_name}': {e}")
            raise Exception(f"Command not found. Ensure 'uv' is installed and in PATH: {e}") from e
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
        self._validate_argument_keys(arguments=arguments)
        validated_script_path = self._validate_path(script_path=script_path, skill_base_dir=skill_base_dir)

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

        self._validate_argument_keys(arguments=arguments)

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
                logger.warning(f"Failed to set permissions for temporary script file: {temp_script_path} {e}")
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
                    logger.warning(f"Failed to clean up temporary script file: {temp_script_path}")
