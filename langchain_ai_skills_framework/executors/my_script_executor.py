import time
from pathlib import Path
from typing import Any

import anyio
from skillkit import SkillMetadata

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)


class MyScriptExecutor:
    # noinspection PyMethodMayBeStatic
    async def execute(
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
            arguments: Arguments to pass as JSON via stdin
            skill_base_dir: Base directory of the skill
            skill_metadata: SkillMetadata instance
            timeout: Timeout in seconds
            use_uv: Use UV

        Returns:
            ScriptExecutionResult with execution details

        Raises:
            PathSecurityError: If path validation fails
            ScriptPermissionError: If script has dangerous permissions
            InterpreterNotFoundError: If interpreter not found
            ArgumentSerializationError: If arguments cannot be serialized
            ArgumentSizeError: If arguments too large

        """
        # Start timing
        start_time = time.perf_counter()

        cmd: list[str]

        # Build command
        if use_uv:
            cmd = ["uv", "run", "-v", script_path.as_posix()]
        else:
            cmd = [script_path.as_posix()]

        if arguments:
            for key, value in arguments.items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                elif isinstance(value, list):
                    for item in value:
                        cmd.append(f"--{key}")
                        cmd.append(str(item))
                elif value is not None:
                    cmd.append(f"--{key}")
                    cmd.append(str(value))

        stdin_data: bytes | None = None
        cwd = str(skill_base_dir.absolute())

        try:
            result = None
            with anyio.move_on_after(timeout) as scope:
                result = await anyio.run_process(
                    cmd,
                    check=False,
                    cwd=cwd,
                    input=stdin_data,
                )

            if scope.cancelled_caught or result is None:
                raise Exception(
                    f"Script '{script_name}' timed out after {timeout} seconds"
                )

            output: str = result.stdout.decode("utf-8", errors="replace")
            stderr = None
            if result.stderr:
                stderr = result.stderr.decode("utf-8", errors="replace")
                # output += f'\n\nStderr:\n{stderr}'

            if result.returncode != 0:
                output += f"\n\nScript exited with code {result.returncode}"

            stdout = output.strip()

        except OSError as e:
            raise Exception(f"Failed to execute script '{script_name}': {e}") from e

        # Calculate execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Return result
        return MyScriptExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
            execution_time_ms=execution_time_ms,
            success=result.returncode == 0,
        )
