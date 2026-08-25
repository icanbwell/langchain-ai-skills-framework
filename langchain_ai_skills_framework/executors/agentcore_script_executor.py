from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import anyio
import boto3
from botocore.exceptions import ClientError

from langchain_ai_skills_framework.executors.base_script_executor import (
    BaseScriptExecutor,
)
from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)

logger = logging.getLogger(__name__)

DEFAULT_CODE_INTERPRETER_IDENTIFIER = "aws.codeinterpreter.v1"


class AgentCoreScriptExecutor(BaseScriptExecutor):
    """Runs inline skill scripts inside an AWS Bedrock AgentCore Code Interpreter sandbox.

    Deterministic, code-driven invocation — no LLM in the execution path.
    Defaults to AWS's shared ``aws.codeinterpreter.v1`` sandbox, which has no
    execution role and therefore no AWS credentials inside it.

    Preserves the same script contract as ``MyScriptExecutor``: ``arguments``
    is JSON and delivered to the script over stdin, stdout/stderr/exit code
    come back unchanged. This is done via ``writeFiles`` (script + args) then
    ``executeCommand`` (``python3 script.py < args.json``), since
    ``executeCode`` has no stdin channel of its own.

    See docs/superpowers/specs/2026-08-25-agentcore-script-executor-design.md
    in baileyai-skills-service for the full design rationale.
    """

    def __init__(
        self,
        *,
        code_interpreter_identifier: str = DEFAULT_CODE_INTERPRETER_IDENTIFIER,
        region_name: str = "us-east-1",
        session_timeout_seconds: int = 300,
        max_timeout: int = 300,
        max_output_size: int = 10 * 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        super().__init__(max_timeout=max_timeout, max_output_size=max_output_size)
        self._identifier = code_interpreter_identifier
        self._session_timeout_seconds = session_timeout_seconds
        self._client = client if client is not None else boto3.client("bedrock-agentcore", region_name=region_name)

    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
    ) -> MyScriptExecutionResult:
        if not script.strip():
            raise ValueError("Script content cannot be empty")

        self._validate_argument_keys(arguments=arguments)

        effective_timeout = min(timeout, self.max_timeout)
        start_time = time.perf_counter()

        try:
            with anyio.fail_after(effective_timeout):
                session_id = await self._start_session()
                try:
                    stdout, stderr, exit_code = await self._run_script(
                        session_id=session_id,
                        script=script,
                        arguments=arguments,
                    )
                finally:
                    await self._stop_session(session_id=session_id)
        except TimeoutError:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "AgentCore execution timed out: script=%s, identifier=%s, timeout=%ds",
                script_name,
                self._identifier,
                effective_timeout,
            )
            return MyScriptExecutionResult(
                stdout=None,
                stderr=f"AgentCore execution timed out after {effective_timeout}s",
                exit_code=124,
                execution_time_ms=execution_time_ms,
                success=False,
            )
        except ClientError as exc:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_message = exc.response.get("Error", {}).get("Message", str(exc))
            logger.error(
                "AgentCore API error: script=%s, identifier=%s, error=%s",
                script_name,
                self._identifier,
                error_message,
            )
            return MyScriptExecutionResult(
                stdout=None,
                stderr=f"AgentCore API error: {error_message}",
                exit_code=1,
                execution_time_ms=execution_time_ms,
                success=False,
            )

        execution_time_ms = (time.perf_counter() - start_time) * 1000
        self._check_output_size(output=(stdout or "").encode("utf-8"))
        if stderr:
            self._check_output_size(output=stderr.encode("utf-8"))

        logger.info(
            "AgentCore execution completed: script=%s, identifier=%s, exit_code=%s, duration_ms=%.2f",
            script_name,
            self._identifier,
            exit_code,
            execution_time_ms,
        )

        output = stdout or ""
        if exit_code != 0:
            output += f"\n\nScript exited with code {exit_code}"

        return MyScriptExecutionResult(
            stdout=output.strip(),
            stderr=stderr,
            exit_code=exit_code,
            execution_time_ms=execution_time_ms,
            success=exit_code == 0,
        )

    async def _start_session(self) -> str:
        response = await anyio.to_thread.run_sync(
            lambda: self._client.start_code_interpreter_session(
                codeInterpreterIdentifier=self._identifier,
                name=f"skill-script-{uuid.uuid4()}",
                sessionTimeoutSeconds=self._session_timeout_seconds,
            )
        )
        return str(response["sessionId"])

    async def _stop_session(self, *, session_id: str) -> None:
        try:
            await anyio.to_thread.run_sync(
                lambda: self._client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=self._identifier,
                    sessionId=session_id,
                )
            )
        except ClientError:
            # Best-effort cleanup — the session times out on its own if this fails.
            logger.warning("Failed to stop AgentCore session %s", session_id, exc_info=True)

    async def _invoke(self, *, session_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await anyio.to_thread.run_sync(
            lambda: self._client.invoke_code_interpreter(
                codeInterpreterIdentifier=self._identifier,
                sessionId=session_id,
                name=name,
                arguments=arguments,
            )
        )
        result: dict[str, Any] | None = None
        for event in response["stream"]:
            if "result" in event:
                result = event["result"]
        if result is None:
            raise RuntimeError(f"AgentCore invoke_code_interpreter returned no result for tool={name}")
        return result

    async def _run_script(
        self,
        *,
        session_id: str,
        script: str,
        arguments: dict[str, Any],
    ) -> tuple[str | None, str | None, int]:
        await self._invoke(
            session_id=session_id,
            name="writeFiles",
            arguments={
                "paths": [
                    {"path": "script.py", "text": script},
                    {"path": "args.json", "text": json.dumps(arguments)},
                ]
            },
        )

        result = await self._invoke(
            session_id=session_id,
            name="executeCommand",
            arguments={"command": "python3 script.py < args.json"},
        )
        structured = result.get("structuredContent", result)
        return (
            structured.get("stdout"),
            structured.get("stderr"),
            int(structured.get("exitCode", 1)),
        )
