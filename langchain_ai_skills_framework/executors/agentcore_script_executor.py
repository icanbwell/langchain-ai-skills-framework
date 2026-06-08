from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import anyio
import boto3
from botocore.exceptions import ClientError

from langchain_ai_skills_framework.executors.agentcore_config import AgentCoreConfig
from langchain_ai_skills_framework.executors.base_script_executor import (
    BaseScriptExecutor,
)
from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Execute this Python script. Pass the following JSON as stdin: {arguments_json}
Return only the script's stdout output. Do not add commentary.

```python
{script_content}
```"""


class AgentCoreScriptExecutor(BaseScriptExecutor):
    def __init__(self, *, config: AgentCoreConfig) -> None:
        super().__init__(max_timeout=config.timeout)
        self._config = config
        self._client = boto3.client(
            "bedrock-agent-runtime",
            region_name=config.region,
        )

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

        effective_timeout = min(timeout, self._config.timeout)
        session_id = str(uuid.uuid4())
        input_text = _PROMPT_TEMPLATE.format(
            arguments_json=json.dumps(arguments),
            script_content=script,
        )

        logger.info(
            "AgentCore execution requested: script=%s, harness=%s, session=%s, timeout=%ds",
            script_name,
            self._config.harness_id,
            session_id,
            effective_timeout,
        )

        start_time = time.perf_counter()

        try:
            output = await self._invoke_harness(
                input_text=input_text,
                session_id=session_id,
                timeout=effective_timeout,
            )
        except TimeoutError:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "AgentCore execution timed out: script=%s, session=%s",
                script_name,
                session_id,
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
            error_message = exc.response["Error"]["Message"]
            logger.error(
                "AgentCore API error: script=%s, error=%s",
                script_name,
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

        logger.info(
            "AgentCore execution completed: script=%s, session=%s, duration_ms=%.2f",
            script_name,
            session_id,
            execution_time_ms,
        )

        return MyScriptExecutionResult(
            stdout=output.strip() if output else None,
            stderr=None,
            exit_code=0,
            execution_time_ms=execution_time_ms,
            success=True,
        )

    async def _invoke_harness(
        self,
        *,
        input_text: str,
        session_id: str,
        timeout: int,
    ) -> str:
        response: dict[str, Any] | None = None

        with anyio.fail_after(timeout):
            response = await anyio.to_thread.run_sync(
                lambda: self._client.invoke_agent(
                    agentId=self._config.harness_id,
                    agentAliasId=self._config.alias_id,
                    sessionId=session_id,
                    inputText=input_text,
                )
            )

        if response is None:
            raise TimeoutError("AgentCore invocation timed out")

        chunks: list[str] = []
        for event in response.get("completion", []):
            if "chunk" in event:
                chunk_bytes = event["chunk"].get("bytes", b"")
                if chunk_bytes:
                    chunks.append(chunk_bytes.decode("utf-8") if isinstance(chunk_bytes, bytes) else chunk_bytes)

        return "".join(chunks)
