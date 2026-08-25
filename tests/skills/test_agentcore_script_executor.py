from __future__ import annotations

import json
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from langchain_ai_skills_framework.executors.agentcore_script_executor import (
    AgentCoreScriptExecutor,
)


def _tool_result(*, stdout: str | None = None, stderr: str | None = None, exit_code: int = 0) -> dict[str, Any]:
    return {
        "stream": [
            {
                "result": {
                    "structuredContent": {
                        "stdout": stdout,
                        "stderr": stderr,
                        "exitCode": exit_code,
                    }
                }
            }
        ]
    }


def _fake_client(
    *, write_files_result: dict[str, Any] | None = None, exec_result: dict[str, Any] | None = None
) -> MagicMock:
    client = MagicMock()
    client.start_code_interpreter_session.return_value = {"sessionId": "session-123"}
    client.stop_code_interpreter_session.return_value = {}

    responses = [write_files_result or _tool_result(exit_code=0), exec_result or _tool_result(stdout="ok", exit_code=0)]
    client.invoke_code_interpreter.side_effect = responses
    return client


@pytest.mark.asyncio
async def test_execute_inline_script_maps_success_result() -> None:
    client = _fake_client(exec_result=_tool_result(stdout="hello world", exit_code=0))
    executor = AgentCoreScriptExecutor(client=client)

    result = await executor.execute_inline_script(
        script_name="analyze.py",
        script="print('hello world')",
        arguments={"name": "guillermo"},
    )

    assert result.success is True
    assert result.stdout == "hello world"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_execute_inline_script_writes_script_and_args_before_running() -> None:
    client = _fake_client()
    executor = AgentCoreScriptExecutor(client=client)

    await executor.execute_inline_script(
        script_name="analyze.py",
        script="import sys; print(sys.stdin.read())",
        arguments={"name": "guillermo"},
    )

    write_call = client.invoke_code_interpreter.call_args_list[0]
    assert write_call.kwargs["name"] == "writeFiles"
    content = {c["path"]: c["text"] for c in write_call.kwargs["arguments"]["content"]}
    assert content["script.py"] == "import sys; print(sys.stdin.read())"
    assert json.loads(content["args.json"]) == {"name": "guillermo"}

    exec_call = client.invoke_code_interpreter.call_args_list[1]
    assert exec_call.kwargs["name"] == "executeCommand"
    assert exec_call.kwargs["arguments"] == {"command": "python3 script.py < args.json"}


@pytest.mark.asyncio
async def test_execute_inline_script_starts_and_stops_a_session_per_call() -> None:
    client = _fake_client()
    executor = AgentCoreScriptExecutor(client=client)

    await executor.execute_inline_script(script_name="analyze.py", script="print(1)", arguments={})

    client.start_code_interpreter_session.assert_called_once()
    assert (
        client.start_code_interpreter_session.call_args.kwargs["codeInterpreterIdentifier"] == "aws.codeinterpreter.v1"
    )
    client.stop_code_interpreter_session.assert_called_once_with(
        codeInterpreterIdentifier="aws.codeinterpreter.v1",
        sessionId="session-123",
    )


@pytest.mark.asyncio
async def test_execute_inline_script_stops_session_even_when_run_fails() -> None:
    client = _fake_client()
    client.invoke_code_interpreter.side_effect = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "rate exceeded"}},
        operation_name="InvokeCodeInterpreter",
    )
    executor = AgentCoreScriptExecutor(client=client)

    result = await executor.execute_inline_script(script_name="analyze.py", script="print(1)", arguments={})

    assert result.success is False
    assert "rate exceeded" in (result.stderr or "")
    client.stop_code_interpreter_session.assert_called_once()


@pytest.mark.asyncio
async def test_execute_inline_script_maps_nonzero_exit_code_to_failure() -> None:
    client = _fake_client(exec_result=_tool_result(stdout="partial", stderr="boom", exit_code=1))
    executor = AgentCoreScriptExecutor(client=client)

    result = await executor.execute_inline_script(script_name="analyze.py", script="raise SystemExit(1)", arguments={})

    assert result.success is False
    assert result.exit_code == 1
    assert result.stderr == "boom"
    assert result.stdout is not None and "exited with code 1" in result.stdout


@pytest.mark.asyncio
async def test_execute_inline_script_returns_failure_on_client_error_starting_session() -> None:
    client = _fake_client()
    client.start_code_interpreter_session.side_effect = ClientError(
        error_response={"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
        operation_name="StartCodeInterpreterSession",
    )
    executor = AgentCoreScriptExecutor(client=client)

    result = await executor.execute_inline_script(script_name="analyze.py", script="print(1)", arguments={})

    assert result.success is False
    assert result.exit_code == 1
    assert "not authorized" in (result.stderr or "")


@pytest.mark.asyncio
async def test_execute_inline_script_times_out_and_still_reports_failure() -> None:
    client = _fake_client()

    # session start is fast, so session_id gets bound; the timeout instead fires
    # while draining the (slow) first invoke_code_interpreter call, mirroring the
    # scenario where a session was already created in AWS and must be cleaned up.
    def _slow_invoke(**kwargs: Any) -> dict[str, Any]:
        time.sleep(0.2)
        return _tool_result(exit_code=0)

    client.invoke_code_interpreter.side_effect = _slow_invoke
    executor = AgentCoreScriptExecutor(client=client)

    result = await executor.execute_inline_script(
        script_name="analyze.py",
        script="print(1)",
        arguments={},
        timeout=0.05,  # type: ignore[arg-type]  # short enough to expire mid-invoke, long enough for the fast session start to bind session_id
    )

    assert result.success is False
    assert result.exit_code == 124
    assert "timed out" in (result.stderr or "")
    client.stop_code_interpreter_session.assert_called_once_with(
        codeInterpreterIdentifier="aws.codeinterpreter.v1",
        sessionId="session-123",
    )


@pytest.mark.asyncio
async def test_execute_inline_script_rejects_invalid_argument_keys() -> None:
    executor = AgentCoreScriptExecutor(client=_fake_client())

    with pytest.raises(ValueError, match="Invalid argument key"):
        await executor.execute_inline_script(
            script_name="analyze.py",
            script="print(1)",
            arguments={"not-a-valid-key": 1},
        )


@pytest.mark.asyncio
async def test_execute_inline_script_rejects_empty_script() -> None:
    executor = AgentCoreScriptExecutor(client=_fake_client())

    with pytest.raises(ValueError, match="Script content cannot be empty"):
        await executor.execute_inline_script(script_name="analyze.py", script="   ", arguments={})


class _ThreadCheckingStream:
    """A fake botocore EventStream that records which thread iterated it."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.iterated_thread: threading.Thread | None = None

    def __iter__(self) -> Any:
        self.iterated_thread = threading.current_thread()
        yield {"result": self._result}


@pytest.mark.asyncio
async def test_invoke_drains_the_response_stream_off_the_event_loop() -> None:
    # Regression test for blocking the event loop: draining response["stream"]
    # does blocking HTTP chunk reads, so it must happen inside the same
    # anyio.to_thread.run_sync offload as the invoke call itself, not back on
    # the event loop thread after awaiting just the call.
    client = _fake_client()
    fake_result = {"structuredContent": {"stdout": "ok", "stderr": None, "exitCode": 0}}
    stream = _ThreadCheckingStream(fake_result)
    client.invoke_code_interpreter.side_effect = None
    client.invoke_code_interpreter.return_value = {"stream": stream}
    executor = AgentCoreScriptExecutor(client=client)

    result = await executor._invoke(session_id="session-123", name="executeCommand", arguments={})

    assert result == fake_result
    assert stream.iterated_thread is not None
    assert stream.iterated_thread is not threading.main_thread()


def test_executor_builds_its_own_client_with_bounded_socket_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_boto3_client(service_name: str, **kwargs: Any) -> MagicMock:
        captured["service_name"] = service_name
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(
        "langchain_ai_skills_framework.executors.agentcore_script_executor.boto3.client",
        _fake_boto3_client,
    )

    AgentCoreScriptExecutor()

    assert captured["service_name"] == "bedrock-agentcore"
    config = captured["kwargs"].get("config")
    assert isinstance(config, Config)
    assert getattr(config, "connect_timeout", None) is not None
    assert getattr(config, "read_timeout", None) is not None


def test_executor_does_not_override_an_injected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    injected_client = MagicMock()

    def _fail_if_called(*args: Any, **kwargs: Any) -> MagicMock:
        raise AssertionError("boto3.client should not be called when a client is injected")

    monkeypatch.setattr(
        "langchain_ai_skills_framework.executors.agentcore_script_executor.boto3.client",
        _fail_if_called,
    )

    executor = AgentCoreScriptExecutor(client=injected_client)

    assert executor._client is injected_client
