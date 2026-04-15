from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.middleware.skills_middleware import SkillMiddleware
from langchain_ai_skills_framework.models.skills_model import SkillDetails, SkillSummary


class _StubSkillLoader(SkillLoaderProtocol):
    def __init__(self, summaries: Sequence[SkillSummary]) -> None:
        self._summaries = tuple(summaries)

    def list_skill_summaries(self, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        del allowed_skills
        return self._summaries

    async def list_all_summaries(
        self, *, user_id: str, allowed_skills: set[str]
    ) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills)

    def get_skill_details(self, skill_name: str) -> SkillDetails:  # pragma: no cover
        del skill_name
        raise NotImplementedError

    async def get_skill_details_for_user(
        self, *, user_id: str, skill_name: str
    ) -> SkillDetails:
        return self.get_skill_details(skill_name)

    def refresh(self) -> None:  # pragma: no cover
        return None

    async def get_instructions(self) -> str:
        skills_lines = "\n".join(
            f"<skill><name> {summary.name} </name><description> {summary.description} </description></skill>"
            for summary in self._summaries
        )
        return f"\n\n<available_skills>{skills_lines}</available_skills>\n\n"

    def get_tools(self) -> list[BaseTool]:
        return []

    def read_skill_resource(self, skill_name: str, resource_name: str) -> str:
        raise NotImplementedError()

    async def run_skill_script(
        self, skill_name: str, script_name: str, arguments: dict[str, Any] | None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError()

    def list_skill_script_names(self, skill_name: str) -> Sequence[str]:
        return []

    async def read_skill_resource_for_user(
        self, *, user_id: str, skill_name: str, resource_name: str
    ) -> str:
        return self.read_skill_resource(skill_name, resource_name)

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name, script_name, arguments)

    def list_skill_resource_names(self, skill_name: str) -> Sequence[str]:
        return []


class _DummyModelRequest:
    def __init__(
        self,
        *,
        system_message: SystemMessage | None,
        messages: Sequence[BaseMessage] | None = None,
    ) -> None:
        self.system_message = system_message
        self.messages = tuple(messages) if messages is not None else None

    def override(self, **kwargs: Any) -> "_DummyModelRequest":
        system_message = kwargs.get("system_message", self.system_message)
        messages = kwargs.get("messages", self.messages)
        if messages is not None:
            messages = tuple(messages)
        return _DummyModelRequest(
            system_message=system_message,
            messages=messages,
        )


@pytest.mark.asyncio
async def test_awrap_model_call_inserts_skills_system_message() -> None:
    summaries = [
        SkillSummary(
            name="alpha",
            description="primary",
            source_path=Path("/skills/alpha/SKILL.md"),
        )
    ]
    middleware = SkillMiddleware(skill_loader=_StubSkillLoader(summaries))
    base_system_message = SystemMessage(content="Base instructions")
    follow_up_message = AIMessage(content="Ready")
    request = _DummyModelRequest(
        system_message=base_system_message,
        messages=(base_system_message, follow_up_message),
    )

    captured_request: dict[str, _DummyModelRequest] = {}

    async def handler(model_request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured_request["request"] = cast(_DummyModelRequest, model_request)
        return cast(ModelResponse[Any], AIMessage(content="ok"))

    response = await middleware.awrap_model_call(
        cast(ModelRequest[Any], request),
        handler,
    )

    assert isinstance(response, AIMessage)
    handled_request = captured_request["request"]
    assert handled_request.system_message is base_system_message
    assert handled_request.messages is not None
    assert handled_request.messages[0] is base_system_message
    assert isinstance(handled_request.messages[1], SystemMessage)
    skills_message_content = handled_request.messages[1].content
    assert "<available_skills>" in skills_message_content
    assert "<name> alpha </name>" in skills_message_content
    assert handled_request.messages[2] is follow_up_message


@pytest.mark.asyncio
async def test_awrap_model_call_sets_system_message_when_missing() -> None:
    summaries = [
        SkillSummary(
            name="beta",
            description="secondary",
            source_path=Path("/skills/beta/SKILL.md"),
        )
    ]
    middleware = SkillMiddleware(skill_loader=_StubSkillLoader(summaries))
    request = _DummyModelRequest(system_message=None, messages=None)

    async def handler(model_request: ModelRequest[Any]) -> ModelResponse[Any]:
        assert model_request.system_message is None
        assert model_request.messages is not None
        assert len(model_request.messages) == 1
        assert isinstance(model_request.messages[0], SystemMessage)
        assert "<available_skills>" in model_request.messages[0].content
        assert "beta" in model_request.messages[0].content
        return cast(ModelResponse[Any], AIMessage(content="ok"))

    response = await middleware.awrap_model_call(
        cast(ModelRequest[Any], request),
        handler,
    )

    assert isinstance(response, AIMessage)


@pytest.mark.asyncio
async def test_awrap_model_call_does_not_duplicate_skills_message() -> None:
    summaries = [
        SkillSummary(
            name="gamma",
            description="tertiary",
            source_path=Path("/skills/gamma/SKILL.md"),
        )
    ]
    middleware = SkillMiddleware(skill_loader=_StubSkillLoader(summaries))
    existing_skills_message = SystemMessage(
        content="\n\n<available_skills> <skill><name> gamma </name></skill> </available_skills>\n\n"
    )
    request = _DummyModelRequest(
        system_message=existing_skills_message,
        messages=(existing_skills_message, AIMessage(content="continue")),
    )

    async def handler(model_request: ModelRequest[Any]) -> ModelResponse[Any]:
        assert model_request.messages is not None
        system_messages = [
            message
            for message in model_request.messages
            if isinstance(message, SystemMessage)
        ]
        assert len(system_messages) == 1
        assert system_messages[0] is existing_skills_message
        return cast(ModelResponse[Any], AIMessage(content="ok"))

    response = await middleware.awrap_model_call(
        cast(ModelRequest[Any], request),
        handler,
    )

    assert isinstance(response, AIMessage)


@pytest.mark.asyncio
async def test_awrap_model_call_does_not_duplicate_for_structured_system_content() -> (
    None
):
    summaries = [
        SkillSummary(
            name="delta",
            description="quaternary",
            source_path=Path("/skills/delta/SKILL.md"),
        )
    ]
    middleware = SkillMiddleware(skill_loader=_StubSkillLoader(summaries))
    existing_skills_message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": "Context\n<available_skills> ... </available_skills>",
            }
        ]
    )
    request = _DummyModelRequest(
        system_message=existing_skills_message,
        messages=(existing_skills_message, AIMessage(content="continue")),
    )

    async def handler(model_request: ModelRequest[Any]) -> ModelResponse[Any]:
        assert model_request.messages is not None
        system_messages = [
            message
            for message in model_request.messages
            if isinstance(message, SystemMessage)
        ]
        assert len(system_messages) == 1
        assert system_messages[0] is existing_skills_message
        return cast(ModelResponse[Any], AIMessage(content="ok"))

    response = await middleware.awrap_model_call(
        cast(ModelRequest[Any], request),
        handler,
    )

    assert isinstance(response, AIMessage)
