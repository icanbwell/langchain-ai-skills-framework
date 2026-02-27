from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from langchain_core.messages import AIMessage, BaseMessage

from aiskills.skills.middleware.skills_middleware import SkillMiddleware
from aiskills.skills.models.skills_model import SkillDetails, SkillSummary


class _StubSkillLoader:
    def __init__(self, summaries: Sequence[SkillSummary]) -> None:
        self._summaries = tuple(summaries)

    def list_skill_summaries(self) -> Sequence[SkillSummary]:
        return self._summaries

    def get_skill_details(self, skill_name: str) -> SkillDetails:  # pragma: no cover
        raise NotImplementedError

    def refresh(self) -> None:  # pragma: no cover
        return None


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
        assert model_request.system_message is not None
        assert isinstance(model_request.system_message, SystemMessage)
        assert "<available_skills>" in model_request.system_message.content
        assert "beta" in model_request.system_message.content
        return cast(ModelResponse[Any], AIMessage(content="ok"))

    response = await middleware.awrap_model_call(
        cast(ModelRequest[Any], request),
        handler,
    )

    assert isinstance(response, AIMessage)
