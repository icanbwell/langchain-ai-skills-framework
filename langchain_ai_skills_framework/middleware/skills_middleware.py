from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
    ExtendedModelResponse,
)
from langchain.messages import SystemMessage
from typing import Callable, Any, Awaitable, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from langchain_ai_skills_framework.loaders.skill_loader import (
    SkillLoaderProtocol,
)


class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    _SKILLS_BLOCK_MARKER = "<available_skills>"

    def __init__(self, skill_loader: SkillLoaderProtocol) -> None:
        """Initialize and generate the skills prompt from the configured directory."""

        self._skill_loader = skill_loader
        if skill_loader is None:
            raise ValueError("skill_loader must not be None")
        if not isinstance(skill_loader, SkillLoaderProtocol):
            raise TypeError(
                f"skill_loader must be SkillLoaderProtocol, got {type(skill_loader)}"
            )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Async: Inject skill descriptions into system prompt."""
        existing_messages: list[AnyMessage] = list(request.messages or ())
        if self._request_has_skills_message(existing_messages):
            return await handler(request)

        skills_block_text = self._skill_loader.get_instructions()
        skills_message = SystemMessage(content=skills_block_text)

        insertion_index = 0
        for idx, message in enumerate(existing_messages):
            if isinstance(message, SystemMessage):
                # insert after the first system message to ensure the skills information is included in the
                # initial instructions but doesn't override any existing system-level context
                insertion_index = idx + 1
                break

        existing_messages.insert(insertion_index, skills_message)
        modified_request = request.override(messages=list(existing_messages))
        return await handler(modified_request)

    @classmethod
    def _request_has_skills_message(cls, messages: Sequence[AnyMessage]) -> bool:
        for message in messages:
            if not isinstance(message, SystemMessage):
                continue
            if cls._content_contains_skills_marker(message.content):
                return True
        return False

    @classmethod
    def _content_contains_skills_marker(cls, content: object) -> bool:
        if isinstance(content, str):
            return cls._SKILLS_BLOCK_MARKER in content
        if isinstance(content, (list, tuple)):
            return any(cls._content_contains_skills_marker(item) for item in content)
        if isinstance(content, dict):
            return any(
                cls._content_contains_skills_marker(item) for item in content.values()
            )
        return False
