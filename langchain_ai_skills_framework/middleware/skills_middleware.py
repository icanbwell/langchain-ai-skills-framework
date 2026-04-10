from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
    ExtendedModelResponse,
)
from langchain.messages import SystemMessage
from typing import Callable, Any, Awaitable, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)


class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt.

    This is a **singleton** — ``user_id`` is read from the LangGraph
    runtime context on each request (``request.runtime.context["user_id"]``).
    When a ``user_id`` is present and the ``skill_loader`` is a
    ``CompositeSkillLoader``, per-user MongoDB skills are included
    alongside shared skills.
    """

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

        user_id = self._extract_user_id(request)
        skills_block_text: str = await self._get_instructions(user_id=user_id)
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

    async def _get_instructions(self, *, user_id: str | None) -> str:
        """Get skill instructions, including per-user skills when available."""
        if user_id and isinstance(self._skill_loader, CompositeSkillLoader):
            return await self._skill_loader.get_instructions_for_user(user_id=user_id)
        return await self._skill_loader.get_instructions()

    @staticmethod
    def _extract_user_id(request: ModelRequest[Any]) -> str | None:
        """Extract user_id from the LangGraph runtime context."""
        try:
            runtime = getattr(request, "runtime", None)
            if runtime is None:
                return None
            context = getattr(runtime, "context", None)
            if context is None:
                return None
            user_id = context.get("user_id")
            if isinstance(user_id, str) and user_id.strip():
                return user_id.strip()
            return None
        except Exception:
            return None

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
