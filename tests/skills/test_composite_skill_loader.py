from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.container.container_factory import (
    _resolve_script_executor,
)
from langchain_ai_skills_framework.executors.my_script_execution_result import (
    MyScriptExecutionResult,
)
from langchain_ai_skills_framework.executors.my_script_executor import (
    MyScriptExecutor,
)
from langchain_ai_skills_framework.loaders.composite_skill_loader import (
    CompositeSkillLoader,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.plugin_definition import PluginDefinition
from langchain_ai_skills_framework.models.plugin_mcp_config import PluginMcpServerEntry
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)


def _make_skill(name: str, *, content: str = "Skill content", source: str = "shared") -> SkillDetails:
    source_path = Path(f"/{source}/{name}/SKILL.md")
    summary = SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path,
        metadata={"source": source},
    )
    return SkillDetails(summary=summary, content=content, source_path=source_path)


class _StubSharedLoader(SkillLoaderProtocol):
    def __init__(self, details: Mapping[str, SkillDetails]) -> None:
        self._details = dict(details)

    def list_skill_summaries(self, *, allowed_skills: set[str]) -> Sequence[SkillSummary]:
        return [d.summary for d in self._details.values()]

    async def list_all_summaries(
        self, *, user_id: str, allowed_skills: set[str], include_staging: bool = False
    ) -> Sequence[SkillSummary]:
        return self.list_skill_summaries(allowed_skills=allowed_skills)

    def get_skill_details(self, *, skill_name: str, plugin_name: str | None = None) -> SkillDetails:
        try:
            return self._details[skill_name]
        except KeyError as exc:
            raise SkillNotFoundError(f"'{skill_name}' not found") from exc

    async def get_skill_details_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> SkillDetails:
        return self.get_skill_details(skill_name=skill_name)

    def refresh(self) -> None:
        pass

    async def refresh_async(self) -> None:
        pass

    async def get_instructions(self) -> str:
        return "<available_skills></available_skills>"

    def read_skill_resource(self, *, skill_name: str, resource_name: str, plugin_name: str | None = None) -> str:
        raise NotImplementedError

    async def run_skill_script(
        self, *, skill_name: str, script_name: str, arguments: dict[str, Any] | None, plugin_name: str | None = None
    ) -> MyScriptExecutionResult:
        raise NotImplementedError

    def list_skill_script_names(self, *, skill_name: str, plugin_name: str | None = None) -> Sequence[str]:
        return []

    async def list_skill_script_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        return self.list_skill_script_names(skill_name=skill_name)

    async def read_skill_resource_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str, resource_name: str
    ) -> str:
        return self.read_skill_resource(skill_name=skill_name, resource_name=resource_name)

    async def run_skill_script_for_user(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
        arguments: dict[str, Any] | None,
    ) -> MyScriptExecutionResult:
        return await self.run_skill_script(skill_name=skill_name, script_name=script_name, arguments=arguments)

    def list_skill_resource_names(self, *, skill_name: str, plugin_name: str | None = None) -> Sequence[str]:
        return []

    async def list_skill_resource_names_for_user(
        self, *, user_id: str, plugin_name: str | None = None, skill_name: str
    ) -> Sequence[str]:
        return self.list_skill_resource_names(skill_name=skill_name)

    async def get_plugin_mcp_configs(self) -> Sequence[PluginMcpServerEntry]:
        return []

    async def list_plugin_definitions(self) -> Sequence[PluginDefinition]:
        return []


def _make_user_loader_mock(
    user_skills: dict[str, SkillDetails] | None = None,
    shared_skills: dict[str, SkillDetails] | None = None,
) -> PluginSkillStore:
    """Create a mock PluginSkillStore that returns a snapshot."""
    loader = AsyncMock(spec=PluginSkillStore)
    skills = user_skills or {}
    shared = shared_skills or {}
    snapshot = SkillSnapshot(
        details_by_name=MappingProxyType(skills),
        ordered_summaries=tuple(sorted([d.summary for d in skills.values()], key=lambda s: s.name)),
    )
    shared_snapshot = SkillSnapshot(
        details_by_name=MappingProxyType(shared),
        ordered_summaries=tuple(sorted([d.summary for d in shared.values()], key=lambda s: s.name)),
    )
    loader.load_snapshot.return_value = snapshot
    loader.load_shared_snapshot.return_value = shared_snapshot
    loader.get_skill_usage_count.return_value = 0
    loader.get_skill_usage_counts.return_value = {}
    loader.get_skill_details.side_effect = lambda *, author, plugin_name, skill_name: (
        skills[skill_name]
        if skill_name in skills
        else (_ for _ in ()).throw(SkillNotFoundError(f"'{skill_name}' not found"))
    )
    return loader


class _FakeScriptExecutor:
    """Test double for ScriptExecutorProtocol that records calls."""

    def __init__(self, *, result: MyScriptExecutionResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def execute_inline_script(
        self,
        *,
        script_name: str,
        script: str,
        arguments: dict[str, Any],
        timeout: int = 30,
    ) -> MyScriptExecutionResult:
        self.calls.append(
            {
                "script_name": script_name,
                "script": script,
                "arguments": arguments,
                "timeout": timeout,
            }
        )
        return self._result


def _make_execution_result(*, stdout: str = "ok") -> MyScriptExecutionResult:
    return MyScriptExecutionResult(
        stdout=stdout,
        stderr=None,
        exit_code=0,
        execution_time_ms=1.23,
        success=True,
    )


class TestCompositeSkillLoaderInit:
    def test_rejects_none_shared_loader(self) -> None:
        user_loader = _make_user_loader_mock()
        with pytest.raises(ValueError, match="shared_loader must not be None"):
            CompositeSkillLoader(shared_loader=None, user_loader=user_loader)  # type: ignore[arg-type]

    def test_rejects_none_user_loader(self) -> None:
        shared = _StubSharedLoader({})
        with pytest.raises(ValueError, match="user_loader must not be None"):
            CompositeSkillLoader(shared_loader=shared, user_loader=None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("pass_explicit_none", [False, True])
    def test_defaults_to_my_script_executor_when_not_injected(self, pass_explicit_none: bool) -> None:
        """Old default behavior is preserved: no injected executor -> MyScriptExecutor."""
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock()

        if pass_explicit_none:
            composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader, script_executor=None)
        else:
            composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        assert isinstance(composite._script_executor, MyScriptExecutor)


class TestListAllSummaries:
    @pytest.mark.asyncio
    async def test_merges_shared_and_user_skills(self) -> None:
        shared_skill = _make_skill("alpha", source="shared")
        user_skill = _make_skill("beta", source="mongodb")

        shared = _StubSharedLoader({"alpha": shared_skill})
        user_loader = _make_user_loader_mock({"beta": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        summaries = await composite.list_all_summaries(user_id="user-1", allowed_skills=set())

        names = [s.name for s in summaries]
        assert "alpha" in names
        assert "beta" in names

    @pytest.mark.asyncio
    async def test_user_skill_overrides_shared_on_name_collision(self) -> None:
        shared_skill = _make_skill("alpha", content="shared version", source="shared")
        user_skill = _make_skill("alpha", content="user version", source="mongodb")

        shared = _StubSharedLoader({"alpha": shared_skill})
        user_loader = _make_user_loader_mock({"alpha": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        summaries = await composite.list_all_summaries(user_id="user-1", allowed_skills=set())

        assert len(summaries) == 1
        assert summaries[0].name == "alpha"
        assert summaries[0].metadata.get("source") == "mongodb"


class TestGetSkillDetailsForUser:
    @pytest.mark.asyncio
    async def test_returns_user_skill_when_exists(self) -> None:
        user_skill = _make_skill("my-skill", content="user version", source="mongodb")
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock({"my-skill": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        detail = await composite.get_skill_details_for_user(
            user_id="user-1", plugin_name="test-plugin", skill_name="my-skill"
        )

        assert detail.content == "user version"

    @pytest.mark.asyncio
    async def test_falls_back_to_shared_when_user_skill_missing(self) -> None:
        shared_skill = _make_skill("shared-skill", content="shared version")
        shared = _StubSharedLoader({"shared-skill": shared_skill})
        user_loader = _make_user_loader_mock({})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        detail = await composite.get_skill_details_for_user(
            user_id="user-1", plugin_name="test-plugin", skill_name="shared-skill"
        )

        assert detail.content == "shared version"

    @pytest.mark.asyncio
    async def test_returns_shared_db_skill_from_another_user(self) -> None:
        shared_db_skill = _make_skill("health-news-monitor", content="shared db version", source="mongodb")
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock(user_skills={}, shared_skills={"health-news-monitor": shared_db_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        detail = await composite.get_skill_details_for_user(
            user_id="different-user", plugin_name="test-plugin", skill_name="health-news-monitor"
        )

        assert detail.content == "shared db version"

    @pytest.mark.asyncio
    async def test_user_skill_takes_precedence_over_shared_db_skill(self) -> None:
        shared_db_skill = _make_skill("my-skill", content="shared db version", source="mongodb")
        user_skill = _make_skill("my-skill", content="user version", source="mongodb")
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock(
            user_skills={"my-skill": user_skill},
            shared_skills={"my-skill": shared_db_skill},
        )
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        detail = await composite.get_skill_details_for_user(
            user_id="user-1", plugin_name="test-plugin", skill_name="my-skill"
        )

        assert detail.content == "user version"

    @pytest.mark.asyncio
    async def test_raises_not_found_when_neither_has_skill(self) -> None:
        shared = _StubSharedLoader({})
        user_loader = _make_user_loader_mock({})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        with pytest.raises(SkillNotFoundError):
            await composite.get_skill_details_for_user(
                user_id="user-1", plugin_name="test-plugin", skill_name="nonexistent"
            )


class TestScriptExecutorInjection:
    """Verifies an injected script_executor is actually used by run_skill_script_for_user.

    Both call sites inside run_skill_script_for_user route through
    _execute_script_content, which is the only place self._script_executor is
    invoked -- one for a user's own Mongo-stored script, one for a shared-DB
    Mongo-stored script owned by another user. Neither path reaches the
    shared filesystem loader (MarketplaceDirectoryLoader), which has its own
    hardcoded executors and is intentionally out of scope here.
    """

    @pytest.mark.asyncio
    async def test_injected_executor_used_for_users_own_mongo_script(self) -> None:
        fake_result = _make_execution_result(stdout="own-script-output")
        fake_executor = _FakeScriptExecutor(result=fake_result)

        shared = _StubSharedLoader({})
        user_loader = cast(AsyncMock, _make_user_loader_mock())
        user_loader.read_script.return_value = "print('hello from user script')"

        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader, script_executor=fake_executor)

        result = await composite.run_skill_script_for_user(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="my-skill",
            script_name="run.py",
            arguments={"x": 1},
        )

        assert result is fake_result
        assert len(fake_executor.calls) == 1
        call = fake_executor.calls[0]
        assert call["script_name"] == "run.py"
        assert call["script"] == "print('hello from user script')"
        assert call["arguments"] == {"x": 1}

        user_loader.read_script.assert_awaited_once_with(
            author="user-1", plugin_name="test-plugin", skill_name="my-skill", script_name="run.py"
        )

    @pytest.mark.asyncio
    async def test_injected_executor_used_for_shared_db_mongo_script(self) -> None:
        fake_result = _make_execution_result(stdout="shared-db-script-output")
        fake_executor = _FakeScriptExecutor(result=fake_result)

        owner_user_id = "owner-user"
        shared_db_skill = SkillDetails(
            summary=SkillSummary(
                name="health-news-monitor",
                description="Description for health-news-monitor",
                source_path=Path("/mongodb/health-news-monitor/SKILL.md"),
                metadata={"source": "mongodb", "user_id": owner_user_id},
            ),
            content="shared db version",
            source_path=Path("/mongodb/health-news-monitor/SKILL.md"),
        )

        shared = _StubSharedLoader({})
        user_loader = cast(
            AsyncMock,
            _make_user_loader_mock(user_skills={}, shared_skills={"health-news-monitor": shared_db_skill}),
        )
        # The requesting user has no script of their own -> falls through to shared DB lookup.
        user_loader.read_script.side_effect = [
            SkillNotFoundError("'health-news-monitor' not found for requesting user"),
            "print('hello from shared db script')",
        ]

        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader, script_executor=fake_executor)

        result = await composite.run_skill_script_for_user(
            user_id="different-user",
            plugin_name="test-plugin",
            skill_name="health-news-monitor",
            script_name="notify.py",
            arguments=None,
        )

        assert result is fake_result
        assert len(fake_executor.calls) == 1
        call = fake_executor.calls[0]
        assert call["script_name"] == "notify.py"
        assert call["script"] == "print('hello from shared db script')"
        assert call["arguments"] == {}

        assert user_loader.read_script.await_count == 2
        second_call = user_loader.read_script.await_args_list[1]
        assert second_call.kwargs == {
            "author": owner_user_id,
            "plugin_name": "test-plugin",
            "skill_name": "health-news-monitor",
            "script_name": "notify.py",
        }


class TestResolveScriptExecutor:
    """Unit tests for container_factory._resolve_script_executor's logging.

    A silently-swallowed ContainerError leaves no trace if a consumer's
    ScriptExecutorProtocol registration is missing/mis-scoped/typo'd, so both
    the "found a consumer-registered executor" and "falling back to default"
    paths must log at some level to be observable.
    """

    def test_logs_and_returns_executor_when_registered(self, caplog: pytest.LogCaptureFixture) -> None:
        fake_executor = _FakeScriptExecutor(result=_make_execution_result())

        class _StubContainer:
            def resolve(self, service_type: type[Any]) -> Any:
                return fake_executor

        with caplog.at_level(logging.INFO, logger="langchain_ai_skills_framework.container.container_factory"):
            resolved = _resolve_script_executor(c=_StubContainer())  # type: ignore[arg-type]

        assert resolved is fake_executor
        assert any(
            "Using consumer-registered ScriptExecutorProtocol" in record.getMessage()
            and "_FakeScriptExecutor" in record.getMessage()
            for record in caplog.records
        )

    def test_logs_and_returns_none_when_not_registered(self, caplog: pytest.LogCaptureFixture) -> None:
        from simple_container.container.simple_container import ContainerError

        class _StubContainer:
            def resolve(self, service_type: type[Any]) -> Any:
                raise ContainerError("ScriptExecutorProtocol not registered")

        with caplog.at_level(logging.DEBUG, logger="langchain_ai_skills_framework.container.container_factory"):
            resolved = _resolve_script_executor(c=_StubContainer())  # type: ignore[arg-type]

        assert resolved is None
        assert any(
            "No ScriptExecutorProtocol registered by consumer" in record.getMessage() for record in caplog.records
        )


class TestGetInstructionsForUser:
    @pytest.mark.asyncio
    async def test_includes_user_skills_in_instructions(self) -> None:
        shared_skill = _make_skill("alpha")
        user_skill = _make_skill("beta", source="mongodb")

        shared = _StubSharedLoader({"alpha": shared_skill})
        user_loader = _make_user_loader_mock({"beta": user_skill})
        composite = CompositeSkillLoader(shared_loader=shared, user_loader=user_loader)

        instructions = await composite.get_instructions_for_user(user_id="user-1")

        assert "<available_skills>" in instructions
        assert "alpha" in instructions
        assert "beta" in instructions
        assert "<usage_count>" in instructions
        assert "save_skill" in instructions
