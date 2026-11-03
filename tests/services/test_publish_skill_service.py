from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    PluginSkillStore,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import (
    GitHubMarketplacePublisher,
)
from langchain_ai_skills_framework.services.publish_skill_service import (
    PublishSkillService,
)
from langchain_ai_skills_framework.services.skill_operation_error import (
    SkillOperationError,
)


def _make_doc(
    *,
    skill_name: str = "test-skill",
    state: str = "in_review",
    plugin_name: str = "test-plugin",
) -> MongoPluginSkillDocument:
    return MongoPluginSkillDocument(
        plugin_name=plugin_name,
        author="user-1",
        skill_name=skill_name,
        path=f"{plugin_name}/skills/{skill_name}/SKILL.md",
        description="A test skill",
        content="# Test\nContent",
        state=state,
        modified_by="user-1",
        date_created=datetime.now(UTC),
        date_modified=datetime.now(UTC),
    )


def _make_skill_details(
    *,
    skill_name: str = "test-skill",
    state: str = "staging",
    plugin_name: str = "test-plugin",
    path: str = "",
) -> SkillDetails:
    summary = SkillSummary(
        name=skill_name,
        description="A test skill",
        plugin_name=plugin_name,
        state=state,
        path=path,
    )
    return SkillDetails(
        summary=summary,
        content="# Test\nContent",
    )


class TestPublishSkillService:
    @pytest.mark.asyncio
    async def test_publish_validates_staging_state(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = True
        store.get_skill_details.return_value = _make_skill_details(state="staging")
        store.set_skill_state.return_value = _make_doc(state="in_review")
        service = PublishSkillService(mongo_skill_loader=store)

        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=True,
        )

        assert "submitted for review" in result
        store.skill_exists.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
        )
        store.get_skill_details.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
        )
        store.set_skill_state.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            state="in_review",
            published_branch=None,
        )

    @pytest.mark.asyncio
    async def test_publish_rejects_non_staging_state(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = True
        store.get_skill_details.return_value = _make_skill_details(state="draft")
        service = PublishSkillService(mongo_skill_loader=store)

        with pytest.raises(SkillOperationError, match="must be in 'staging' state"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                published=True,
            )

    @pytest.mark.asyncio
    async def test_publish_auto_saves_when_skill_not_exists(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = False
        store.get_skill_details.return_value = _make_skill_details(state="staging")
        store.set_skill_state.return_value = _make_doc(state="in_review")

        loader = AsyncMock(spec=SkillLoaderProtocol)
        loader.get_skill_details_for_user.return_value = _make_skill_details()

        service = PublishSkillService(
            mongo_skill_loader=store,
            skill_loader=loader,
        )

        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=True,
        )

        assert "submitted for review" in result
        loader.get_skill_details_for_user.assert_awaited_once_with(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
        )
        store.save_skill.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            content="# Test\nContent",
            state="staging",
            modified_by="user-1",
        )
        store.set_skill_state.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            state="in_review",
            published_branch=None,
        )

    @pytest.mark.asyncio
    async def test_publish_raises_when_skill_not_found_and_no_loader(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = False
        service = PublishSkillService(mongo_skill_loader=store, skill_loader=None)

        with pytest.raises(
            SkillOperationError,
            match="does not exist and no skill_loader is configured",
        ):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                published=True,
            )

    @pytest.mark.asyncio
    async def test_publish_raises_when_loader_cannot_find_skill(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = False

        loader = AsyncMock(spec=SkillLoaderProtocol)
        loader.get_skill_details_for_user.side_effect = SkillNotFoundError("Skill not found")

        service = PublishSkillService(
            mongo_skill_loader=store,
            skill_loader=loader,
        )

        with pytest.raises(SkillOperationError, match="not found in skill_loader"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                published=True,
            )

    @pytest.mark.asyncio
    async def test_unpublish_sets_draft_state(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.set_skill_state.return_value = _make_doc(state="draft")
        service = PublishSkillService(mongo_skill_loader=store)

        result = await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=False,
        )

        assert "unpublished" in result
        store.set_skill_state.assert_awaited_once_with(
            author="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            state="draft",
            published_branch=None,
        )
        # Should NOT call skill_exists or get_skill_details for unpublish
        store.skill_exists.assert_not_awaited()
        store.get_skill_details.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_empty_user_id(self) -> None:
        service = PublishSkillService(mongo_skill_loader=AsyncMock())

        with pytest.raises(SkillOperationError, match="user_id is required"):
            await service.execute(
                user_id="",
                plugin_name="test-plugin",
                skill_name="test-skill",
                published=True,
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_skill_name(self) -> None:
        service = PublishSkillService(mongo_skill_loader=AsyncMock())

        with pytest.raises(SkillOperationError, match="skill_name must be a non-empty"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="  ",
                published=True,
            )

    @pytest.mark.asyncio
    async def test_rejects_when_store_not_configured(self) -> None:
        service = PublishSkillService(mongo_skill_loader=None)

        with pytest.raises(SkillOperationError, match="mongo_skill_loader is not configured"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                published=True,
            )

    @pytest.mark.asyncio
    async def test_wraps_unexpected_exception(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = True
        store.get_skill_details.side_effect = RuntimeError("db down")
        service = PublishSkillService(mongo_skill_loader=store)

        with pytest.raises(SkillOperationError, match="Unable to update publishing"):
            await service.execute(
                user_id="user-1",
                plugin_name="test-plugin",
                skill_name="test-skill",
                published=True,
            )


def _make_resource_doc(*, name: str, path: str, content: str = "ref") -> MongoPluginResourceDocument:
    return MongoPluginResourceDocument(
        plugin_name="test-plugin",
        skill_name="test-skill",
        resource_name=name,
        path=path,
        content=content,
        author="user-1",
    )


def _make_script_doc(*, name: str, path: str, content: str = "code") -> MongoPluginScriptDocument:
    return MongoPluginScriptDocument(
        plugin_name="test-plugin",
        skill_name="test-skill",
        script_name=name,
        path=path,
        content=content,
        author="user-1",
    )


class TestPublishSkillServicePathPropagation:
    """Verify that stored materialized paths flow through to the publisher."""

    @pytest.mark.asyncio
    async def test_publish_forwards_stored_paths_for_custom_folder(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = True
        store.get_skill_details.return_value = _make_skill_details(
            state="staging", path="test-plugin/skills/folder/test-skill/SKILL.md"
        )
        store.set_skill_state.return_value = _make_doc(state="in_review")
        store.list_resource_documents.return_value = [
            _make_resource_doc(name="REF.md", path="test-plugin/skills/folder/test-skill/references/REF.md"),
        ]
        store.list_script_documents.return_value = [
            _make_script_doc(name="run.py", path="test-plugin/skills/folder/test-skill/scripts/run.py"),
        ]

        publisher = AsyncMock(spec=GitHubMarketplacePublisher)
        publisher.use_branch = False
        publisher.publish_skill.return_value = "abc123"

        service = PublishSkillService(mongo_skill_loader=store, marketplace_publisher=publisher)

        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=True,
        )

        # Wait for the background task to complete
        pending = list(PublishSkillService._pending_tasks.values())
        for task in pending:
            await task

        publisher.publish_skill.assert_awaited_once()
        kwargs = publisher.publish_skill.await_args.kwargs
        assert kwargs["skill_path"] == "plugins/test-plugin/skills/folder/test-skill/SKILL.md"
        assert kwargs["resource_paths"] == {"REF.md": "plugins/test-plugin/skills/folder/test-skill/references/REF.md"}
        assert kwargs["script_paths"] == {"run.py": "plugins/test-plugin/skills/folder/test-skill/scripts/run.py"}

    @pytest.mark.asyncio
    async def test_publish_omits_path_overrides_when_no_stored_path(self) -> None:
        # When stored paths are empty, the publisher must fall back to its default layout.
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = True
        store.get_skill_details.return_value = _make_skill_details(state="staging", path="")
        store.set_skill_state.return_value = _make_doc(state="in_review")
        store.list_resource_documents.return_value = [_make_resource_doc(name="REF.md", path="")]
        store.list_script_documents.return_value = [_make_script_doc(name="run.py", path="")]

        publisher = AsyncMock(spec=GitHubMarketplacePublisher)
        publisher.use_branch = False
        publisher.publish_skill.return_value = "abc123"

        service = PublishSkillService(mongo_skill_loader=store, marketplace_publisher=publisher)
        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=True,
        )
        for task in list(PublishSkillService._pending_tasks.values()):
            await task

        kwargs = publisher.publish_skill.await_args.kwargs
        assert kwargs["skill_path"] is None
        assert kwargs["resource_paths"] is None
        assert kwargs["script_paths"] is None

    @pytest.mark.asyncio
    async def test_unpublish_forwards_resolved_skill_dir(self) -> None:
        store = AsyncMock(spec=PluginSkillStore)
        store.set_skill_state.return_value = _make_doc(state="draft")
        store.get_skill_details.return_value = _make_skill_details(
            state="draft", path="test-plugin/skills/folder/test-skill/SKILL.md"
        )

        publisher = AsyncMock(spec=GitHubMarketplacePublisher)
        publisher.use_branch = False
        publisher.unpublish_skill.return_value = "abc"

        service = PublishSkillService(mongo_skill_loader=store, marketplace_publisher=publisher)
        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=False,
        )
        for task in list(PublishSkillService._pending_tasks.values()):
            await task

        publisher.unpublish_skill.assert_awaited_once_with(
            plugin_name="test-plugin",
            skill_name="test-skill",
            user_id="user-1",
            skill_dir="plugins/test-plugin/skills/folder/test-skill",
        )

    @pytest.mark.asyncio
    async def test_unpublish_resolves_skill_dir_to_none_when_skill_missing(self) -> None:
        # If the skill is not in the store, fall back to the publisher's default layout.
        store = AsyncMock(spec=PluginSkillStore)
        store.set_skill_state.return_value = _make_doc(state="draft")
        store.get_skill_details.side_effect = SkillNotFoundError("not found")

        publisher = AsyncMock(spec=GitHubMarketplacePublisher)
        publisher.use_branch = False
        publisher.unpublish_skill.return_value = "abc"

        service = PublishSkillService(mongo_skill_loader=store, marketplace_publisher=publisher)
        await service.execute(
            user_id="user-1",
            plugin_name="test-plugin",
            skill_name="test-skill",
            published=False,
        )
        for task in list(PublishSkillService._pending_tasks.values()):
            await task

        publisher.unpublish_skill.assert_awaited_once_with(
            plugin_name="test-plugin",
            skill_name="test-skill",
            user_id="user-1",
            skill_dir=None,
        )


class TestPublishSkillServiceTaskIsolation:
    """A background publish is replaced only by a repeat publish from the same author (BAI-1510)."""

    @staticmethod
    def _build_service() -> tuple[PublishSkillService, AsyncMock, asyncio.Event]:
        store = AsyncMock(spec=PluginSkillStore)
        store.skill_exists.return_value = True
        store.get_skill_details.return_value = _make_skill_details(state="staging")
        store.set_skill_state.return_value = _make_doc(state="in_review")
        store.list_resource_documents.return_value = []
        store.list_script_documents.return_value = []

        release = asyncio.Event()

        async def _blocked_publish(**_kwargs: object) -> str:
            await release.wait()
            return "abc123"

        publisher = AsyncMock(spec=GitHubMarketplacePublisher)
        publisher.use_branch = False
        publisher.publish_skill.side_effect = _blocked_publish
        service = PublishSkillService(mongo_skill_loader=store, marketplace_publisher=publisher)
        return service, publisher, release

    @staticmethod
    async def _publish(service: PublishSkillService, *, author: str) -> asyncio.Task[None]:
        before = set(PublishSkillService._pending_tasks.values())
        await service.execute(user_id=author, plugin_name="test-plugin", skill_name="test-skill", published=True)
        (task,) = set(PublishSkillService._pending_tasks.values()) - before
        return task

    @pytest.fixture(autouse=True)
    def _isolate_pending_tasks(self) -> None:
        PublishSkillService._pending_tasks.clear()

    @pytest.mark.parametrize(
        ("first_author", "second_author", "first_is_cancelled"),
        [
            ("author-a", "author-b", False),
            ("author-a", "author-a", True),
        ],
    )
    @pytest.mark.asyncio
    async def test_only_same_author_repeat_publish_cancels_earlier_task(
        self, first_author: str, second_author: str, first_is_cancelled: bool
    ) -> None:
        service, _publisher, release = self._build_service()

        first = await self._publish(service, author=first_author)
        await asyncio.sleep(0)  # let the first publish reach the (blocked) marketplace call
        second = await self._publish(service, author=second_author)
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)

        assert first.cancelled() is first_is_cancelled
        assert not second.cancelled()
        assert second.done()

    @pytest.mark.asyncio
    async def test_two_authors_publishing_same_skill_both_reach_marketplace(self) -> None:
        service, publisher, release = self._build_service()

        task_a = await self._publish(service, author="author-a")
        task_b = await self._publish(service, author="author-b")
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task_a, task_b)

        assert publisher.publish_skill.await_count == 2
        assert {call.kwargs["user_id"] for call in publisher.publish_skill.await_args_list} == {"author-a", "author-b"}

    @pytest.mark.asyncio
    async def test_cancelling_in_flight_publish_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        service, _publisher, release = self._build_service()

        first = await self._publish(service, author="author-a")
        with caplog.at_level(logging.WARNING):
            second = await self._publish(service, author="author-a")
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)

        assert any(r.levelno == logging.WARNING and "test-plugin/test-skill" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        ("author", "readable_prefix"),
        [
            ("author-a", "author-a-"),
            ("a.b+c@example.org", "a.b-c-example.org-"),
            ("@@", "author-"),
            ("a..b@x.com", "a-b-x.com-"),
            ("x.lock", "x.lock-"),
            (".hidden.", "hidden-"),
        ],
    )
    @pytest.mark.asyncio
    async def test_default_publish_branch_is_scoped_to_author(self, author: str, readable_prefix: str) -> None:
        service, publisher, release = self._build_service()
        publisher.use_branch = True

        task = await self._publish(service, author=author)
        await asyncio.sleep(0)
        release.set()
        await task

        branch = publisher.publish_skill.await_args.kwargs["branch_name"]
        assert branch.startswith(f"skill-publish/{readable_prefix}")
        assert branch.endswith("/test-plugin/test-skill")

    @pytest.mark.parametrize(
        "author_pair",
        [
            ("a+b@x.com", "a-b@x.com"),
            ("a+b@x.com", "a b@x.com"),
            ("@@", "!!"),
            ("a..b", "a.b"),
            ("alice", "Alice"),
        ],
    )
    def test_branch_segment_is_collision_free(self, author_pair: tuple[str, str]) -> None:
        first, second = (PublishSkillService._branch_segment(value=a) for a in author_pair)

        assert first != second

    @pytest.mark.parametrize(
        "author",
        ["a..b@x.com", "x.lock", "", "...", ".", "-", "a/b", "a b", "@{", "é☃", "x" * 500, "a.lock.", "~^:?*[\\"],
    )
    def test_branch_segment_is_valid_git_ref_component(self, author: str) -> None:
        segment = PublishSkillService._branch_segment(value=author)
        branch = f"skill-publish/{segment}/test-plugin/test-skill"

        assert segment == PublishSkillService._branch_segment(value=author)  # stable
        assert "/" not in segment
        assert len(segment) <= 60
        result = subprocess.run(
            ["git", "check-ref-format", f"refs/heads/{branch}"],
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, branch
