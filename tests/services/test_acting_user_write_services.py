"""Write services record the acting user separately from the author (BAI-1500)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.publishing.github_marketplace_publisher import GitHubMarketplacePublisher
from langchain_ai_skills_framework.services.delete_skill_resource_service import DeleteSkillResourceService
from langchain_ai_skills_framework.services.delete_skill_script_service import DeleteSkillScriptService
from langchain_ai_skills_framework.services.delete_skill_service import DeleteSkillService
from langchain_ai_skills_framework.services.publish_skill_service import PublishSkillService
from langchain_ai_skills_framework.services.save_skill_resource_service import SaveSkillResourceService
from langchain_ai_skills_framework.services.save_skill_script_service import SaveSkillScriptService
from langchain_ai_skills_framework.services.save_skill_service import SaveSkillService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError

CONTENT = "---\nname: my-skill\ndescription: A test skill\n---\n# Body"
ADMIN = "admin@example.com"
OWNER = "author-a@example.com"
_NOW = datetime.now(UTC)


def _skill_doc(*, state: str = "in_review") -> MongoPluginSkillDocument:
    return MongoPluginSkillDocument(
        plugin_name="p",
        author=OWNER,
        skill_name="my-skill",
        path="p/skills/my-skill/SKILL.md",
        description="d",
        content=CONTENT,
        modified_by=OWNER,
        state=state,
        date_created=_NOW,
        date_modified=_NOW,
    )


def _store() -> AsyncMock:
    store = AsyncMock(spec=PluginSkillStore)
    store.skill_exists.return_value = True
    store.save_skill.return_value = _skill_doc()
    store.save_resource.return_value = MongoPluginResourceDocument(
        plugin_name="p", skill_name="my-skill", resource_name="r.md", path="x", content="c", author=OWNER
    )
    store.save_script.return_value = MongoPluginScriptDocument(
        plugin_name="p", skill_name="my-skill", script_name="s.py", path="x", content="c", author=OWNER
    )
    store.delete_skill.return_value = True
    store.delete_resource.return_value = True
    store.delete_script.return_value = True
    store.set_skill_state.return_value = _skill_doc()
    store.get_skill_details.return_value = AsyncMock(summary=AsyncMock(state="staging"))
    return store


async def _save_skill(store: AsyncMock, **kw: Any) -> None:
    await SaveSkillService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", content=CONTENT, **kw
    )


async def _save_resource(store: AsyncMock, **kw: Any) -> None:
    await SaveSkillResourceService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", resource_name="r.md", content="c", **kw
    )


async def _save_script(store: AsyncMock, **kw: Any) -> None:
    await SaveSkillScriptService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", script_name="s.py", content="c", **kw
    )


async def _delete_skill(store: AsyncMock, **kw: Any) -> None:
    await DeleteSkillService(mongo_skill_loader=store).execute(plugin_name="p", skill_name="my-skill", **kw)


async def _delete_resource(store: AsyncMock, **kw: Any) -> None:
    await DeleteSkillResourceService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", resource_name="r.md", **kw
    )


async def _delete_script(store: AsyncMock, **kw: Any) -> None:
    await DeleteSkillScriptService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", script_name="s.py", **kw
    )


async def _publish(store: AsyncMock, **kw: Any) -> None:
    await PublishSkillService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", published=True, **kw
    )


async def _unpublish(store: AsyncMock, **kw: Any) -> None:
    await PublishSkillService(mongo_skill_loader=store).execute(
        plugin_name="p", skill_name="my-skill", published=False, **kw
    )


# (runner, store method that receives author=/modified_by=)
CASES = (
    pytest.param(_save_skill, "save_skill", id="save_skill"),
    pytest.param(_save_resource, "save_resource", id="save_resource"),
    pytest.param(_save_script, "save_script", id="save_script"),
    pytest.param(_delete_skill, "delete_skill", id="delete_skill"),
    pytest.param(_delete_resource, "delete_resource", id="delete_resource"),
    pytest.param(_delete_script, "delete_script", id="delete_script"),
    pytest.param(_publish, "set_skill_state", id="publish"),
    pytest.param(_unpublish, "set_skill_state", id="unpublish"),
)


@pytest.mark.parametrize(("runner", "method"), CASES)
async def test_modified_by_is_recorded_separately_from_author(runner: Any, method: str) -> None:
    store = _store()

    await runner(store, user_id=ADMIN, author=OWNER, modified_by=ADMIN)

    kwargs = getattr(store, method).call_args.kwargs
    assert kwargs["author"] == OWNER
    assert kwargs["modified_by"] == ADMIN


# Methods whose call shape must stay exactly as before BAI-1500 when the owner acts for themself.
LEGACY_SHAPE_CASES = tuple(
    c for c in CASES if c.values[1] in {"delete_skill", "delete_resource", "delete_script", "set_skill_state"}
)
SAVE_CASES = tuple(c for c in CASES if c not in LEGACY_SHAPE_CASES)


@pytest.mark.parametrize(("runner", "method"), SAVE_CASES)
async def test_save_defaults_author_and_modified_by_to_user_id(runner: Any, method: str) -> None:
    store = _store()

    await runner(store, user_id=OWNER)

    kwargs = getattr(store, method).call_args.kwargs
    assert kwargs["author"] == OWNER
    assert kwargs["modified_by"] == OWNER


ABSENT = object()


@pytest.mark.parametrize(("runner", "method"), LEGACY_SHAPE_CASES)
@pytest.mark.parametrize(
    ("user_id", "author", "modified_by", "expected_author", "expected_modified_by"),
    [
        pytest.param(OWNER, None, None, OWNER, ABSENT, id="owner-defaults"),
        pytest.param(OWNER, OWNER, None, OWNER, ABSENT, id="owner-explicit-author"),
        pytest.param(OWNER, OWNER, OWNER, OWNER, ABSENT, id="owner-all-same"),
        pytest.param(ADMIN, OWNER, OWNER, OWNER, ABSENT, id="admin-token-actor-is-owner"),
        pytest.param(ADMIN, OWNER, None, OWNER, ADMIN, id="admin-for-owner-actor-defaults-to-user"),
        pytest.param(ADMIN, OWNER, ADMIN, OWNER, ADMIN, id="admin-for-owner-explicit"),
        pytest.param(ADMIN, ADMIN, OWNER, ADMIN, OWNER, id="actor-differs-from-author"),
        pytest.param(ADMIN, None, None, ADMIN, ABSENT, id="admin-on-own-skill"),
    ],
)
async def test_owner_actor_matrix_for_store_calls(
    runner: Any,
    method: str,
    user_id: str,
    author: str | None,
    modified_by: str | None,
    expected_author: str,
    expected_modified_by: object,
) -> None:
    """``modified_by`` is passed only when the actor differs from the owner; otherwise the pre-BAI-1500 shape."""
    store = _store()
    extra = {k: v for k, v in (("author", author), ("modified_by", modified_by)) if v is not None}

    await runner(store, user_id=user_id, **extra)

    kwargs = getattr(store, method).call_args.kwargs
    assert kwargs["author"] == expected_author
    if expected_modified_by is ABSENT:
        assert "modified_by" not in kwargs
    else:
        assert kwargs["modified_by"] == expected_modified_by


class _LegacyStore:
    """Store written against the pre-BAI-1500 protocol: no ``modified_by`` on delete or set_skill_state."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def skill_exists(self, *, author: str, plugin_name: str, skill_name: str) -> bool:
        return True

    async def get_skill_details(self, *, author: str, plugin_name: str | None = None, skill_name: str) -> Any:
        return SimpleNamespace(summary=SimpleNamespace(state="staging"))

    async def set_skill_state(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        state: str,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument:
        self.calls.append("set_skill_state")
        return _skill_doc()

    async def delete_skill(self, *, author: str, plugin_name: str, skill_name: str) -> bool:
        self.calls.append("delete_skill")
        return True

    async def delete_resource(self, *, author: str, plugin_name: str, skill_name: str, resource_name: str) -> bool:
        self.calls.append("delete_resource")
        return True

    async def delete_script(self, *, author: str, plugin_name: str, skill_name: str, script_name: str) -> bool:
        self.calls.append("delete_script")
        return True


class _LegacyPublisher:
    """Publisher written against the pre-BAI-1500 signatures: no ``acting_user_id``."""

    use_branch = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def publish_skill(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        skill_content: str,
        resources: dict[str, str],
        scripts: dict[str, str],
        user_id: str,
        branch_name: str | None = None,
        skill_path: str | None = None,
        resource_paths: dict[str, str] | None = None,
        script_paths: dict[str, str] | None = None,
    ) -> str:
        self.calls.append("publish_skill")
        return "sha"

    async def unpublish_skill(
        self, *, plugin_name: str, skill_name: str, user_id: str, skill_dir: str | None = None
    ) -> str | None:
        self.calls.append("unpublish_skill")
        return "sha"


@pytest.mark.parametrize(
    ("runner", "method"),
    tuple(c for c in LEGACY_SHAPE_CASES),
)
async def test_legacy_store_signatures_still_work_for_owner_calls(runner: Any, method: str) -> None:
    store = _LegacyStore()

    await runner(cast(PluginSkillStore, store), user_id=OWNER)

    assert store.calls == [method]


@pytest.mark.parametrize(("published", "method"), [(True, "publish_skill"), (False, "unpublish_skill")])
async def test_legacy_publisher_signatures_still_work_for_owner_calls(published: bool, method: str) -> None:
    publisher = _LegacyPublisher()
    service = PublishSkillService(
        mongo_skill_loader=_publish_store(), marketplace_publisher=cast(GitHubMarketplacePublisher, publisher)
    )

    await service.execute(user_id=OWNER, plugin_name="p", skill_name="my-skill", published=published)
    await _drain_pending()

    assert publisher.calls == [method]


@pytest.mark.parametrize(("runner", "method"), CASES)
@pytest.mark.parametrize("field", ["author", "modified_by"])
@pytest.mark.parametrize("blank", ["", "   "])
async def test_explicitly_blank_author_or_modified_by_is_rejected(
    runner: Any, method: str, field: str, blank: str
) -> None:
    store = _store()

    with pytest.raises(SkillOperationError, match=field):
        await runner(store, user_id=OWNER, **{field: blank})

    getattr(store, method).assert_not_called()


async def test_publish_gate_checks_the_authors_skill_not_the_actors() -> None:
    store = _store()

    await _publish(store, user_id=ADMIN, author=OWNER, modified_by=ADMIN)

    assert store.skill_exists.call_args.kwargs["author"] == OWNER
    assert store.get_skill_details.call_args.kwargs["author"] == OWNER


def _publisher() -> MagicMock:
    publisher = MagicMock()
    publisher.use_branch = False
    publisher.publish_skill = AsyncMock(return_value="sha")
    publisher.unpublish_skill = AsyncMock(return_value="sha")
    return publisher


@pytest.fixture(autouse=True)
def _clear_pending_publish_tasks() -> Iterator[None]:
    """``_pending_tasks`` is class-level state; keep it from leaking between tests."""
    PublishSkillService._pending_tasks.clear()
    yield
    PublishSkillService._pending_tasks.clear()


async def _drain_pending() -> None:
    await asyncio.gather(*PublishSkillService._pending_tasks.values())


def _publish_store() -> AsyncMock:
    store = _store()
    store.set_skill_state.return_value = _skill_doc()
    store.get_skill_details.return_value = AsyncMock(
        summary=AsyncMock(state="staging", folder=None, path="p/skills/my-skill/SKILL.md"),
        content=CONTENT,
    )
    store.list_resource_documents.return_value = []
    store.list_script_documents.return_value = []
    return store


@pytest.mark.parametrize(("published", "method"), [(True, "publish_skill"), (False, "unpublish_skill")])
@pytest.mark.parametrize(
    ("user_id", "author", "expected_acting"),
    [(ADMIN, OWNER, ADMIN), (OWNER, None, None), (OWNER, OWNER, None)],
)
async def test_background_publish_carries_owner_and_actor(
    published: bool, method: str, user_id: str, author: str | None, expected_acting: str | None
) -> None:
    publisher = _publisher()
    service = PublishSkillService(mongo_skill_loader=_publish_store(), marketplace_publisher=publisher)

    await service.execute(user_id=user_id, author=author, plugin_name="p", skill_name="my-skill", published=published)
    await _drain_pending()

    kwargs = getattr(publisher, method).call_args.kwargs
    assert kwargs["user_id"] == OWNER
    if expected_acting is None:
        assert "acting_user_id" not in kwargs
    else:
        assert kwargs["acting_user_id"] == expected_acting
