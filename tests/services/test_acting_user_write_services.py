"""Write services record the acting user separately from the author (BAI-1500)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.services.delete_skill_resource_service import DeleteSkillResourceService
from langchain_ai_skills_framework.services.delete_skill_script_service import DeleteSkillScriptService
from langchain_ai_skills_framework.services.delete_skill_service import DeleteSkillService
from langchain_ai_skills_framework.services.publish_skill_service import PublishSkillService
from langchain_ai_skills_framework.services.save_skill_resource_service import SaveSkillResourceService
from langchain_ai_skills_framework.services.save_skill_script_service import SaveSkillScriptService
from langchain_ai_skills_framework.services.save_skill_service import SaveSkillService

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
CASES = [
    pytest.param(_save_skill, "save_skill", id="save_skill"),
    pytest.param(_save_resource, "save_resource", id="save_resource"),
    pytest.param(_save_script, "save_script", id="save_script"),
    pytest.param(_delete_skill, "delete_skill", id="delete_skill"),
    pytest.param(_delete_resource, "delete_resource", id="delete_resource"),
    pytest.param(_delete_script, "delete_script", id="delete_script"),
    pytest.param(_publish, "set_skill_state", id="publish"),
    pytest.param(_unpublish, "set_skill_state", id="unpublish"),
]


@pytest.mark.parametrize(("runner", "method"), CASES)
async def test_modified_by_is_recorded_separately_from_author(runner: Any, method: str) -> None:
    store = _store()

    await runner(store, user_id=ADMIN, author=OWNER, modified_by=ADMIN)

    kwargs = getattr(store, method).call_args.kwargs
    assert kwargs["author"] == OWNER
    assert kwargs["modified_by"] == ADMIN


@pytest.mark.parametrize(("runner", "method"), CASES)
async def test_author_and_modified_by_default_to_user_id(runner: Any, method: str) -> None:
    store = _store()

    await runner(store, user_id=OWNER)

    kwargs = getattr(store, method).call_args.kwargs
    assert kwargs["author"] == OWNER
    assert kwargs["modified_by"] == OWNER


async def test_publish_gate_checks_the_authors_skill_not_the_actors() -> None:
    store = _store()

    await _publish(store, user_id=ADMIN, author=OWNER, modified_by=ADMIN)

    assert store.skill_exists.call_args.kwargs["author"] == OWNER
    assert store.get_skill_details.call_args.kwargs["author"] == OWNER
