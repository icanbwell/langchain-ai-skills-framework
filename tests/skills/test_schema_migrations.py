"""Tests for schema_version migration of Mongo plugin-skill documents."""

from __future__ import annotations

import hashlib
import inspect
import logging
import time
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymongo.errors import DuplicateKeyError, OperationFailure

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.loaders.schema_migrations import (
    MIN_MIGRATABLE_VERSION,
    MissingMigrationError,
    content_digest_and_size,
    migrate_fields,
    registered_versions,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginSkillDocument

CURRENT = MongoPluginSkillDocument.SCHEMA_VERSION


def test_every_schema_version_hop_has_a_registered_migration() -> None:
    """Bumping SCHEMA_VERSION without registering its step would silently hide user-authored skills."""
    assert registered_versions() == frozenset(range(MIN_MIGRATABLE_VERSION, CURRENT))


def test_skill_migration_computes_manifest_fields_from_content() -> None:
    content = "# hello"
    doc = {"content": content, "schema_version": 3}

    fields = migrate_fields(kind="skills", doc=doc, from_version=3, to_version=4)

    assert fields == {
        "digest": f"sha256:{hashlib.sha256(content.encode()).hexdigest()}",
        "size": len(content.encode()),
        "is_dynamic": False,
    }


def test_multi_version_hop_is_chained() -> None:
    fields = migrate_fields(kind="resources", doc={"content": "x"}, from_version=2, to_version=4)
    assert fields["size"] == 1


def test_missing_hop_raises() -> None:
    with pytest.raises(MissingMigrationError):
        migrate_fields(kind="skills", doc={}, from_version=1, to_version=2)


def _matches(doc: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    for key, cond in query.items():
        value = doc.get(key)
        if isinstance(cond, dict):
            if "$gte" in cond and not (value is not None and value >= cond["$gte"]):
                return False
            if "$lt" in cond and not (value is not None and value < cond["$lt"]):
                return False
            if "$nin" in cond and value in cond["$nin"]:
                return False
        elif value != cond:
            return False
    return True


class _FakeCollection:
    def __init__(self, name: str, docs: list[dict[str, Any]] | None = None) -> None:
        self.name = name
        self.docs = docs or []
        self.raise_duplicate_on_update = False

    async def count_documents(self, query: Mapping[str, Any], limit: int = 0) -> int:
        return sum(1 for d in self.docs if _matches(d, query))

    def find(self, query: Mapping[str, Any]) -> Any:
        matched = [dict(d) for d in self.docs if _matches(d, query)]

        class _Cursor:
            def __init__(self, items: list[dict[str, Any]]) -> None:
                self._items = items

            def limit(self, n: int) -> _Cursor:
                return _Cursor(self._items[:n])

            def __aiter__(self) -> Any:
                async def gen() -> Any:
                    for d in self._items:
                        yield d

                return gen()

        return _Cursor(matched)

    async def find_one(self, query: Mapping[str, Any], projection: object = None) -> dict[str, Any] | None:
        return next((dict(d) for d in self.docs if _matches(d, query)), None)

    async def update_one(self, query: Mapping[str, Any], update: Mapping[str, Any]) -> MagicMock:
        if self.raise_duplicate_on_update:
            raise DuplicateKeyError("dup")
        for d in self.docs:
            if _matches(d, query):
                d.update(update["$set"])
                return MagicMock(modified_count=1)
        return MagicMock(modified_count=0)


class _FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def __getitem__(self, name: str) -> _FakeCollection:
        return self.collections.setdefault(name, _FakeCollection(name))


def _skill(version: int, **extra: Any) -> dict[str, Any]:
    return {
        "_id": f"{version}-{extra.get('skill_name', 's')}",
        "author": "u1",
        "plugin_name": "p",
        "skill_name": "s",
        "content": "body",
        "schema_version": version,
        **extra,
    }


async def test_stale_user_skill_is_migrated_with_manifest_fields() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(3))
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    counts = await loader.repair_stale_documents()

    assert counts["plugin_skills.migrated"] == 1
    migrated = db["plugin_skills"].docs[0]
    assert migrated["schema_version"] == CURRENT
    assert migrated["digest"].startswith("sha256:")
    assert migrated["size"] == 4


async def test_skill_two_versions_behind_is_migrated() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(2))
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    await loader.repair_stale_documents()

    assert db["plugin_skills"].docs[0]["schema_version"] == CURRENT


async def test_current_version_sibling_is_never_overwritten() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.extend([_skill(3), _skill(CURRENT, content="newer", _id="current")])
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    counts = await loader.repair_stale_documents()

    assert counts["plugin_skills.conflicts"] == 1
    assert counts["plugin_skills.migrated"] == 0
    stale, current = db["plugin_skills"].docs
    assert stale["schema_version"] == 3
    assert current["content"] == "newer"


async def test_concurrent_duplicate_key_is_counted_as_conflict() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(3))
    db["plugin_skills"].raise_duplicate_on_update = True
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    counts = await loader.repair_stale_documents()

    assert counts["plugin_skills.conflicts"] == 1


async def test_documents_without_schema_version_or_below_minimum_are_untouched() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.extend([_skill(1, _id="old"), {"_id": "none", "author": "u1", "skill_name": "x"}])
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    await loader.repair_stale_documents()

    assert db["plugin_skills"].docs[0]["schema_version"] == 1
    assert "schema_version" not in db["plugin_skills"].docs[1]


async def test_public_read_repairs_before_querying_and_rechecks_only_after_interval() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(3))
    loader = MongoPluginSkillLoader(database=db, repair_interval_seconds=3600)  # type: ignore[arg-type]

    assert await loader.skill_exists(author="u1", plugin_name="p", skill_name="s") is True

    # A straggler written by an old pod after the first check is not picked up until the interval elapses.
    db["plugin_skills"].docs.append(_skill(3, _id="late", skill_name="late"))
    assert await loader.skill_exists(author="u1", plugin_name="p", skill_name="late") is False
    loader._last_repair_check = (
        time.monotonic() - 3601
    )  # monotonic is boot-relative; never assume it exceeds the interval
    assert await loader.skill_exists(author="u1", plugin_name="p", skill_name="late") is True


async def test_auto_repair_can_be_disabled() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(3))
    loader = MongoPluginSkillLoader(database=db, auto_repair=False)  # type: ignore[arg-type]

    assert await loader.skill_exists(author="u1", plugin_name="p", skill_name="s") is False


async def test_repair_is_bounded_per_call_and_drains_across_calls() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.extend(_skill(3, _id=f"id{i}", skill_name=f"s{i}") for i in range(3))
    loader = MongoPluginSkillLoader(database=db, repair_batch_size=2)  # type: ignore[arg-type]

    await loader.skill_exists(author="u1", plugin_name="p", skill_name="s0")
    assert sum(d["schema_version"] == CURRENT for d in db["plugin_skills"].docs) == 2

    await loader.skill_exists(author="u1", plugin_name="p", skill_name="s0")
    assert all(d["schema_version"] == CURRENT for d in db["plugin_skills"].docs)


async def test_known_conflict_is_logged_once_and_does_not_consume_batch(caplog: pytest.LogCaptureFixture) -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.extend([_skill(3, _id="stale"), _skill(CURRENT, _id="current")])
    db["plugin_skills"].docs.append(_skill(3, _id="other", skill_name="other"))
    loader = MongoPluginSkillLoader(database=db, repair_batch_size=1)  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR):
        await loader.repair_stale_documents(batch_size=1)
        await loader.repair_stale_documents(batch_size=1)
        await loader.repair_stale_documents(batch_size=1)

    assert sum("Resolve manually" in r.message for r in caplog.records) == 1
    assert next(d for d in db["plugin_skills"].docs if d["_id"] == "other")["schema_version"] == CURRENT


def test_every_public_coroutine_repairs_first() -> None:
    """A public coroutine added without ``@_repairs_first`` would read past stale documents."""
    exempt = {"ensure_indexes", "repair_stale_documents"}
    undecorated = [
        name
        for name, member in vars(MongoPluginSkillLoader).items()
        if not name.startswith("_")
        and name not in exempt
        and inspect.iscoroutinefunction(member)
        and not getattr(member, "__repairs_first__", False)
    ]
    assert undecorated == []


def test_loader_and_migration_share_one_digest_definition() -> None:
    assert MongoPluginSkillLoader._digest_and_size("café") == content_digest_and_size("café")


async def test_repair_failure_is_logged_and_does_not_fail_the_read(caplog: pytest.LogCaptureFixture) -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(CURRENT, _id="current"))
    collection = db["plugin_skills"]
    real_count = collection.count_documents

    async def count_documents(query: Mapping[str, Any], limit: int = 0) -> int:
        if not isinstance(query.get("schema_version"), dict):
            return await real_count(query, limit=limit)
        raise RuntimeError("boom")  # the stale-document probe, not the read

    collection.count_documents = count_documents  # type: ignore[method-assign]
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR):
        assert await loader.skill_exists(author="u1", plugin_name="p", skill_name="s") is True

    assert any("Schema-version repair failed" in r.message for r in caplog.records)


async def test_unauthorized_repair_disables_itself_after_one_attempt(caplog: pytest.LogCaptureFixture) -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(3))
    db["plugin_skills"].update_one = AsyncMock(side_effect=OperationFailure("not authorized", code=13))  # type: ignore[method-assign]
    loader = MongoPluginSkillLoader(database=db, repair_interval_seconds=0)  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING):
        await loader.skill_exists(author="u1", plugin_name="p", skill_name="s")
        await loader.skill_exists(author="u1", plugin_name="p", skill_name="s")

    assert db["plugin_skills"].update_one.await_count == 1
    assert sum("repair disabled" in r.message for r in caplog.records) == 1


async def test_other_operation_failure_is_retried_next_interval() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(3))
    db["plugin_skills"].update_one = AsyncMock(side_effect=OperationFailure("transient", code=91))  # type: ignore[method-assign]
    loader = MongoPluginSkillLoader(database=db, repair_interval_seconds=0)  # type: ignore[arg-type]

    await loader.skill_exists(author="u1", plugin_name="p", skill_name="s")
    await loader.skill_exists(author="u1", plugin_name="p", skill_name="s")

    assert db["plugin_skills"].update_one.await_count == 2


async def test_unmigratable_document_is_logged_once_and_left_in_place(caplog: pytest.LogCaptureFixture) -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.append(_skill(CURRENT, _id="no-step"))
    # A loader targeting CURRENT + 1 has no registered step for CURRENT -> CURRENT + 1.
    loader = MongoPluginSkillLoader(database=db, schema_version=CURRENT + 1)  # type: ignore[arg-type]

    with caplog.at_level(logging.ERROR):
        first = await loader.repair_stale_documents()
        second = await loader.repair_stale_documents()

    assert first["plugin_skills.migrated"] == 0
    assert second["plugin_skills.migrated"] == 0
    assert db["plugin_skills"].docs[0]["schema_version"] == CURRENT
    assert sum("No migration registered" in r.message for r in caplog.records) == 1


async def test_non_integer_schema_version_is_skipped_without_starving_the_batch() -> None:
    db = _FakeDatabase()
    db["plugin_skills"].docs.extend([_skill(3, _id="float", skill_name="f"), _skill(3, _id="ok", skill_name="ok")])
    db["plugin_skills"].docs[0]["schema_version"] = 3.5
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]

    await loader.repair_stale_documents(batch_size=1)
    await loader.repair_stale_documents(batch_size=1)

    assert db["plugin_skills"].docs[0]["schema_version"] == 3.5
    assert db["plugin_skills"].docs[1]["schema_version"] == CURRENT


async def test_ensure_indexes_creates_schema_version_index_on_usage() -> None:
    db = _FakeDatabase()
    loader = MongoPluginSkillLoader(database=db)  # type: ignore[arg-type]
    loader._ensure_index = AsyncMock()  # type: ignore[method-assign]

    await loader.ensure_indexes()

    usage_calls = [c for c in loader._ensure_index.await_args_list if c.args[0] is db["plugin_skill_usage"]]
    assert {c.kwargs["name"] for c in usage_calls} >= {MongoPluginSkillLoader.USAGE_SCHEMA_VERSION_INDEX_NAME}
