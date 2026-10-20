"""Tests for script history snapshots that keep their content in the external ScriptContentStore (BAI-1200)."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from key_value.aio.stores.memory import MemoryStore

from langchain_ai_skills_framework.loaders.history_tracking_plugin_skill_store import HistoryTrackingPluginSkillStore
from langchain_ai_skills_framework.models.history_record import HistoryRecord
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginScriptDocument
from langchain_ai_skills_framework.persistence.script_content_store import (
    DEFAULT_SCRIPT_CONTENT_COLLECTION,
    KeyValueScriptContentStore,
    build_script_history_content_ref,
)
from tests.skills.test_script_content_store import KEY, _Env

# Larger than MongoDB's 16 MiB document limit once it sits inline in a history record.
OVER_MONGO_LIMIT = "x" * (17 * 1024 * 1024)
MAX_BSON_BYTES = 16 * 1024 * 1024


class _FakeHistoryWriter:
    """In-memory stand-in for ``HistoryWriter`` that, like Mongo, refuses documents over the 16 MiB limit."""

    def __init__(self) -> None:
        self.script_records: list[HistoryRecord] = []
        self.rejected: list[HistoryRecord] = []
        self.fail_writes = False

    async def ensure_indexes(self) -> None:
        return None

    async def write_script_history(self, *, record: HistoryRecord) -> bool:
        too_big = len(json.dumps(record.to_mongo_dict(), default=str)) > MAX_BSON_BYTES
        if too_big or self.fail_writes:
            self.rejected.append(record)  # HistoryWriter logs a warning and swallows the error here
            return False
        self.script_records.append(record)
        return True

    async def get_script_history(self, *, limit: int = 50, offset: int = 0, **_: Any) -> list[HistoryRecord]:
        return list(reversed(self.script_records))[offset : offset + limit]


def _script_doc(content: str) -> MongoPluginScriptDocument:
    return MongoPluginScriptDocument(
        plugin_name="my-plugin",
        skill_name="my-skill",
        script_name="run.py",
        path="p/run.py",
        content=content,
        digest="sha256:abc",
        author="user-1",
    )


def _inner(*, content: str = "print(1)", exists: bool = False) -> AsyncMock:
    inner = AsyncMock()
    inner.script_exists = AsyncMock(return_value=exists)
    inner.save_script = AsyncMock(return_value=_script_doc(content))
    inner.read_script = AsyncMock(return_value=content)
    inner.delete_script = AsyncMock(return_value=True)
    return inner


def _tracking(
    *, inner: Any, writer: _FakeHistoryWriter, store: KeyValueScriptContentStore | None, enabled: bool = True
) -> HistoryTrackingPluginSkillStore:
    return HistoryTrackingPluginSkillStore(
        inner_store=inner,
        history_writer=writer,  # type: ignore[arg-type]
        script_content_store=store,
        write_script_history_to_content_store=enabled,
    )


def _stored_refs(kv: MemoryStore) -> list[str]:
    # MemoryStore has no public key listing; peek at the collection the way test_script_content_store does.
    collection = kv._cache.get(DEFAULT_SCRIPT_CONTENT_COLLECTION)
    return sorted(collection.keys()) if collection is not None else []


async def _save(tracking: HistoryTrackingPluginSkillStore, content: str = "print(1)") -> Any:
    return await tracking.save_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py", content=content
    )


async def _history(tracking: HistoryTrackingPluginSkillStore) -> list[HistoryRecord]:
    return list(
        await tracking.get_script_history(
            user_id="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
        )
    )


@pytest.fixture
def kv() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def store(kv: MemoryStore) -> KeyValueScriptContentStore:
    return KeyValueScriptContentStore(key_value=kv)


@pytest.fixture
def writer() -> _FakeHistoryWriter:
    return _FakeHistoryWriter()


def test_history_ref_is_distinct_from_live_refs_and_unique() -> None:
    first = build_script_history_content_ref(script_key=KEY, digest="sha256:ab")
    second = build_script_history_content_ref(script_key=KEY, digest="sha256:ab")
    assert first.startswith(f"{KEY}@sha256:ab#history-")
    assert first != second


# --- flag off / no store: unchanged ---------------------------------------------


@pytest.mark.parametrize("variant", ["flag_off", "no_store"])
async def test_flag_off_or_no_store_records_inline_content_exactly_as_before(
    variant: str, store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    tracking = _tracking(
        inner=_inner(),
        writer=writer,
        store=store if variant == "flag_off" else None,
        enabled=variant != "flag_off",
    )
    doc = await _save(tracking)

    assert writer.script_records[0].document_snapshot == doc.to_mongo_dict()
    assert "content_ref" not in writer.script_records[0].document_snapshot
    assert (await _history(tracking))[0].document_snapshot["content"] == "print(1)"


# --- flag on ------------------------------------------------------------------


async def test_save_records_a_ref_not_inline_content_and_read_resolves_it(
    kv: MemoryStore, store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    tracking = _tracking(inner=_inner(), writer=writer, store=store)
    doc = await _save(tracking)

    assert doc.content == "print(1)"  # the caller still gets the full document back
    stored = writer.script_records[0].document_snapshot
    assert "content" not in stored
    assert stored["content_ref"].startswith(f"{KEY}@sha256:abc#history-")
    assert await store.get(ref=stored["content_ref"]) == "print(1)"

    resolved = (await _history(tracking))[0].document_snapshot
    assert resolved["content"] == "print(1)"
    assert "content_ref" not in resolved  # same shape callers always saw
    assert {k: v for k, v in resolved.items() if k != "content"} == {
        k: v for k, v in stored.items() if k != "content_ref"
    }
    assert "content" not in writer.script_records[0].document_snapshot  # reading did not mutate the stored record


async def test_delete_script_snapshot_goes_through_the_store(
    store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    tracking = _tracking(inner=_inner(content="doomed"), writer=writer, store=store)
    assert await tracking.delete_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )

    stored = writer.script_records[0].document_snapshot
    assert writer.script_records[0].action == "deleted"
    assert "content" not in stored and stored["script_name"] == "run.py"
    assert (await _history(tracking))[0].document_snapshot["content"] == "doomed"


async def test_delete_script_that_deletes_nothing_writes_no_content_copy(
    kv: MemoryStore, store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    inner = _inner()
    inner.delete_script.return_value = False
    await _tracking(inner=inner, writer=writer, store=store).delete_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )
    assert writer.script_records == []
    assert _stored_refs(kv) == []


async def test_delete_script_with_unreadable_content_keeps_the_empty_snapshot(
    store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    inner = _inner()
    inner.read_script.side_effect = RuntimeError("gone")
    await _tracking(inner=inner, writer=writer, store=store).delete_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )
    assert writer.script_records[0].document_snapshot == {}


# --- retention against the real loader --------------------------------------------


async def test_history_content_survives_the_live_ref_being_retired(mock_mongo_database: MagicMock) -> None:
    env = _Env(mock_mongo_database)
    writer = _FakeHistoryWriter()
    tracking = _tracking(inner=env.loader, writer=writer, store=env.store)

    await _save(tracking, "v1")
    first_live_ref = env.scripts.docs["run.py"]["content_ref"]
    await _save(tracking, "v2")  # retires (deletes) the v1 live ref
    assert await env.store.get(ref=first_live_ref) is None

    history = await _history(tracking)
    assert [r.document_snapshot["content"] for r in history] == ["v2", "v1"]
    assert {r.document_snapshot.get("content_ref") for r in history} == {None}

    assert await tracking.delete_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )  # retires the v2 live ref too
    contents = [r.document_snapshot["content"] for r in await _history(tracking)]
    assert contents == ["v2", "v2", "v1"]  # the delete record holds its own copy of v2
    assert len(await env.stored_refs()) == 3  # exactly the three history copies remain


# --- old records and failures ---------------------------------------------------


async def test_old_inline_history_records_are_still_readable_with_the_flag_on(
    store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    writer.script_records.append(
        HistoryRecord(
            action="created",
            document_snapshot={"script_name": "run.py", "content": "legacy inline"},
            source_collection="plugin_scripts",
            user_id="user-1",
            plugin_name="my-plugin",
            skill_name="my-skill",
            script_name="run.py",
        )
    )
    tracking = _tracking(inner=_inner(content="new"), writer=writer, store=store)
    await _save(tracking, "new")

    assert [r.document_snapshot["content"] for r in await _history(tracking)] == ["new", "legacy inline"]


async def test_ref_record_with_missing_content_is_returned_without_content_and_warns(
    store: KeyValueScriptContentStore, writer: _FakeHistoryWriter, caplog: pytest.LogCaptureFixture
) -> None:
    tracking = _tracking(inner=_inner(), writer=writer, store=store)
    await _save(tracking)
    ref = writer.script_records[0].document_snapshot["content_ref"]
    await store.delete(ref=ref)

    with caplog.at_level(logging.WARNING):
        snapshot = (await _history(tracking))[0].document_snapshot

    assert "content" not in snapshot  # never an empty string standing in for lost content
    assert snapshot["content_ref"] == ref
    assert ref in caplog.text


async def test_ref_record_is_not_resolved_into_an_error_when_the_store_raises(
    writer: _FakeHistoryWriter, caplog: pytest.LogCaptureFixture
) -> None:
    failing = MagicMock()
    failing.get = AsyncMock(side_effect=RuntimeError("store down"))
    tracking = _tracking(inner=_inner(), writer=writer, store=failing)
    writer.script_records.append(
        HistoryRecord(
            action="updated",
            document_snapshot={"content_ref": "some-ref"},
            source_collection="plugin_scripts",
            user_id="user-1",
            plugin_name="my-plugin",
            script_name="run.py",
        )
    )
    with caplog.at_level(logging.WARNING):
        assert (await _history(tracking))[0].document_snapshot == {"content_ref": "some-ref"}
    assert "some-ref" in caplog.text


async def test_store_put_failure_falls_back_to_the_inline_record_and_does_not_fail_the_save(
    writer: _FakeHistoryWriter, caplog: pytest.LogCaptureFixture
) -> None:
    failing = MagicMock()
    failing.put = AsyncMock(side_effect=RuntimeError("store down"))
    tracking = _tracking(inner=_inner(), writer=writer, store=failing)

    with caplog.at_level(logging.WARNING):
        doc = await _save(tracking)

    assert doc.content == "print(1)"
    assert writer.script_records[0].document_snapshot["content"] == "print(1)"
    assert "Could not store script history content" in caplog.text


async def test_content_copy_is_removed_when_the_history_record_is_not_stored(
    kv: MemoryStore, store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    writer.fail_writes = True
    await _save(_tracking(inner=_inner(), writer=writer, store=store))
    assert _stored_refs(kv) == []


# --- the 16 MiB case --------------------------------------------------------------


async def test_script_over_the_inline_limit_still_gets_a_small_history_record_when_the_flag_is_on(
    store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    tracking = _tracking(inner=_inner(content=OVER_MONGO_LIMIT), writer=writer, store=store)
    await _save(tracking, OVER_MONGO_LIMIT)

    assert writer.rejected == []
    record = writer.script_records[0]
    assert "content" not in record.document_snapshot
    assert len(json.dumps(record.to_mongo_dict(), default=str)) < 4096  # history document stays tiny
    assert (await _history(tracking))[0].document_snapshot["content"] == OVER_MONGO_LIMIT


async def test_script_over_the_inline_limit_loses_its_history_record_when_the_flag_is_off(
    store: KeyValueScriptContentStore, writer: _FakeHistoryWriter
) -> None:
    """Documents the pre-existing behaviour the flag removes: the oversized inline record is rejected."""
    tracking = _tracking(inner=_inner(content=OVER_MONGO_LIMIT), writer=writer, store=store, enabled=False)
    await _save(tracking, OVER_MONGO_LIMIT)
    assert writer.script_records == [] and len(writer.rejected) == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_factory_wires_history_to_the_store_with_the_same_flag(mock_mongo_database: MagicMock, enabled: bool) -> None:
    from langchain_ai_skills_framework.loaders.plugin_skill_store_factory import PluginSkillStoreFactory

    env_vars = MagicMock()
    env_vars.enable_skill_history = True
    env_vars.plugin_scripts_external_content_enabled = enabled
    env_vars.plugin_skills_collection = env_vars.plugin_references_collection = None
    env_vars.plugin_scripts_collection = env_vars.plugins_collection = None
    database_factory = MagicMock()
    database_factory.create_database.return_value = mock_mongo_database
    store = KeyValueScriptContentStore(key_value=MemoryStore())

    created = PluginSkillStoreFactory(
        mongo_database_factory=database_factory, environment_variables=env_vars, script_content_store=store
    ).create()

    assert isinstance(created, HistoryTrackingPluginSkillStore)
    assert created._script_content_store is store
    assert created._write_script_history_to_content_store is enabled
