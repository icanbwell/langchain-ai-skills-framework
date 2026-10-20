"""Tests for external script content: the key-value adapter, the loader's use of it, and the injection wiring."""

from __future__ import annotations

import hashlib
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from key_value.aio.stores.memory import MemoryStore
from pymongo import ReturnDocument
from simple_container.container.simple_container import SimpleContainer

from langchain_ai_skills_framework.container.container_factory import _resolve_script_content_store
from langchain_ai_skills_framework.environment.environment_variables import (
    LangchainAISkillsFrameworkEnvironmentVariables,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_loader_error import SkillLoaderError
from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.loaders.plugin_skill_store_factory import PluginSkillStoreFactory
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginSkillDocument
from langchain_ai_skills_framework.persistence.script_content_store import (
    DEFAULT_SCRIPT_CONTENT_COLLECTION,
    KeyValueScriptContentStore,
    ScriptContentStore,
    build_script_content_key,
    build_script_content_ref,
)

KEY = "user-1/my-plugin/my-skill/run.py"
IDENTITY: dict[str, Any] = {"author": "user-1", "plugin_name": "my-plugin", "skill_name": "my-skill"}


def _digest_of(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


class _AsyncIter:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def limit(self, _n: int) -> _AsyncIter:
        return self


class _FakeScriptsCollection:
    """Stateful stand-in for ``plugin_scripts`` honouring the Mongo semantics the loader relies on."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.fail_next_upsert = False
        self.before_upsert: Any = None  # async hook run just before an upsert is applied
        self.upserts = 0

    @staticmethod
    def _matches(doc: dict[str, Any], filter_: dict[str, Any]) -> bool:
        for field, wanted in filter_.items():
            if field == "schema_version":
                continue  # seeded legacy documents carry no version; every document here is current
            if isinstance(wanted, dict) and "$in" in wanted:
                if doc.get(field) not in wanted["$in"]:
                    return False
            elif doc.get(field) != wanted:
                return False
        return True

    def _match(self, filter_: dict[str, Any]) -> dict[str, Any] | None:
        return next((doc for doc in self.docs.values() if self._matches(doc, filter_)), None)

    async def find_one_and_update(
        self, filter_: dict[str, Any], update: dict[str, dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any] | None:
        if self.before_upsert is not None:
            hook, self.before_upsert = self.before_upsert, None
            await hook()
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("upsert failed")
        self.upserts += 1
        doc = self._match(filter_)
        before = dict(doc) if doc is not None else None
        if doc is None:
            doc = {"_id": len(self.docs) + 1, **update.get("$setOnInsert", {})}
            self.docs[doc["script_name"]] = doc
        doc.update(update.get("$set", {}))
        for field in update.get("$unset", {}):
            doc.pop(field, None)
        if kwargs.get("return_document") == ReturnDocument.BEFORE:
            return before
        return dict(doc)

    async def find_one(self, filter_: dict[str, Any], projection: Any = None) -> dict[str, Any] | None:
        doc = self._match(filter_)
        return dict(doc) if doc is not None else None

    async def find_one_and_delete(self, filter_: dict[str, Any]) -> dict[str, Any] | None:
        doc = self._match(filter_)
        if doc is None:
            return None
        del self.docs[doc["script_name"]]
        return dict(doc)

    async def delete_one(self, filter_: dict[str, Any]) -> Any:
        return MagicMock(deleted_count=1 if await self.find_one_and_delete(filter_) is not None else 0)

    async def delete_many(self, filter_: dict[str, Any]) -> Any:
        count = 0
        while await self.find_one_and_delete(filter_) is not None:
            count += 1
        return MagicMock(deleted_count=count)

    def find(self, filter_: dict[str, Any]) -> _AsyncIter:
        return _AsyncIter([dict(d) for d in self.docs.values() if self._matches(d, filter_)])

    async def update_one(self, filter_: dict[str, Any], update: dict[str, dict[str, Any]]) -> Any:
        doc = self._match(filter_)
        if doc is None:
            return MagicMock(modified_count=0)
        doc.update(update.get("$set", {}))
        for field in update.get("$unset", {}):
            doc.pop(field, None)
        return MagicMock(modified_count=1)


class _Env:
    """A loader wired to a fake scripts collection and a real KeyValueScriptContentStore over a MemoryStore."""

    def __init__(self, db: MagicMock, *, write_to_store: bool = True, with_store: bool = True) -> None:
        self.kv = MemoryStore()
        self.store = KeyValueScriptContentStore(key_value=self.kv)
        self.scripts = _FakeScriptsCollection()
        self.db = db
        for name in (
            "find_one_and_update",
            "find_one",
            "find_one_and_delete",
            "delete_one",
            "delete_many",
            "update_one",
        ):
            setattr(db["plugin_scripts"], name, getattr(self.scripts, name))
        db["plugin_scripts"].find = MagicMock(side_effect=self.scripts.find)
        db["plugin_skills"].find_one = AsyncMock(return_value=None)
        db["plugin_skills"].delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
        self.loader = MongoPluginSkillLoader(
            database=db,
            script_content_store=self.store if with_store else None,
            write_scripts_to_content_store=write_to_store,
        )

    async def stored_refs(self) -> list[str]:
        # MemoryStore has no key listing in the public API; the collection's cache keys are what a test may peek at.
        collection = self.kv._cache.get(DEFAULT_SCRIPT_CONTENT_COLLECTION)
        return sorted(collection.keys()) if collection is not None else []

    async def save(
        self, content: str, *, script_name: str = "run.py", loader: MongoPluginSkillLoader | None = None
    ) -> Any:
        return await (loader or self.loader).save_script(
            **IDENTITY, script_name=script_name, content=content, path=f"p/{script_name}"
        )

    async def read(self, *, script_name: str = "run.py") -> str:
        return await self.loader.read_script(**IDENTITY, script_name=script_name)


@pytest.fixture
def env(mock_mongo_database: MagicMock) -> _Env:
    return _Env(mock_mongo_database)


# --- KeyValueScriptContentStore ------------------------------------------------


async def test_put_get_delete_roundtrip() -> None:
    store = KeyValueScriptContentStore(key_value=MemoryStore())
    assert await store.get(ref="r") is None
    await store.put(ref="r", content="print('héllo')")
    assert await store.get(ref="r") == "print('héllo')"
    await store.put(ref="r", content="replaced")
    assert await store.get(ref="r") == "replaced"
    assert await store.delete(ref="r") is True
    assert await store.get(ref="r") is None
    assert await store.delete(ref="r") is False


async def test_value_shape_and_collection_in_the_underlying_store() -> None:
    kv = MemoryStore()
    store = KeyValueScriptContentStore(key_value=kv, collection="my_collection")
    await store.put(ref="r", content="x")
    assert await kv.get("r", collection="my_collection") == {"content": "x"}
    assert await kv.get("r", collection=DEFAULT_SCRIPT_CONTENT_COLLECTION) is None
    assert await kv.ttl("r", collection="my_collection") == ({"content": "x"}, None)  # never expires


@pytest.mark.parametrize("value", [{"other": "x"}, {"content": 5}, {}])
async def test_get_treats_a_malformed_value_as_missing(value: dict[str, Any]) -> None:
    kv = MemoryStore()
    await kv.put("r", value, collection=DEFAULT_SCRIPT_CONTENT_COLLECTION)
    assert await KeyValueScriptContentStore(key_value=kv).get(ref="r") is None


def test_adapter_satisfies_protocol() -> None:
    assert isinstance(KeyValueScriptContentStore(key_value=MemoryStore()), ScriptContentStore)


def test_refs_carry_key_and_digest_and_are_unique_per_save() -> None:
    assert build_script_content_key(**IDENTITY, script_name="run.py") == KEY
    first = build_script_content_ref(script_key=KEY, digest="sha256:ab")
    second = build_script_content_ref(script_key=KEY, digest="sha256:ab")
    assert first.startswith(f"{KEY}@sha256:ab#") and second.startswith(f"{KEY}@sha256:ab#")
    assert first != second


# --- Writes --------------------------------------------------------------------


async def test_flag_on_stores_content_externally_and_unsets_inline(env: _Env) -> None:
    doc = await env.save("print(1)")
    stored = env.scripts.docs["run.py"]
    assert "content" not in stored
    assert stored["digest"] == _digest_of("print(1)")
    ref = stored["content_ref"]
    assert ref.startswith(f"{KEY}@{_digest_of('print(1)')}#")
    assert await env.store.get(ref=ref) == "print(1)"
    assert await env.stored_refs() == [ref]
    assert doc.content == "print(1)" and doc.digest == stored["digest"]  # the caller still gets the content back
    assert await env.read() == "print(1)"


async def test_flag_off_keeps_content_inline_and_stores_nothing(mock_mongo_database: MagicMock) -> None:
    env = _Env(mock_mongo_database, write_to_store=False)
    await env.save("print(1)")
    stored = env.scripts.docs["run.py"]
    assert stored["content"] == "print(1)" and "content_ref" not in stored
    assert await env.stored_refs() == []
    assert await env.read() == "print(1)"


async def test_no_store_behaves_exactly_as_before(mock_mongo_database: MagicMock) -> None:
    env = _Env(mock_mongo_database, with_store=False)
    doc = await env.save("x")
    assert env.scripts.docs["run.py"]["content"] == "x" and doc.content == "x"
    assert await env.read() == "x"
    assert await env.loader.delete_script(**IDENTITY, script_name="run.py") is True
    assert await env.loader.delete_script(**IDENTITY, script_name="run.py") is False


async def test_update_deletes_the_previous_version(env: _Env) -> None:
    await env.save("v1")
    first_ref = env.scripts.docs["run.py"]["content_ref"]
    await env.save("v2")
    await env.save("v3")
    refs = await env.stored_refs()
    assert refs == [env.scripts.docs["run.py"]["content_ref"]] and first_ref not in refs
    assert await env.read() == "v3"


async def test_saving_identical_content_again_still_leaves_one_entry(env: _Env) -> None:
    await env.save("same")
    first_ref = env.scripts.docs["run.py"]["content_ref"]
    await env.save("same")
    second_ref = env.scripts.docs["run.py"]["content_ref"]
    assert first_ref != second_ref
    assert await env.stored_refs() == [second_ref]
    assert await env.read() == "same"


async def test_inline_save_over_an_external_script_retires_its_content(mock_mongo_database: MagicMock) -> None:
    env = _Env(mock_mongo_database)
    await env.save("external")
    env.loader._write_scripts_to_content_store = False  # the flag is switched off (rollback)
    await env.save("inline")
    stored = env.scripts.docs["run.py"]
    assert stored["content"] == "inline" and "content_ref" not in stored
    assert await env.stored_refs() == []


async def test_failed_upsert_removes_the_new_content_and_keeps_the_old_readable(env: _Env) -> None:
    await env.save("v1")
    old_ref = env.scripts.docs["run.py"]["content_ref"]
    env.scripts.fail_next_upsert = True
    with pytest.raises(RuntimeError, match="upsert failed"):
        await env.save("v2")
    assert await env.stored_refs() == [old_ref]
    assert await env.read() == "v1"
    assert env.scripts.docs["run.py"]["digest"] == _digest_of("v1")


async def test_failed_put_prevents_the_metadata_write(env: _Env) -> None:
    env.store.put = AsyncMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="store down"):
        await env.save("x")
    assert env.scripts.upserts == 0 and env.scripts.docs == {}


async def test_cleanup_failure_after_a_successful_save_is_logged_not_raised(
    env: _Env, caplog: pytest.LogCaptureFixture
) -> None:
    await env.save("v1")
    old_ref = env.scripts.docs["run.py"]["content_ref"]
    env.store.delete = AsyncMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        doc = await env.save("v2")
    assert doc.content == "v2" and await env.read() == "v2"
    assert any(
        r.levelno == logging.WARNING and r.exc_info is not None and old_ref in r.getMessage() for r in caplog.records
    )


# --- Concurrency -----------------------------------------------------------------


@pytest.mark.parametrize("concurrent_content", ["Y", "X"])
async def test_concurrent_saves_never_dangle_or_leak(mock_mongo_database: MagicMock, concurrent_content: str) -> None:
    """Save C (content X) has stored its content but not yet upserted when save B finishes completely.

    With ``concurrent_content == "X"`` B saves the very same content (the A -> B -> A case that makes
    content-addressed refs unsafe to delete without a grace window): refs are unique per save, so B retiring its
    predecessor can never remove what C is about to point at.
    """
    env = _Env(mock_mongo_database)
    await env.save("seed")
    other = MongoPluginSkillLoader(
        database=mock_mongo_database, script_content_store=env.store, write_scripts_to_content_store=True
    )
    env.scripts.before_upsert = lambda: env.save(concurrent_content, loader=other)

    await env.save("X")  # C

    final_ref = env.scripts.docs["run.py"]["content_ref"]
    assert await env.read() == "X"
    assert await env.stored_refs() == [final_ref]  # no dangling pointer, and nothing left behind


async def test_reader_follows_the_pointer_when_the_content_it_read_was_just_replaced(env: _Env) -> None:
    await env.save("v1")
    stale_raw = dict(env.scripts.docs["run.py"])  # a reader fetched the document here...
    await env.save("v2")  # ...a save then replaced the pointer and deleted v1's content...
    hydrated = await env.loader._hydrate_script_content(stale_raw)  # ...before the reader fetched v1's content
    assert hydrated["content"] == "v2"


# --- Reads -----------------------------------------------------------------------


async def test_dual_read_old_inline_documents_and_new_external_ones(env: _Env) -> None:
    env.scripts.docs["old.py"] = {"_id": 9, **IDENTITY, "script_name": "old.py", "path": "p", "content": "legacy"}
    await env.save("new")
    assert await env.read(script_name="old.py") == "legacy"
    assert await env.read() == "new"
    docs = await env.loader.list_script_documents(author="user-1", plugin_name="my-plugin", skill_name="my-skill")
    assert [(d.script_name, d.content) for d in docs] == [("old.py", "legacy"), ("run.py", "new")]


async def test_inline_content_wins_and_empty_inline_is_not_missing(env: _Env) -> None:
    env.scripts.docs["run.py"] = {
        "_id": 1,
        **IDENTITY,
        "script_name": "run.py",
        "content": "",
        "content_ref": f"{KEY}@sha256:x#gone",
    }
    assert await env.read() == ""


@pytest.mark.parametrize("with_ref", [True, False])
async def test_missing_content_raises_instead_of_returning_empty(env: _Env, with_ref: bool) -> None:
    doc: dict[str, Any] = {"_id": 1, **IDENTITY, "script_name": "run.py"}
    if with_ref:
        doc["content_ref"] = f"{KEY}@sha256:x#lost"
    env.scripts.docs["run.py"] = doc
    with pytest.raises(SkillLoaderError, match="content missing"):
        await env.read()
    with pytest.raises(SkillLoaderError, match="content missing"):
        await env.loader.list_script_documents(author="user-1", plugin_name="my-plugin", skill_name="my-skill")


async def test_external_document_without_a_configured_store_raises_a_clear_error(
    mock_mongo_database: MagicMock,
) -> None:
    env = _Env(mock_mongo_database, with_store=False)
    env.scripts.docs["run.py"] = {"_id": 1, **IDENTITY, "script_name": "run.py", "content_ref": f"{KEY}@sha256:x#r"}
    with pytest.raises(SkillLoaderError, match="no ScriptContentStore is configured"):
        await env.read()


async def test_load_snapshot_never_fetches_script_content(env: _Env) -> None:
    skill_doc = MongoPluginSkillDocument(
        **IDENTITY, content="# s", path="p", digest=_digest_of("# s"), size=3
    ).to_mongo_dict()
    env.db["plugin_skills"].find = MagicMock(return_value=_AsyncIter([skill_doc]))
    env.db["plugin_references"].find = MagicMock(return_value=_AsyncIter([]))
    await env.save("script body")
    env.store.get = AsyncMock(side_effect=AssertionError("snapshot must not read content"))  # type: ignore[method-assign]
    snapshot = await env.loader.load_snapshot(author="user-1")
    manifest = snapshot.details_by_name["my-skill"].summary.manifest
    assert isinstance(manifest, tuple) and any(e.path == "scripts/run.py" for e in manifest)


# --- Deletes -----------------------------------------------------------------------


async def test_delete_script_removes_metadata_and_its_content_only(env: _Env) -> None:
    await env.save("a")
    await env.save("b", script_name="other.py")
    other_ref = env.scripts.docs["other.py"]["content_ref"]
    assert await env.loader.delete_script(**IDENTITY, script_name="run.py") is True
    assert "run.py" not in env.scripts.docs
    assert await env.stored_refs() == [other_ref]
    assert await env.loader.delete_script(**IDENTITY, script_name="run.py") is False


async def test_delete_skill_removes_every_scripts_content_and_nothing_else(env: _Env) -> None:
    await env.save("a")
    await env.save("b", script_name="b.py")
    await env.store.put(ref="user-1/my-plugin/other-skill/c.py@sha256:z#r", content="c")  # another skill's content
    await env.loader.delete_skill(**IDENTITY)
    assert env.scripts.docs == {}
    assert await env.stored_refs() == ["user-1/my-plugin/other-skill/c.py@sha256:z#r"]


@pytest.mark.parametrize("target", ["script", "skill"])
async def test_delete_content_failure_is_logged_and_the_delete_still_succeeds(
    env: _Env, caplog: pytest.LogCaptureFixture, target: str
) -> None:
    await env.save("a")
    ref = env.scripts.docs["run.py"]["content_ref"]
    env.store.delete = AsyncMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        if target == "script":
            assert await env.loader.delete_script(**IDENTITY, script_name="run.py") is True
        else:
            assert await env.loader.delete_skill(**IDENTITY) is True
    assert env.scripts.docs == {}  # the metadata is gone
    assert any(
        r.levelno == logging.WARNING and r.exc_info is not None and ref in r.getMessage() for r in caplog.records
    )


# --- Backfill ------------------------------------------------------------------------


async def test_migrate_moves_inline_content_into_the_store(env: _Env) -> None:
    env.scripts.docs["a.py"] = {"_id": 1, **IDENTITY, "script_name": "a.py", "content": "legacy A"}
    env.scripts.docs["b.py"] = {"_id": 2, **IDENTITY, "script_name": "b.py", "content": "legacy B"}
    env.db["plugin_scripts"].find = MagicMock(return_value=_AsyncIter([dict(d) for d in env.scripts.docs.values()]))
    assert await env.loader.migrate_inline_script_content(batch_size=10) == 2
    assert all("content" not in d and "content_ref" in d for d in env.scripts.docs.values())
    assert await env.read(script_name="a.py") == "legacy A"
    assert await env.read(script_name="b.py") == "legacy B"
    assert len(await env.stored_refs()) == 2


async def test_migrate_discards_its_copy_when_the_script_changed_meanwhile(env: _Env) -> None:
    env.scripts.docs["a.py"] = {"_id": 1, **IDENTITY, "script_name": "a.py", "content": "legacy"}
    snapshot = dict(env.scripts.docs["a.py"])
    env.db["plugin_scripts"].find = MagicMock(return_value=_AsyncIter([snapshot]))
    # A real save rewrites content, digest and date_modified together.
    env.scripts.docs["a.py"].update({"content": "edited meanwhile", "digest": "sha256:new", "date_modified": "later"})
    assert await env.loader.migrate_inline_script_content(batch_size=10) == 0
    assert env.scripts.docs["a.py"]["content"] == "edited meanwhile"
    assert await env.stored_refs() == []


async def test_migrate_without_store_is_a_noop(mock_mongo_database: MagicMock) -> None:
    assert await MongoPluginSkillLoader(database=mock_mongo_database).migrate_inline_script_content() == 0


# --- Injection and configuration ------------------------------------------------------


def test_container_resolves_a_registered_content_store_and_none_otherwise() -> None:
    container = SimpleContainer()
    container.clear_singletons()  # the singleton registry is shared across SimpleContainer instances
    try:
        assert _resolve_script_content_store(c=container) is None
        store = KeyValueScriptContentStore(key_value=MemoryStore())
        container.singleton(ScriptContentStore, lambda c: store)
        assert _resolve_script_content_store(c=container) is store
    finally:
        container.clear_singletons()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("with_store", [True, False])
def test_factory_hands_the_store_and_flag_to_the_loader(
    mock_mongo_database: MagicMock, enabled: bool, with_store: bool
) -> None:
    env_vars = MagicMock()
    env_vars.enable_skill_history = False
    env_vars.plugin_scripts_external_content_enabled = enabled
    env_vars.plugin_skills_collection = env_vars.plugin_references_collection = None
    env_vars.plugin_scripts_collection = env_vars.plugins_collection = None
    database_factory = MagicMock()
    database_factory.create_database.return_value = mock_mongo_database
    store = KeyValueScriptContentStore(key_value=MemoryStore()) if with_store else None

    created = PluginSkillStoreFactory(
        mongo_database_factory=database_factory, environment_variables=env_vars, script_content_store=store
    ).create()

    assert isinstance(created, MongoPluginSkillLoader)
    assert created._script_content_store is store
    assert created._write_scripts_to_content_store is enabled


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("false", False),
        ("", False),
        ("true", True),
        ("TRUE", True),
        (" 1 ", True),
        ("yes", True),
        ("no", False),
    ],
)
def test_external_content_flag_parsing(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool) -> None:
    monkeypatch.delenv("PLUGIN_SCRIPTS_EXTERNAL_CONTENT_ENABLED", raising=False)
    if raw is not None:
        monkeypatch.setenv("PLUGIN_SCRIPTS_EXTERNAL_CONTENT_ENABLED", raw)
    assert LangchainAISkillsFrameworkEnvironmentVariables().plugin_scripts_external_content_enabled is expected
