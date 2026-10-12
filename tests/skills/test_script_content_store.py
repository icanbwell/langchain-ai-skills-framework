"""Tests for GridFS-backed script content: the store itself and the loader's use of it."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gridfs.errors import NoFile

from langchain_ai_skills_framework.loaders.exceptions.skill_loader_error import SkillLoaderError
from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginSkillDocument
from langchain_ai_skills_framework.persistence.script_content_store import (
    GridFSScriptContentStore,
    ScriptContentStore,
    build_script_content_key,
    build_script_content_ref,
)

KEY = "user-1/my-plugin/my-skill/run.py"


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


class _FakeStream:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeBucket:
    """In-memory stand-in for AsyncIOMotorGridFSBucket supporting the calls the store makes."""

    def __init__(self) -> None:
        self.files: list[dict[str, Any]] = []
        self._next_id = 0
        self.clock = datetime.now(UTC) - timedelta(hours=1)  # uploads start outside the GC grace window

    async def upload_from_stream(self, filename: str, source: bytes, metadata: dict[str, Any] | None = None) -> int:
        self._next_id += 1
        self.clock += timedelta(seconds=1)
        self.files.append(
            {"_id": self._next_id, "filename": filename, "data": source, "metadata": metadata, "uploadDate": self.clock}
        )
        return self._next_id

    def find(self, query: dict[str, Any]) -> Any:
        def matches(f: dict[str, Any]) -> bool:
            for field, cond in query.items():
                if field == "filename" and isinstance(cond, dict):
                    if "$regex" in cond and not re.search(cond["$regex"], f["filename"]):
                        return False
                    if "$ne" in cond and f["filename"] == cond["$ne"]:
                        return False
                elif isinstance(cond, dict) and "$lt" in cond:
                    if not f[field] < cond["$lt"]:
                        return False
                elif f[field] != cond:
                    return False
            return True

        found = [f for f in self.files if matches(f)]

        class _Cursor(_AsyncIter):
            async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
                return found[:length]

        return _Cursor(found)

    async def open_download_stream_by_name(self, filename: str) -> _FakeStream:
        matching = [f for f in self.files if f["filename"] == filename]
        if not matching:
            raise NoFile(filename)
        return _FakeStream(max(matching, key=lambda f: f["uploadDate"])["data"])

    async def delete(self, file_id: int) -> None:
        self.files = [f for f in self.files if f["_id"] != file_id]


@pytest.fixture
def bucket() -> _FakeBucket:
    return _FakeBucket()


@pytest.fixture
def gridfs_store(bucket: _FakeBucket) -> GridFSScriptContentStore:
    with patch(
        "langchain_ai_skills_framework.persistence.script_content_store.AsyncIOMotorGridFSBucket",
        return_value=bucket,
    ):
        return GridFSScriptContentStore(database=MagicMock())


class _MemoryStore:
    """Dict-backed ScriptContentStore for loader tests."""

    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}
        self.get_calls = 0
        self.delete_versions_calls: list[dict[str, Any]] = []

    async def put(self, *, ref: str, content: str, metadata: Mapping[str, object] | None = None) -> None:
        self.blobs.setdefault(ref, content)

    async def get(self, *, ref: str) -> str | None:
        self.get_calls += 1
        return self.blobs.get(ref)

    async def delete_versions(
        self, *, script_key: str, keep_ref: str | None = None, older_than: datetime | None = None
    ) -> int:
        self.delete_versions_calls.append({"script_key": script_key, "keep_ref": keep_ref, "older_than": older_than})
        doomed = [k for k in self.blobs if k.startswith(script_key + "@") and k != keep_ref]
        for k in doomed:
            del self.blobs[k]
        return len(doomed)

    async def delete_prefix(self, *, prefix: str) -> int:
        doomed = [k for k in self.blobs if k.startswith(prefix)]
        for k in doomed:
            del self.blobs[k]
        return len(doomed)


class _FakeScriptsCollection:
    """Single-document stateful stand-in for the plugin_scripts collection."""

    def __init__(self) -> None:
        self.doc: dict[str, Any] | None = None
        self.fail_next_upsert = False

    async def find_one_and_update(
        self, filter_: dict[str, Any], update: dict[str, dict[str, Any]], **kwargs: Any
    ) -> dict[str, Any]:
        if self.fail_next_upsert:
            self.fail_next_upsert = False
            raise RuntimeError("upsert failed")
        if self.doc is None:
            self.doc = {"_id": 1, **update.get("$setOnInsert", {})}
        self.doc.update(update.get("$set", {}))
        for field in update.get("$unset", {}):
            self.doc.pop(field, None)
        return dict(self.doc)

    async def find_one(self, filter_: dict[str, Any], projection: Any = None) -> dict[str, Any] | None:
        return dict(self.doc) if self.doc is not None else None


def _digest_of(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


def _ref_of(content: str) -> str:
    return f"{KEY}@{_digest_of(content)}"


# --- GridFSScriptContentStore ------------------------------------------------


def test_build_key_and_ref() -> None:
    assert (
        build_script_content_key(author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py")
        == KEY
    )
    assert build_script_content_ref(script_key=KEY, digest="sha256:ab") == KEY + "@sha256:ab"


async def test_put_get_roundtrip_and_missing(gridfs_store: GridFSScriptContentStore) -> None:
    assert await gridfs_store.get(ref=_ref_of("x")) is None
    await gridfs_store.put(ref=_ref_of("print('héllo')"), content="print('héllo')", metadata={"digest": "d"})
    assert await gridfs_store.get(ref=_ref_of("print('héllo')")) == "print('héllo')"


async def test_put_is_idempotent_and_never_overwrites(
    gridfs_store: GridFSScriptContentStore, bucket: _FakeBucket
) -> None:
    await gridfs_store.put(ref=_ref_of("a"), content="a")
    await gridfs_store.put(ref=_ref_of("a"), content="a")
    await gridfs_store.put(ref=_ref_of("b"), content="b")
    assert sorted(f["data"] for f in bucket.files) == [b"a", b"b"]


async def test_delete_versions_removes_only_stale_blobs_of_that_script(
    gridfs_store: GridFSScriptContentStore, bucket: _FakeBucket
) -> None:
    other = "user-1/my-plugin/my-skill/run.py.bak"  # name merely starts with the script's key
    await gridfs_store.put(ref=_ref_of("v1"), content="v1")
    await gridfs_store.put(ref=_ref_of("v2"), content="v2")
    await gridfs_store.put(ref=f"{other}@{_digest_of('o')}", content="o")
    await gridfs_store.put(ref=f"user-1/my-plugin/my-skill/run.py@{_digest_of('v3')}", content="v3")
    assert await gridfs_store.delete_versions(script_key=KEY, keep_ref=_ref_of("v2")) == 2
    assert sorted(f["data"] for f in bucket.files) == [b"o", b"v2"]


async def test_delete_versions_respects_grace_cutoff(
    gridfs_store: GridFSScriptContentStore, bucket: _FakeBucket
) -> None:
    await gridfs_store.put(ref=_ref_of("old"), content="old")
    cutoff = bucket.clock + timedelta(seconds=1)
    await gridfs_store.put(ref=_ref_of("new"), content="new")  # uploaded after the cutoff
    await gridfs_store.put(ref=_ref_of("keep"), content="keep")
    assert await gridfs_store.delete_versions(script_key=KEY, keep_ref=_ref_of("keep"), older_than=cutoff) == 1
    assert sorted(f["data"] for f in bucket.files) == [b"keep", b"new"]


async def test_delete_prefix(gridfs_store: GridFSScriptContentStore, bucket: _FakeBucket) -> None:
    await gridfs_store.put(ref="u/p/s/a.py@x", content="a")
    await gridfs_store.put(ref="u/p/s/b.py@x", content="b")
    await gridfs_store.put(ref="u/p/s2/c.py@x", content="c")
    assert await gridfs_store.delete_prefix(prefix="u/p/s/") == 2
    assert [f["filename"] for f in bucket.files] == ["u/p/s2/c.py@x"]


def test_memory_store_satisfies_protocol() -> None:
    assert isinstance(_MemoryStore(), ScriptContentStore)


# --- Loader ------------------------------------------------------------------


def _script_raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "_id": 1,
        "author": "user-1",
        "plugin_name": "my-plugin",
        "skill_name": "my-skill",
        "script_name": "run.py",
        "path": "p",
        "digest": "sha256:x",
        "size": 1,
    }
    raw.update(overrides)
    return raw


def _loader(db: MagicMock, store: _MemoryStore, *, write_gridfs: bool = True) -> MongoPluginSkillLoader:
    return MongoPluginSkillLoader(database=db, script_content_store=store, write_scripts_to_gridfs=write_gridfs)


async def _save(loader: MongoPluginSkillLoader, db: MagicMock, content: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_find_one_and_update(
        filter_: dict[str, object], update: dict[str, dict[str, object]], **kwargs: object
    ) -> dict[str, object]:
        captured.update(update)
        return {
            "author": "user-1",
            "plugin_name": "my-plugin",
            "skill_name": "my-skill",
            **{k: v for k, v in update["$set"].items()},
            **update["$setOnInsert"],
        }

    db["plugin_scripts"].find_one_and_update = fake_find_one_and_update
    db["plugin_skills"].find_one = AsyncMock(return_value=None)
    doc = await loader.save_script(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        script_name="run.py",
        content=content,
        path="p",
    )
    assert doc.content == content
    return captured


async def test_save_script_puts_blob_and_points_metadata_at_it(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    update = await _save(_loader(mock_mongo_database, store), mock_mongo_database, "print(1)")
    assert store.blobs == {_ref_of("print(1)"): "print(1)"}
    assert "content" not in update["$set"]
    assert update["$set"]["content_ref"] == _ref_of("print(1)")
    assert update["$set"]["digest"] == _digest_of("print(1)")
    assert update["$unset"] == {"content": ""}


async def test_save_script_inline_when_gridfs_writes_disabled(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    update = await _save(_loader(mock_mongo_database, store, write_gridfs=False), mock_mongo_database, "print(1)")
    assert store.blobs == {}
    assert update["$set"]["content"] == "print(1)"
    assert update["$unset"] == {"content_ref": ""}  # a stale pointer must not outlive an inline write


async def test_save_script_without_store_is_unchanged(mock_mongo_database: MagicMock) -> None:
    update = await _save(MongoPluginSkillLoader(database=mock_mongo_database), mock_mongo_database, "x")
    assert update["$set"]["content"] == "x"
    assert "$unset" not in update


async def test_put_failure_prevents_metadata_write(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    store.put = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    mock_mongo_database["plugin_scripts"].find_one_and_update = AsyncMock()
    mock_mongo_database["plugin_skills"].find_one = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError):
        await _loader(mock_mongo_database, store).save_script(
            author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py", content="x", path="p"
        )
    mock_mongo_database["plugin_scripts"].find_one_and_update.assert_not_awaited()


# --- End to end against a stateful metadata fake and the fake bucket ---------------


@pytest.fixture
def e2e(
    mock_mongo_database: MagicMock, gridfs_store: GridFSScriptContentStore
) -> tuple[MongoPluginSkillLoader, _FakeScriptsCollection, _FakeBucket, GridFSScriptContentStore]:
    scripts = _FakeScriptsCollection()
    mock_mongo_database["plugin_scripts"].find_one_and_update = scripts.find_one_and_update
    mock_mongo_database["plugin_scripts"].find_one = scripts.find_one
    mock_mongo_database["plugin_skills"].find_one = AsyncMock(return_value=None)
    loader = MongoPluginSkillLoader(database=mock_mongo_database, script_content_store=gridfs_store)
    bucket = cast(_FakeBucket, gridfs_store._bucket)
    return loader, scripts, bucket, gridfs_store


async def _save_e2e(loader: MongoPluginSkillLoader, content: str) -> None:
    await loader.save_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py", content=content, path="p"
    )


async def _read_e2e(loader: MongoPluginSkillLoader) -> str:
    return await loader.read_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )


async def test_failed_upsert_leaves_old_content_readable_and_old_blob_intact(
    e2e: tuple[MongoPluginSkillLoader, _FakeScriptsCollection, _FakeBucket, GridFSScriptContentStore],
) -> None:
    loader, scripts, bucket, _ = e2e
    await _save_e2e(loader, "v1")
    scripts.fail_next_upsert = True
    with pytest.raises(RuntimeError):
        await _save_e2e(loader, "v2")
    assert await _read_e2e(loader) == "v1"
    assert scripts.doc is not None and scripts.doc["digest"] == _digest_of("v1")
    assert _ref_of("v1") in {f["filename"] for f in bucket.files}  # the old blob was never touched


async def test_interleaved_saves_never_mismatch_digest_and_content(
    e2e: tuple[MongoPluginSkillLoader, _FakeScriptsCollection, _FakeBucket, GridFSScriptContentStore],
) -> None:
    loader, scripts, _, store = e2e
    # A and B both put their blobs; B's upsert lands first, then A's (the order that used to desync them).
    await store.put(ref=_ref_of("A"), content="A")
    await store.put(ref=_ref_of("B"), content="B")
    for content in ("B", "A"):
        await scripts.find_one_and_update(
            {},
            {
                "$set": {"content_ref": _ref_of(content), "digest": _digest_of(content), "size": 1},
                "$unset": {"content": ""},
                "$setOnInsert": {
                    "author": "user-1",
                    "plugin_name": "my-plugin",
                    "skill_name": "my-skill",
                    "script_name": "run.py",
                },
            },
        )
    assert scripts.doc is not None
    assert await _read_e2e(loader) == "A"
    assert _digest_of(await _read_e2e(loader)) == scripts.doc["digest"]
    # Each saver's GC then runs; a grace window plus the re-read ref means neither deletes the live blob.
    for _ in range(2):
        await loader._collect_stale_script_blobs(script_key=KEY, document_id=1)
    assert await _read_e2e(loader) == "A"


async def test_gc_after_save_removes_only_stale_old_blobs(
    e2e: tuple[MongoPluginSkillLoader, _FakeScriptsCollection, _FakeBucket, GridFSScriptContentStore],
) -> None:
    loader, _, bucket, _ = e2e
    await _save_e2e(loader, "v1")  # uploaded an hour ago
    bucket.clock = datetime.now(UTC)  # later uploads are recent
    await _save_e2e(loader, "v2")
    assert {f["filename"] for f in bucket.files} == {_ref_of("v2")}
    await _save_e2e(loader, "v3")  # v2 is recent, so it survives this save's GC (grace window)
    assert {f["filename"] for f in bucket.files} == {_ref_of("v2"), _ref_of("v3")}
    assert await _read_e2e(loader) == "v3"


async def test_saving_identical_content_is_idempotent(
    e2e: tuple[MongoPluginSkillLoader, _FakeScriptsCollection, _FakeBucket, GridFSScriptContentStore],
) -> None:
    loader, _, bucket, _ = e2e
    await _save_e2e(loader, "same")
    await _save_e2e(loader, "same")
    assert len(bucket.files) == 1


# --- Reads -----------------------------------------------------------------------


async def test_read_script_from_gridfs_and_inline_fallback(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    store.blobs[_ref_of("from-blob")] = "from-blob"
    loader = _loader(mock_mongo_database, store)
    mock_mongo_database["plugin_scripts"].find_one = AsyncMock(
        return_value=_script_raw(content_ref=_ref_of("from-blob"))
    )
    assert (
        await loader.read_script(author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py")
        == "from-blob"
    )

    mock_mongo_database["plugin_scripts"].find_one = AsyncMock(
        return_value=_script_raw(content="inline", content_ref=_ref_of("from-blob"))
    )
    assert (
        await loader.read_script(author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py")
        == "inline"
    )
    assert store.get_calls == 1  # inline wins without touching the store


async def test_read_script_empty_inline_content_is_not_treated_as_missing(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    mock_mongo_database["plugin_scripts"].find_one = AsyncMock(return_value=_script_raw(content=""))
    loader = _loader(mock_mongo_database, store)
    assert (
        await loader.read_script(author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py")
        == ""
    )
    assert store.get_calls == 0


@pytest.mark.parametrize("raw_extra", [{"content_ref": "user-1/my-plugin/my-skill/run.py@sha256:gone"}, {}])
async def test_read_script_raises_when_content_is_missing(
    mock_mongo_database: MagicMock, raw_extra: dict[str, Any]
) -> None:
    mock_mongo_database["plugin_scripts"].find_one = AsyncMock(return_value=_script_raw(**raw_extra))
    with pytest.raises(SkillLoaderError, match="content missing"):
        await _loader(mock_mongo_database, _MemoryStore()).read_script(
            author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
        )


async def test_list_script_documents_rehydrates(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    store.blobs[_ref_of("A")] = "A"
    mock_mongo_database["plugin_scripts"].find = MagicMock(
        return_value=_AsyncIter(
            [
                _script_raw(script_name="a.py", content_ref=_ref_of("A")),
                _script_raw(_id=2, script_name="b.py", content="B"),
            ]
        )
    )
    docs = await _loader(mock_mongo_database, store).list_script_documents(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill"
    )
    assert [(d.script_name, d.content) for d in docs] == [("a.py", "A"), ("b.py", "B")]


async def test_list_script_documents_raises_on_missing_blob(mock_mongo_database: MagicMock) -> None:
    # Callers (e.g. publishing) take this content verbatim, so a lost blob must not become an empty script.
    mock_mongo_database["plugin_scripts"].find = MagicMock(
        return_value=_AsyncIter([_script_raw(content_ref=_ref_of("lost"))])
    )
    with pytest.raises(SkillLoaderError, match="content missing"):
        await _loader(mock_mongo_database, _MemoryStore()).list_script_documents(
            author="user-1", plugin_name="my-plugin", skill_name="my-skill"
        )


# --- Deletes -----------------------------------------------------------------------


async def test_delete_script_removes_metadata_then_all_its_blobs(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    store.blobs.update({_ref_of("x"): "x", _ref_of("y"): "y", "user-1/my-plugin/my-skill/other.py@sha256:z": "z"})
    mock_mongo_database["plugin_scripts"].delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    assert await _loader(mock_mongo_database, store).delete_script(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill", script_name="run.py"
    )
    assert list(store.blobs) == ["user-1/my-plugin/my-skill/other.py@sha256:z"]


async def test_delete_skill_removes_only_that_skills_blobs(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    store.blobs.update(
        {
            _ref_of("x"): "x",
            "user-1/my-plugin/my-skill/b.py@sha256:y": "y",
            "user-1/my-plugin/my-skill-2/c.py@sha256:z": "z",  # a different skill sharing a name prefix
        }
    )
    mock_mongo_database["plugin_skills"].delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    await _loader(mock_mongo_database, store).delete_skill(
        author="user-1", plugin_name="my-plugin", skill_name="my-skill"
    )
    assert list(store.blobs) == ["user-1/my-plugin/my-skill-2/c.py@sha256:z"]


# --- Snapshot / manifest never fetch content -----------------------------------------


async def test_load_snapshot_does_not_fetch_script_content(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    skill_doc = MongoPluginSkillDocument(
        author="user-1",
        plugin_name="my-plugin",
        skill_name="my-skill",
        content="# s",
        path="p",
        digest=_digest_of("# s"),
        size=3,
    ).to_mongo_dict()
    mock_mongo_database["plugin_skills"].find = MagicMock(return_value=_AsyncIter([skill_doc]))
    mock_mongo_database["plugin_references"].find = MagicMock(return_value=_AsyncIter([]))
    # Even a script whose blob is gone must not break the snapshot: manifests only need digest and size.
    mock_mongo_database["plugin_scripts"].find = MagicMock(
        return_value=_AsyncIter([_script_raw(content_ref=_ref_of("lost"), digest=_digest_of("v"), size=1)])
    )
    snapshot = await _loader(mock_mongo_database, store).load_snapshot(author="user-1")
    assert store.get_calls == 0
    manifest = snapshot.details_by_name["my-skill"].summary.manifest
    assert isinstance(manifest, tuple) and any(e.path == "scripts/run.py" for e in manifest)


# --- Backfill ------------------------------------------------------------------------


async def test_migrate_inline_script_content_sets_content_ref(mock_mongo_database: MagicMock) -> None:
    store = _MemoryStore()
    coll = mock_mongo_database["plugin_scripts"]
    coll.find = MagicMock(
        return_value=_AsyncIter([_script_raw(content="legacy"), _script_raw(_id=2, script_name="b.py", content="B")])
    )
    coll.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    assert await _loader(mock_mongo_database, store).migrate_inline_script_content(batch_size=10) == 2
    assert store.blobs == {_ref_of("legacy"): "legacy", f"user-1/my-plugin/my-skill/b.py@{_digest_of('B')}": "B"}
    first_filter, first_update = coll.update_one.await_args_list[0].args
    assert first_filter == {"_id": 1, "content": "legacy"}  # only flips if content is unchanged
    assert first_update == {"$set": {"content_ref": _ref_of("legacy")}, "$unset": {"content": ""}}


async def test_migrate_without_store_is_noop(mock_mongo_database: MagicMock) -> None:
    assert await MongoPluginSkillLoader(database=mock_mongo_database).migrate_inline_script_content() == 0
