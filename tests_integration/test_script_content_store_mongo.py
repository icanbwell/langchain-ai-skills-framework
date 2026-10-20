"""Integration tests for the external script content store against a real MongoDB.

The unit tests in ``tests/skills/test_script_content_store.py`` run the loader on a fake ``plugin_scripts``
collection. The guarantee that matters under concurrency is MongoDB's own: one ``find_one_and_update`` /
``find_one_and_delete`` atomically swaps (or removes) the document and hands back the *previous* one, so exactly one
writer retires each ``content_ref``. That is only provable against a real server.

Backends: ``plugin_scripts`` is a real Motor collection; script content lives in py-key-value's real
``MongoDBStore`` (same server, separate database), i.e. the same driver path a consumer would use.

Requires ``MONGO_URL`` (set by ``make tests-integration``); skipped otherwise. Each test uses its own databases,
dropped afterwards.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import pytest
from key_value.aio.stores.mongodb import MongoDBStore
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import AsyncMongoClient

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginScriptDocument
from langchain_ai_skills_framework.persistence.script_content_store import (
    DEFAULT_SCRIPT_CONTENT_COLLECTION,
    KeyValueScriptContentStore,
)

MONGO_URL = os.environ.get("MONGO_URL")

pytestmark = pytest.mark.skipif(
    not MONGO_URL, reason="MONGO_URL is not set; start MongoDB with `make tests-integration`"
)

IDENTITY: dict[str, str] = {"author": "user-1", "plugin_name": "my-plugin", "skill_name": "my-skill"}
SCRIPT = "run.py"
CONCURRENT_WRITERS = 12


@dataclass
class Env:
    database: AsyncIOMotorDatabase[dict[str, object]]
    store: KeyValueScriptContentStore
    loader: MongoPluginSkillLoader
    inline_loader: MongoPluginSkillLoader  # same collections, flag off: writes inline like an older pod

    kv_store: MongoDBStore  # the py-key-value store behind ``store``, for wrapping it in a test double
    kv_database: Any  # the content store's own Mongo database, for listing what is physically stored

    @property
    def scripts(self) -> Any:
        return self.database["plugin_scripts"]

    async def save(
        self, content: str, *, script_name: str = SCRIPT, loader: MongoPluginSkillLoader | None = None
    ) -> MongoPluginScriptDocument:
        return await (loader or self.loader).save_script(**IDENTITY, script_name=script_name, content=content)

    async def read(self, *, script_name: str = SCRIPT) -> str:
        return await self.loader.read_script(**IDENTITY, script_name=script_name)

    async def raw(self, *, script_name: str = SCRIPT) -> dict[str, Any] | None:
        return cast(dict[str, Any] | None, await self.scripts.find_one({**IDENTITY, "script_name": script_name}))

    async def stored_content(self, ref: str) -> str | None:
        return await self.store.get(ref=ref)

    async def stored_refs(self) -> list[str]:
        """Every key physically present in the content store's collection (not just the ones the loader knows)."""
        # MongoDBStore keeps one Mongo collection per py-key-value collection, named after it.
        kv_collection = self.kv_database[DEFAULT_SCRIPT_CONTENT_COLLECTION]
        return sorted([doc["key"] async for doc in kv_collection.find({}, {"key": 1})])


@pytest.fixture
async def env() -> AsyncIterator[Env]:
    assert MONGO_URL
    suffix = uuid.uuid4().hex[:12]
    meta_db_name = f"it_scripts_meta_{suffix}"
    kv_db_name = f"it_scripts_kv_{suffix}"
    motor_client: AsyncIOMotorClient[dict[str, object]] = AsyncIOMotorClient(MONGO_URL)
    kv_client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(MONGO_URL)
    database = motor_client[meta_db_name]
    kv_store = MongoDBStore(client=kv_client, db_name=kv_db_name)
    store = KeyValueScriptContentStore(key_value=kv_store)
    loader = MongoPluginSkillLoader(
        database=database, script_content_store=store, write_scripts_to_content_store=True, auto_repair=False
    )
    inline_loader = MongoPluginSkillLoader(
        database=database, script_content_store=store, write_scripts_to_content_store=False, auto_repair=False
    )
    await loader.ensure_indexes()
    environment = Env(
        database=database,
        store=store,
        loader=loader,
        inline_loader=inline_loader,
        kv_store=kv_store,
        kv_database=kv_client[kv_db_name],
    )
    try:
        yield environment
    finally:
        await motor_client.drop_database(meta_db_name)
        await kv_client.drop_database(kv_db_name)
        await kv_store.close()
        motor_client.close()
        await kv_client.close()


async def test_save_with_flag_on_keeps_content_in_the_store_and_only_a_ref_in_mongo(env: Env) -> None:
    doc = await env.save("print('hi')")
    assert doc.content == "print('hi')"

    raw = await env.raw()
    assert raw is not None
    assert "content" not in raw
    ref = raw["content_ref"]
    assert ref.startswith("user-1/my-plugin/my-skill/run.py@")
    assert raw["digest"] and raw["size"] == len("print('hi')")
    assert await env.stored_content(ref) == "print('hi')"
    assert await env.stored_refs() == [ref]
    assert await env.read() == "print('hi')"


async def test_update_writes_the_new_ref_and_deletes_the_old_entry(env: Env) -> None:
    await env.save("v1")
    ref1 = (await env.raw() or {})["content_ref"]
    await env.save("v2")
    ref2 = (await env.raw() or {})["content_ref"]

    assert ref2 != ref1
    assert await env.stored_content(ref1) is None
    assert await env.stored_content(ref2) == "v2"
    assert await env.stored_refs() == [ref2]
    assert await env.read() == "v2"


async def test_identical_resave_gets_a_new_ref_and_still_leaves_exactly_one_entry(env: Env) -> None:
    await env.save("same")
    ref1 = (await env.raw() or {})["content_ref"]
    await env.save("same")
    ref2 = (await env.raw() or {})["content_ref"]

    assert ref2 != ref1  # unique per save, even for identical content
    assert await env.stored_content(ref1) is None
    assert await env.stored_refs() == [ref2]
    assert await env.read() == "same"
    assert await env.scripts.count_documents({**IDENTITY, "script_name": SCRIPT}) == 1


async def test_concurrent_saves_leave_one_ref_and_no_orphaned_content(env: Env) -> None:
    await env.save("seed")
    contents = [f"writer-{i}" for i in range(CONCURRENT_WRITERS)]
    contents.append("writer-0")  # an identical payload racing with another writer

    docs = await asyncio.gather(*(env.save(content) for content in contents))
    assert len(docs) == len(contents)

    assert await env.scripts.count_documents({**IDENTITY, "script_name": SCRIPT}) == 1
    raw = await env.raw()
    assert raw is not None and "content" not in raw
    final_ref = raw["content_ref"]
    # Every retired ref (the seed and all losing writers) is gone; only the winner's content remains.
    assert await env.stored_refs() == [final_ref]
    assert await env.read() in contents
    assert await env.stored_content(final_ref) == await env.read()


async def test_concurrent_saves_to_different_scripts_do_not_interfere(env: Env) -> None:
    names = [f"s{i}.py" for i in range(CONCURRENT_WRITERS)]
    await asyncio.gather(*(env.save(f"body-{n}", script_name=n) for n in names))
    await asyncio.gather(*(env.save(f"body2-{n}", script_name=n) for n in names))

    refs = sorted([(await env.raw(script_name=n) or {})["content_ref"] for n in names])
    assert await env.stored_refs() == refs
    for n in names:
        assert await env.read(script_name=n) == f"body2-{n}"


async def test_delete_script_removes_its_store_entry_only(env: Env) -> None:
    await env.save("a")
    await env.save("b", script_name="other.py")
    other_ref = (await env.raw(script_name="other.py") or {})["content_ref"]

    assert await env.loader.delete_script(**IDENTITY, script_name=SCRIPT) is True
    assert await env.raw() is None
    assert await env.stored_refs() == [other_ref]
    assert await env.loader.delete_script(**IDENTITY, script_name=SCRIPT) is False


async def test_delete_skill_removes_every_store_entry_of_the_skill(env: Env) -> None:
    await env.loader.save_skill(author="user-1", plugin_name="my-plugin", skill_name="my-skill", content="# my-skill\n")
    names = ["a.py", "b.py", "c.py"]
    for n in names:
        await env.save(f"body-{n}", script_name=n)
    await env.loader.save_skill(author="user-1", plugin_name="my-plugin", skill_name="keep-skill", content="# keep\n")
    await env.loader.save_script(
        author="user-1", plugin_name="my-plugin", skill_name="keep-skill", script_name="k.py", content="keep me"
    )
    keep_doc = await env.scripts.find_one({"skill_name": "keep-skill"})
    assert keep_doc is not None
    keep_ref = keep_doc["content_ref"]

    assert await env.loader.delete_skill(**IDENTITY) is True

    assert await env.scripts.count_documents({"skill_name": "my-skill"}) == 0
    assert await env.stored_refs() == [keep_ref]


async def test_dual_read_inline_document_is_readable_with_the_flag_on(env: Env) -> None:
    await env.save("legacy body", loader=env.inline_loader)
    raw = await env.raw()
    assert raw is not None and raw["content"] == "legacy body" and "content_ref" not in raw
    assert await env.stored_refs() == []

    assert await env.read() == "legacy body"
    docs = await env.loader.list_script_documents(**IDENTITY)
    assert [(d.script_name, d.content) for d in docs] == [(SCRIPT, "legacy body")]


async def test_migrate_moves_inline_content_into_the_store(env: Env) -> None:
    await env.save("inline-1", script_name="a.py", loader=env.inline_loader)
    await env.save("inline-2", script_name="b.py", loader=env.inline_loader)
    await env.save("external", script_name="c.py")

    assert await env.loader.migrate_inline_script_content() == 2
    assert await env.loader.migrate_inline_script_content() == 0  # idempotent

    for name, body in [("a.py", "inline-1"), ("b.py", "inline-2"), ("c.py", "external")]:
        raw = await env.raw(script_name=name)
        assert raw is not None and "content" not in raw
        assert await env.stored_content(raw["content_ref"]) == body
        assert await env.read(script_name=name) == body
    assert len(await env.stored_refs()) == 3


async def test_migrate_keeps_a_concurrent_newer_edit_and_leaves_no_orphan(env: Env) -> None:
    await env.save("old inline", loader=env.inline_loader)

    class EditDuringMigration(KeyValueScriptContentStore):
        """Lets a newer edit land between the migration's copy to the store and its guarded update."""

        edited = False

        async def put(self, *, ref: str, content: str) -> None:
            await super().put(ref=ref, content=content)
            if not self.edited:
                self.edited = True
                await env.save("newer edit", loader=env.inline_loader)

    migrating = MongoPluginSkillLoader(
        database=env.database,
        script_content_store=EditDuringMigration(key_value=env.kv_store),
        write_scripts_to_content_store=False,
        auto_repair=False,
    )

    assert await migrating.migrate_inline_script_content() == 0  # the guarded update matched nothing

    raw = await env.raw()
    assert raw is not None and raw["content"] == "newer edit" and "content_ref" not in raw
    assert await env.read() == "newer edit"
    assert await env.stored_refs() == []  # the copy of the stale content was cleaned up

    assert await env.loader.migrate_inline_script_content() == 1  # a later pass migrates the newer edit
    raw = await env.raw()
    assert raw is not None and "content" not in raw
    assert await env.stored_content(raw["content_ref"]) == "newer edit"
    assert await env.stored_refs() == [raw["content_ref"]]
