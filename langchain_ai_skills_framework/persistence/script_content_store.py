"""Storage for skill script content outside the ``plugin_scripts`` metadata documents.

Script content lives in GridFS so a single script can never push its metadata document past MongoDB's 16 MB
document limit. The metadata (identity, path, digest, size, timestamps) stays in ``plugin_scripts``.

Blobs are content-addressed: the key is ``<script key>@<digest>``, so a blob is never overwritten. The metadata
document points at its blob through ``content_ref``, which makes the pointer and the digest change atomically in
one upsert; stale blobs are garbage-collected only after that upsert succeeds.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

from gridfs.errors import NoFile
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket

DEFAULT_SCRIPT_CONTENT_BUCKET = "plugin_scripts_content"


def build_script_content_key(*, author: str, plugin_name: str, skill_name: str, script_name: str) -> str:
    """Stable identity of one script: ``author/plugin/skill/script`` (no version)."""
    return f"{author}/{plugin_name}/{skill_name}/{script_name}"


def build_script_content_ref(*, script_key: str, digest: str) -> str:
    """Content-addressed blob key (the value stored in ``content_ref``): ``<script key>@<digest>``."""
    return f"{script_key}@{digest}"


def build_skill_content_prefix(*, author: str, plugin_name: str, skill_name: str) -> str:
    """Key prefix shared by every script of one skill (the trailing ``/`` keeps other skills out)."""
    return f"{author}/{plugin_name}/{skill_name}/"


# ``sha256:<64 hex>``, as produced by ``content_digest_and_size``.
_DIGEST_SUFFIX = r"@sha256:[0-9a-f]{64}"


@runtime_checkable
class ScriptContentStore(Protocol):
    """Async blob store for script content, addressed by content-addressed refs."""

    async def put(self, *, ref: str, content: str, metadata: Mapping[str, object] | None = None) -> None:
        """Store ``content`` under ``ref``. Idempotent: a ref that already exists is left untouched."""
        ...

    async def get(self, *, ref: str) -> str | None:
        """Return the content stored under ``ref``, or ``None`` when there is none."""
        ...

    async def delete_versions(
        self, *, script_key: str, keep_ref: str | None = None, older_than: datetime | None = None
    ) -> int:
        """Remove blobs of the script ``script_key`` except ``keep_ref``; return how many were removed.

        ``older_than`` restricts removal to blobs uploaded before that instant, which protects a blob that a
        concurrent save has uploaded but whose metadata upsert has not landed yet.

        Only refs of exactly the form ``<script_key>@sha256:<hex>`` match, so a script whose name merely starts
        with ``script_key`` is never touched.
        """
        ...

    async def delete_prefix(self, *, prefix: str) -> int:
        """Remove every blob whose ref starts with ``prefix``; return how many were removed."""
        ...


class GridFSScriptContentStore:
    """``ScriptContentStore`` backed by a motor ``AsyncIOMotorGridFSBucket`` on the existing database."""

    def __init__(
        self,
        *,
        database: AsyncIOMotorDatabase[dict[str, object]],
        bucket_name: str = DEFAULT_SCRIPT_CONTENT_BUCKET,
    ) -> None:
        self._bucket = AsyncIOMotorGridFSBucket(database, bucket_name=bucket_name)

    async def put(self, *, ref: str, content: str, metadata: Mapping[str, object] | None = None) -> None:
        if await self._bucket.find({"filename": ref}).to_list(length=1):
            return  # content-addressed: same ref means same content
        await self._bucket.upload_from_stream(
            ref, content.encode("utf-8"), metadata=dict(metadata) if metadata else None
        )

    async def get(self, *, ref: str) -> str | None:
        try:
            stream = await self._bucket.open_download_stream_by_name(ref)
        except NoFile:
            return None
        data: bytes = await stream.read()
        return data.decode("utf-8")

    async def delete_versions(
        self, *, script_key: str, keep_ref: str | None = None, older_than: datetime | None = None
    ) -> int:
        filename: dict[str, object] = {"$regex": f"^{re.escape(script_key)}{_DIGEST_SUFFIX}$"}
        if keep_ref is not None:
            filename["$ne"] = keep_ref
        query: dict[str, object] = {"filename": filename}
        if older_than is not None:
            query["uploadDate"] = {"$lt": older_than}
        return await self._delete_files(query)

    async def delete_prefix(self, *, prefix: str) -> int:
        return await self._delete_files({"filename": {"$regex": f"^{re.escape(prefix)}"}})

    async def _delete_files(self, query: dict[str, object]) -> int:
        removed = 0
        async for file_doc in self._bucket.find(query):
            await self._bucket.delete(file_doc["_id"])
            removed += 1
        return removed
