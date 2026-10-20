"""Storage for skill script content outside the ``plugin_scripts`` metadata documents.

A script's content can be stored in a separate key-value store so a single script can never push its metadata
document past MongoDB's 16 MB document limit. The metadata (identity, path, digest, size, timestamps) stays in
``plugin_scripts``; the document points at the content through its ``content_ref`` field.

``ScriptContentStore`` is the seam: it only needs ``put``/``get``/``delete`` by key, so it needs no enumeration or
prefix delete from the backing store. ``KeyValueScriptContentStore`` implements it over any py-key-value
``AsyncKeyValueProtocol`` store that the consuming service injects (for example one that keeps small values inline
and spills large ones to GridFS). The framework itself does not depend on any particular backend.

Refs are ``<script key>@<digest>#<save id>``. The digest part makes a ref self-describing, and the unique save id
means no two saves ever share a ref, so the save that retires a ref (by replacing the pointer to it) is the only one
that can delete it, and nobody can re-reference it afterwards. That is what makes deleting superseded content safe
without listing keys, grace windows or compare-and-delete support from the store.

Script history snapshots use the same store but their own refs (``<script key>@<digest>#history-<save id>``, see
``build_script_history_content_ref``). Each history record owns a private copy of the content, so retiring the live
script's ref (a newer save, or deleting the script) can never remove content a history record still points at.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol, runtime_checkable

from key_value.aio.protocols.key_value import AsyncKeyValueProtocol

from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

DEFAULT_SCRIPT_CONTENT_COLLECTION = "plugin_scripts_content"
_CONTENT_FIELD = "content"
HISTORY_REF_MARKER = "history-"

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


def build_script_content_key(*, author: str, plugin_name: str, skill_name: str, script_name: str) -> str:
    """Stable identity of one script: ``author/plugin/skill/script`` (no version)."""
    return f"{author}/{plugin_name}/{skill_name}/{script_name}"


def build_script_content_ref(*, script_key: str, digest: str) -> str:
    """New unique key for one save's content (the value stored in ``content_ref``).

    ``<script key>@<digest>#<save id>``: every call returns a different ref, even for identical content.
    """
    return f"{script_key}@{digest}#{uuid.uuid4().hex}"


def build_script_history_content_ref(*, script_key: str, digest: str) -> str:
    """New unique key for the content copy owned by one script history record.

    ``<script key>@<digest>#history-<save id>``: distinct from every live ref (``build_script_content_ref``), so the
    code that retires live content never touches it.
    """
    return f"{script_key}@{digest}#{HISTORY_REF_MARKER}{uuid.uuid4().hex}"


@runtime_checkable
class ScriptContentStore(Protocol):
    """Async store for script content addressed by opaque refs."""

    async def put(self, *, ref: str, content: str) -> None:
        """Store ``content`` under ``ref`` (replacing any value already there)."""
        ...

    async def get(self, *, ref: str) -> str | None:
        """Return the content stored under ``ref``, or ``None`` when there is none."""
        ...

    async def delete(self, *, ref: str) -> bool:
        """Remove the content under ``ref``; return whether there was any."""
        ...


class KeyValueScriptContentStore:
    """``ScriptContentStore`` over a py-key-value store; each ref is a key holding ``{"content": <text>}``.

    Values never expire (``ttl=None``): content is deleted explicitly by the loader.
    """

    def __init__(
        self,
        *,
        key_value: AsyncKeyValueProtocol,
        collection: str = DEFAULT_SCRIPT_CONTENT_COLLECTION,
    ) -> None:
        self._key_value = key_value
        self._collection = collection

    async def put(self, *, ref: str, content: str) -> None:
        await self._key_value.put(ref, {_CONTENT_FIELD: content}, collection=self._collection, ttl=None)

    async def get(self, *, ref: str) -> str | None:
        value = await self._key_value.get(ref, collection=self._collection)
        if value is None:
            return None
        content = value.get(_CONTENT_FIELD)
        if not isinstance(content, str):
            logger.warning("Script content under %s has no text %r field; treating it as missing", ref, _CONTENT_FIELD)
            return None
        return content

    async def delete(self, *, ref: str) -> bool:
        return await self._key_value.delete(ref, collection=self._collection)
