"""Opaque keyset cursor for paged all-authors skill listings (BAI-1509).

The cursor encodes the last returned ``(plugin_name, skill_name, author)`` key plus a short fingerprint of the
filters it was issued under, so a page resumes strictly after that key and a cursor reused with different filters
is rejected rather than silently skipping or repeating rows. Because it is a position in the sort order and not an offset, concurrent inserts and deletes never
duplicate or skip a surviving row, and a cursor whose row has since been deleted still resumes correctly.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass

DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 500
_CURSOR_VERSION = 1
# A real cursor is a few hundred bytes; reject anything larger before base64/JSON decoding it.
MAX_CURSOR_LENGTH = 2048


class InvalidPageCursorError(ValueError):
    """The cursor is malformed, tampered with, or from an incompatible version."""


@dataclass(frozen=True, slots=True)
class SummaryPageKey:
    """Sort key of one skill: ``(plugin_name, skill_name, author)``."""

    plugin_name: str
    skill_name: str
    author: str


def validate_page_limit(*, limit: int) -> int:
    """Return ``limit`` clamped to ``MAX_PAGE_LIMIT``; raise ``ValueError`` if it is below 1."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit!r}")
    return min(limit, MAX_PAGE_LIMIT)


def filters_fingerprint(
    *, include_staging: bool, plugin_name: str | None, folder: str | None, allow_states: Collection[str] | None
) -> str:
    """Short, order-insensitive hash of the filters a page walk runs under.

    Raises:
        TypeError: If ``allow_states`` is a bare string, which would be iterated as single characters.
    """
    if isinstance(allow_states, str):
        raise TypeError("allow_states must be a collection of state names, not a single string")
    normalised = json.dumps(
        [include_staging, plugin_name, folder, None if allow_states is None else sorted(set(allow_states))]
    )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def encode_cursor(*, key: SummaryPageKey, fingerprint: str = "") -> str:
    """Encode ``key`` and the filter ``fingerprint`` as an opaque URL-safe string."""
    payload = json.dumps({"v": _CURSOR_VERSION, "k": [key.plugin_name, key.skill_name, key.author], "f": fingerprint})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_cursor(*, cursor: str, fingerprint: str = "") -> SummaryPageKey:
    """Decode a cursor from ``encode_cursor``.

    Raises ``InvalidPageCursorError`` for anything else, including a cursor issued under different filters
    (``fingerprint`` mismatch).
    """
    if not isinstance(cursor, str) or len(cursor) > MAX_CURSOR_LENGTH:
        raise InvalidPageCursorError("cursor is not a valid page cursor")
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
    except (binascii.Error, UnicodeError, ValueError, AttributeError, RecursionError) as exc:
        raise InvalidPageCursorError("cursor is not a valid page cursor") from exc
    if not isinstance(data, dict) or data.get("v") != _CURSOR_VERSION:
        raise InvalidPageCursorError("cursor has an unsupported version")
    parts = data.get("k")
    if not isinstance(parts, list) or len(parts) != 3 or not all(isinstance(p, str) for p in parts):
        raise InvalidPageCursorError("cursor has an invalid key")
    if data.get("f") != fingerprint:
        raise InvalidPageCursorError("cursor was issued for different filters")
    return SummaryPageKey(plugin_name=parts[0], skill_name=parts[1], author=parts[2])
