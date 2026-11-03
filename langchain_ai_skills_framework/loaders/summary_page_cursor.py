"""Opaque keyset cursor for paged all-authors skill listings (BAI-1509).

The cursor encodes the last returned ``(plugin_name, skill_name, author)`` key, so a page resumes strictly after
that key. Because it is a position in the sort order and not an offset, concurrent inserts and deletes never
duplicate or skip a surviving row, and a cursor whose row has since been deleted still resumes correctly.
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass

DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 500
_CURSOR_VERSION = 1


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


def encode_cursor(*, key: SummaryPageKey) -> str:
    """Encode ``key`` as an opaque URL-safe string."""
    payload = json.dumps({"v": _CURSOR_VERSION, "k": [key.plugin_name, key.skill_name, key.author]})
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_cursor(*, cursor: str) -> SummaryPageKey:
    """Decode a cursor from ``encode_cursor``; raise ``InvalidPageCursorError`` for anything else."""
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
    except (binascii.Error, UnicodeError, ValueError, AttributeError) as exc:
        raise InvalidPageCursorError("cursor is not a valid page cursor") from exc
    if not isinstance(data, dict) or data.get("v") != _CURSOR_VERSION:
        raise InvalidPageCursorError("cursor has an unsupported version")
    parts = data.get("k")
    if not isinstance(parts, list) or len(parts) != 3 or not all(isinstance(p, str) for p in parts):
        raise InvalidPageCursorError("cursor has an invalid key")
    return SummaryPageKey(plugin_name=parts[0], skill_name=parts[1], author=parts[2])
