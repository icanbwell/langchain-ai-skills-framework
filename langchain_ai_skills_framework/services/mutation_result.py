from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MutationResult:
    """Outcome of a skill-mutation operation (save/delete).

    ``ok=True`` means the change was persisted. ``ok=False`` is a soft
    failure (validation rejected the change, or a deletion target wasn't
    found) that callers should surface as a client error — HTTP 400/404,
    not 200 — so downstream UI does not treat the operation as successful.
    """

    ok: bool
    message: str
