"""Per-document schema migrations for the Mongo plugin-skill collections.

``MongoPluginSkillLoader`` filters every query by an exact ``schema_version``. When
``MongoPluginSkillDocument.SCHEMA_VERSION`` is bumped, documents stored under an older
version would otherwise become invisible. Marketplace documents are rewritten by
``SkillSync``; user-authored documents are not, so they are migrated in place here.

Each registered step upgrades a raw document from ``v`` to ``v + 1`` and returns the
fields to ``$set`` (it never sets ``schema_version`` itself). Bumping ``SCHEMA_VERSION``
without registering the matching step fails ``test_schema_migrations``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Literal

CollectionKind = Literal["skills", "resources", "scripts", "plugins", "usage"]

# Lowest schema_version that can still be migrated forward. Older documents are left alone.
MIN_MIGRATABLE_VERSION = 2

# Fields that, together with schema_version, make up each collection's unique index.
# ``usage`` has no unique index.
IDENTITY_FIELDS: Mapping[CollectionKind, tuple[str, ...]] = {
    "skills": ("author", "plugin_name", "skill_name"),
    "resources": ("author", "plugin_name", "skill_name", "resource_name"),
    "scripts": ("author", "plugin_name", "skill_name", "script_name"),
    "plugins": ("plugin_name",),
    "usage": (),
}

MigrationStep = Callable[[Mapping[str, object]], dict[str, object]]


class MissingMigrationError(RuntimeError):
    """No migration step is registered for a schema_version hop."""


def _unchanged(_doc: Mapping[str, object]) -> dict[str, object]:
    return {}


def content_digest_and_size(content: str) -> tuple[str, int]:
    """Return the ``sha256:<hex>`` digest and UTF-8 byte length of ``content``.

    The single definition used both when saving documents and when migrating them, so the two cannot drift.
    """
    encoded = content.encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(encoded)


def _digest_and_size(doc: Mapping[str, object]) -> dict[str, object]:
    """Compute the SHA-256 manifest fields v4 added, from the stored ``content``."""
    content = doc.get("content")
    if not isinstance(content, str):
        return {}
    digest, size = content_digest_and_size(content)
    return {"digest": digest, "size": size}


def _skill_v3_to_v4(doc: Mapping[str, object]) -> dict[str, object]:
    fields = _digest_and_size(doc)
    if "is_dynamic" not in doc:
        fields["is_dynamic"] = False
    return fields


# {from_version: {collection_kind: step}} — each step moves a document one version forward.
_STEPS: Mapping[int, Mapping[CollectionKind, MigrationStep]] = {
    # v2 -> v3 added ``required_external_servers`` (defaulted by the model); nothing to rewrite.
    2: {
        "skills": _unchanged,
        "resources": _unchanged,
        "scripts": _unchanged,
        "plugins": _unchanged,
        "usage": _unchanged,
    },
    # v3 -> v4 added digest/size/is_dynamic; without them a migrated skill has no manifest.
    3: {
        "skills": _skill_v3_to_v4,
        "resources": _digest_and_size,
        "scripts": _digest_and_size,
        "plugins": _unchanged,
        "usage": _unchanged,
    },
}


def registered_versions() -> frozenset[int]:
    """Versions that have a step moving them to ``version + 1``."""
    return frozenset(_STEPS)


def migrate_fields(
    *,
    kind: CollectionKind,
    doc: Mapping[str, object],
    from_version: int,
    to_version: int,
) -> dict[str, object]:
    """Return the combined ``$set`` fields that upgrade ``doc`` from ``from_version`` to ``to_version``.

    Raises :class:`MissingMigrationError` if any hop in between has no registered step.
    """
    fields: dict[str, object] = {}
    working: dict[str, object] = dict(doc)
    for version in range(from_version, to_version):
        step = _STEPS.get(version, {}).get(kind)
        if step is None:
            raise MissingMigrationError(f"No migration registered for {kind} schema_version {version} -> {version + 1}")
        step_fields = step(working)
        working.update(step_fields)
        fields.update(step_fields)
    return fields
