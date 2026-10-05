"""MongoDB-backed implementation of :class:`PluginSkillStore`.

Uses three collections (``plugin_skills``, ``plugin_references``,
``plugin_scripts``) with the **Materialized Paths** tree-structure pattern.
Every document carries a ``path`` field that mirrors the on-disk plugin
directory layout, enabling tree-style queries.

Replaces the legacy ``MongoUserSkillLoader``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Concatenate, Literal

import yaml
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError, OperationFailure

from langchain_ai_skills_framework.loaders.exceptions.skill_loader_error import (
    SkillLoaderError,
)
from langchain_ai_skills_framework.loaders.exceptions.skill_not_found_error import (
    SkillNotFoundError,
)
from langchain_ai_skills_framework.loaders.schema_migrations import (
    IDENTITY_FIELDS,
    MIN_MIGRATABLE_VERSION,
    CollectionKind,
    MissingMigrationError,
    content_digest_and_size,
    migrate_fields,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
    MongoPluginSkillUsageDocument,
    build_resource_path,
    build_script_path,
    build_skill_path,
    normalize_folder,
)
from langchain_ai_skills_framework.models.skills_model import (
    ManifestFileEntry,
    SkillDetails,
    SkillSnapshot,
    SkillSummary,
)
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Default collection names — overridable via environment variables.
DEFAULT_SKILLS_COLLECTION = "plugin_skills"
DEFAULT_REFERENCES_COLLECTION = "plugin_references"
DEFAULT_SCRIPTS_COLLECTION = "plugin_scripts"
DEFAULT_USAGE_COLLECTION = "plugin_skill_usage"
DEFAULT_PLUGINS_COLLECTION = "plugins"

# How often a loader re-probes for documents left at an older schema_version (for example, written by a
# not-yet-upgraded pod during a rolling deploy). The probe is a single indexed ``count_documents`` per collection.
DEFAULT_REPAIR_INTERVAL_SECONDS = 300.0
DEFAULT_REPAIR_BATCH_SIZE = 500
UNAUTHORIZED_ERROR_CODE = 13  # MongoDB "Unauthorized"


def _repairs_first[**P, R](
    method: Callable[Concatenate[MongoPluginSkillLoader, P], Awaitable[R]],
) -> Callable[Concatenate[MongoPluginSkillLoader, P], Coroutine[Any, Any, R]]:
    """Run the throttled stale-document repair before a public data-access coroutine.

    Applied explicitly to every public coroutine except ``ensure_indexes`` and ``repair_stale_documents``, so any
    caller (including consumers that never call ``ensure_indexes``) sees user-authored documents saved under an
    older ``schema_version``. ``test_every_public_coroutine_repairs_first`` fails if a new one is left undecorated.
    """

    async def wrapper(self: MongoPluginSkillLoader, /, *args: P.args, **kwargs: P.kwargs) -> R:
        await self._ensure_current()
        return await method(self, *args, **kwargs)

    functools.update_wrapper(wrapper, method)
    wrapper.__repairs_first__ = True  # type: ignore[attr-defined]
    return wrapper


class MongoPluginSkillLoader:
    """Reads and writes plugin-scoped skills in MongoDB.

    This is a **singleton** service — ``author`` and ``plugin_name`` are
    provided on each call, matching the gateway pattern where tools
    receive identity as a tool-input parameter.
    """

    INDEX_NAME = "ux_plugin_skill"
    RESOURCE_INDEX_NAME = "ux_plugin_skill_resource"
    SCRIPT_INDEX_NAME = "ux_plugin_skill_script"
    PATH_INDEX_NAME = "ix_plugin_path"
    USAGE_INDEX_NAME = "ix_plugin_skill_usage_lookup"
    USAGE_SCHEMA_VERSION_INDEX_NAME = "ix_plugin_skill_usage_schema_version"
    PLUGIN_INDEX_NAME = "ux_plugin_name"

    SCHEMA_VERSION_FIELD = "schema_version"

    def __init__(
        self,
        *,
        database: AsyncIOMotorDatabase[dict[str, object]],
        schema_version: int = MongoPluginSkillDocument.SCHEMA_VERSION,
        skills_collection_name: str = DEFAULT_SKILLS_COLLECTION,
        references_collection_name: str = DEFAULT_REFERENCES_COLLECTION,
        scripts_collection_name: str = DEFAULT_SCRIPTS_COLLECTION,
        usage_collection_name: str = DEFAULT_USAGE_COLLECTION,
        plugins_collection_name: str = DEFAULT_PLUGINS_COLLECTION,
        auto_repair: bool = True,
        repair_interval_seconds: float = DEFAULT_REPAIR_INTERVAL_SECONDS,
        repair_batch_size: int = DEFAULT_REPAIR_BATCH_SIZE,
    ) -> None:
        self._database = database
        self._schema_version = schema_version
        self._auto_repair = auto_repair
        self._repair_interval_seconds = repair_interval_seconds
        self._repair_lock = asyncio.Lock()
        self._repair_batch_size = repair_batch_size
        self._last_repair_check: float | None = None
        self._repair_incomplete = False
        self._skipped_ids: set[object] = set()
        self._skills_collection: AsyncIOMotorCollection[dict[str, object]] = database[skills_collection_name]
        self._resources_collection: AsyncIOMotorCollection[dict[str, object]] = database[references_collection_name]
        self._scripts_collection: AsyncIOMotorCollection[dict[str, object]] = database[scripts_collection_name]
        self._usage_collection: AsyncIOMotorCollection[dict[str, object]] = database[usage_collection_name]
        self._plugins_collection: AsyncIOMotorCollection[dict[str, object]] = database[plugins_collection_name]

    # --- Index management ---------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create compound unique indexes and Materialized Paths index.

        Handles migration from older index schemas by dropping indexes whose
        key spec no longer matches the expected definition.
        """
        sv = self.SCHEMA_VERSION_FIELD
        await self._ensure_index(
            self._skills_collection,
            keys=[(sv, 1), ("author", 1), ("plugin_name", 1), ("skill_name", 1)],
            unique=True,
            name=self.INDEX_NAME,
        )
        await self._ensure_index(
            self._resources_collection,
            keys=[(sv, 1), ("author", 1), ("plugin_name", 1), ("skill_name", 1), ("resource_name", 1)],
            unique=True,
            name=self.RESOURCE_INDEX_NAME,
        )
        await self._ensure_index(
            self._scripts_collection,
            keys=[(sv, 1), ("author", 1), ("plugin_name", 1), ("skill_name", 1), ("script_name", 1)],
            unique=True,
            name=self.SCRIPT_INDEX_NAME,
        )
        await self._ensure_index(
            self._skills_collection,
            keys=[(sv, 1), ("plugin_name", 1), ("path", 1)],
            unique=False,
            name=self.PATH_INDEX_NAME,
        )
        await self._ensure_index(
            self._usage_collection,
            keys=[("skill_name", 1), ("author", 1), ("date_used", -1)],
            unique=False,
            name=self.USAGE_INDEX_NAME,
        )
        # Lets the periodic stale-document probe use an index instead of scanning the growing usage collection.
        await self._ensure_index(
            self._usage_collection,
            keys=[(sv, 1)],
            unique=False,
            name=self.USAGE_SCHEMA_VERSION_INDEX_NAME,
        )
        await self._ensure_index(
            self._plugins_collection,
            keys=[(sv, 1), ("plugin_name", 1)],
            unique=True,
            name=self.PLUGIN_INDEX_NAME,
        )

    @staticmethod
    async def _ensure_index(
        collection: AsyncIOMotorCollection[dict[str, object]],
        *,
        keys: list[tuple[str, int]],
        unique: bool,
        name: str,
    ) -> None:
        """Create an index, dropping the old one first if its key spec conflicts."""
        try:
            await collection.create_index(keys, unique=unique, name=name)
        except OperationFailure as exc:
            if exc.code == 86:
                logger.warning(
                    "Index '%s' on %s has conflicting key spec — dropping and recreating.",
                    name,
                    collection.name,
                )
                await collection.drop_index(name)
                await collection.create_index(keys, unique=unique, name=name)
            else:
                raise

    # --- Schema-version repair ------------------------------------------------

    async def _ensure_current(self) -> None:
        """Re-tag documents stored under an older ``schema_version`` so reads can see them.

        Runs on first use and again at most every ``repair_interval_seconds``. Each pass handles at most
        ``repair_batch_size`` documents per collection, so a large backlog is drained across successive calls
        instead of blocking one caller. Failures are logged and never propagate: a repair problem must not take
        reads down.

        Rolling deploys: a document migrated to the new ``schema_version`` is invisible to pods still running the
        old version until they are upgraded; the periodic re-check exists to pick up documents those old pods
        write meanwhile. Repair writes to the collections, so a loader using read-only credentials disables itself
        after the first authorization failure (see ``auto_repair``).
        """
        if not self._auto_repair:
            return
        if self._last_repair_check is not None and (
            time.monotonic() - self._last_repair_check < self._repair_interval_seconds
        ):
            return
        async with self._repair_lock:
            if self._last_repair_check is not None and (
                time.monotonic() - self._last_repair_check < self._repair_interval_seconds
            ):
                return
            try:
                await self.repair_stale_documents(batch_size=self._repair_batch_size)
            except OperationFailure as exc:
                self._repair_incomplete = False
                if exc.code == UNAUTHORIZED_ERROR_CODE:
                    self._auto_repair = False
                    logger.warning(
                        "Schema-version repair disabled: the database user cannot write. Documents saved under an "
                        "older schema_version stay invisible to this loader until repaired by a writer (or "
                        "repair_stale_documents is run with write access): %s",
                        exc,
                    )
                else:
                    logger.exception(
                        "Schema-version repair failed; stale documents may remain invisible until it succeeds"
                    )
            except Exception:
                self._repair_incomplete = False
                logger.exception("Schema-version repair failed; stale documents may remain invisible until it succeeds")
            finally:
                # A truncated pass leaves the timestamp unset so the next call continues draining the backlog.
                self._last_repair_check = None if self._repair_incomplete else time.monotonic()

    async def repair_stale_documents(self, *, batch_size: int | None = None) -> Mapping[str, int]:
        """Migrate every document with ``MIN_MIGRATABLE_VERSION <= schema_version < current`` in place.

        Safe to run concurrently from several pods: each update is conditional on the document still being at the
        old version, so a race is a no-op for the loser. A document is left untouched (and logged at ERROR) when a
        document with the same identity already exists at the current version, because that one holds the user's
        latest save and the unique index would reject the re-tag anyway.

        ``batch_size`` caps the documents examined per collection in this call (``None`` = unbounded); when the cap
        is hit, the remainder is picked up by the next automatic pass.

        Returns counts keyed ``"<collection>.migrated"`` / ``"<collection>.conflicts"``.
        """
        self._repair_incomplete = False
        targets: tuple[tuple[CollectionKind, AsyncIOMotorCollection[dict[str, object]]], ...] = (
            ("skills", self._skills_collection),
            ("resources", self._resources_collection),
            ("scripts", self._scripts_collection),
            ("plugins", self._plugins_collection),
            ("usage", self._usage_collection),
        )
        counts: dict[str, int] = {}
        for kind, collection in targets:
            migrated, conflicts = await self._repair_collection(kind=kind, collection=collection, batch_size=batch_size)
            counts[f"{collection.name}.migrated"] = migrated
            counts[f"{collection.name}.conflicts"] = conflicts
        return counts

    async def _repair_collection(
        self,
        *,
        kind: CollectionKind,
        collection: AsyncIOMotorCollection[dict[str, object]],
        batch_size: int | None,
    ) -> tuple[int, int]:
        sv = self.SCHEMA_VERSION_FIELD
        stale_filter: dict[str, object] = {sv: {"$gte": MIN_MIGRATABLE_VERSION, "$lt": self._schema_version}}
        if self._skipped_ids:
            # Known unresolvable documents must not occupy batch slots forever.
            stale_filter["_id"] = {"$nin": list(self._skipped_ids)}
        if await collection.count_documents(stale_filter, limit=1) == 0:
            return 0, 0

        identity_fields = IDENTITY_FIELDS[kind]
        migrated = 0
        conflicts = 0
        examined = 0
        cursor = collection.find(stale_filter)
        if batch_size is not None:
            cursor = cursor.limit(batch_size)
        async for doc in cursor:
            examined += 1
            # Digest computation is CPU-bound; yield so a large batch cannot starve other tasks on the event loop.
            await asyncio.sleep(0)
            old_version = doc.get(sv)
            if not isinstance(old_version, int):
                self._skip(collection, doc, f"schema_version {old_version!r} is not an integer")
                continue
            if identity_fields:
                sibling = await collection.find_one(
                    {sv: self._schema_version, **{f: doc.get(f) for f in identity_fields}}, {"_id": 1}
                )
                if sibling is not None:
                    conflicts += 1
                    self._skip(
                        collection,
                        doc,
                        f"a document with the same identity already exists at schema_version {self._schema_version} "
                        f"(_id={sibling.get('_id')}). Resolve manually",
                    )
                    continue
            try:
                fields = migrate_fields(kind=kind, doc=doc, from_version=old_version, to_version=self._schema_version)
            except MissingMigrationError as exc:
                self._skip(collection, doc, str(exc))
                continue
            try:
                result = await collection.update_one(
                    {"_id": doc["_id"], sv: old_version},
                    {"$set": {**fields, sv: self._schema_version}},
                )
            except DuplicateKeyError:
                conflicts += 1
                self._skip(
                    collection,
                    doc,
                    f"a same-identity document was created at schema_version {self._schema_version} concurrently",
                )
                continue
            migrated += result.modified_count
        if batch_size is not None and examined >= batch_size:
            self._repair_incomplete = True
        if migrated or conflicts:
            logger.info(
                "Schema-version repair on %s: migrated=%d conflicts=%d (target schema_version=%s)",
                collection.name,
                migrated,
                conflicts,
                self._schema_version,
            )
        return migrated, conflicts

    def _skip(
        self, collection: AsyncIOMotorCollection[dict[str, object]], doc: Mapping[str, object], reason: str
    ) -> None:
        """Log a document that cannot be migrated, once per process, and exclude it from later passes.

        Excluding it keeps a permanently unmigratable document from repeating its log line every interval and from
        occupying a batch slot forever.
        """
        self._skipped_ids.add(doc["_id"])
        logger.error("Not migrating %s _id=%s: %s.", collection.name, doc.get("_id"), reason)

    def _version_filter(self, query: dict[str, object]) -> dict[str, object]:
        """Add schema_version to a query filter."""
        return {**query, self.SCHEMA_VERSION_FIELD: self._schema_version}

    # --- Skill write operations ----------------------------------------------

    @_repairs_first
    async def save_skill(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        content: str,
        modified_by: str = "",
        folder: str | None = None,
        path: str | None = None,
        state: str | None = None,
        is_dynamic: bool = False,
    ) -> MongoPluginSkillDocument:
        self._validate_author(author)
        normalized_name = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")

        folder = normalize_folder(folder)

        if folder is None:
            existing = await self._skills_collection.find_one(
                self._version_filter({"author": author, "plugin_name": plugin_name, "skill_name": normalized_name}),
                {"folder": 1},
            )
            if existing:
                raw_folder = existing.get("folder")
                folder = normalize_folder(raw_folder if isinstance(raw_folder, str) else None)

        description = self._extract_description(content)
        path = (
            path.strip()
            if isinstance(path, str) and path.strip()
            else build_skill_path(plugin_name=plugin_name, skill_name=normalized_name, folder=folder)
        )
        now = datetime.now(UTC)
        effective_modified_by = modified_by or author
        sv = self.SCHEMA_VERSION_FIELD

        digest, size = self._digest_and_size(content)
        set_fields: dict[str, object] = {
            "content": content,
            "description": description,
            "path": path,
            "folder": folder,
            "modified_by": effective_modified_by,
            "date_modified": now,
            "digest": digest,
            "size": size,
            "is_dynamic": is_dynamic,
        }

        if state is not None:
            set_fields["state"] = state

        set_on_insert: dict[str, object] = {
            "author": author,
            "plugin_name": plugin_name,
            "skill_name": normalized_name,
            sv: self._schema_version,
            "date_created": now,
        }
        if state is None:
            set_on_insert["state"] = "draft"

        raw = await self._skills_collection.find_one_and_update(
            self._version_filter({"author": author, "plugin_name": plugin_name, "skill_name": normalized_name}),
            {
                "$set": set_fields,
                "$setOnInsert": set_on_insert,
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

        return MongoPluginSkillDocument.from_mongo_dict(raw)

    @_repairs_first
    async def set_skill_state(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        state: str,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument:
        self._validate_author(author)
        normalized_name = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")

        now = datetime.now(UTC)
        update_fields: dict[str, object] = {
            "state": state,
            "date_modified": now,
        }
        if state == "published":
            update_fields["published_date"] = now
        if published_branch is not None:
            update_fields["published_branch"] = published_branch
        raw = await self._skills_collection.find_one_and_update(
            self._version_filter({"author": author, "plugin_name": plugin_name, "skill_name": normalized_name}),
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        if raw is None:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in plugin '{plugin_name}' for author '{author}'")
        return MongoPluginSkillDocument.from_mongo_dict(raw)

    @_repairs_first
    async def delete_skill(self, *, author: str, plugin_name: str, skill_name: str) -> bool:
        self._validate_author(author)
        normalized_name = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")

        filter_base = self._version_filter(
            {"author": author, "plugin_name": plugin_name, "skill_name": normalized_name}
        )
        await self._resources_collection.delete_many(filter_base)
        await self._scripts_collection.delete_many(filter_base)

        result = await self._skills_collection.delete_one(filter_base)
        return result.deleted_count > 0

    @_repairs_first
    async def skill_exists(self, *, author: str, plugin_name: str | None = None, skill_name: str) -> bool:
        self._validate_author(author)
        normalized_name = self._normalize(skill_name)
        query: dict[str, object] = {"author": author, "skill_name": normalized_name}
        if plugin_name:
            query["plugin_name"] = plugin_name
        count = await self._skills_collection.count_documents(self._version_filter(query), limit=1)
        return count > 0

    # --- Resource write operations -------------------------------------------

    @_repairs_first
    async def save_resource(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        content: str,
        modified_by: str = "",
        folder: str | None = None,
        path: str | None = None,
    ) -> MongoPluginResourceDocument:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")
        self._validate_not_empty(resource_name.strip(), "resource_name")

        path = path.strip() if isinstance(path, str) and path.strip() else ""
        if not path:
            folder = await self._ensure_skill_folder(
                author=author, plugin_name=plugin_name, skill_name=normalized_skill, folder=folder
            )
            path = build_resource_path(
                plugin_name=plugin_name,
                skill_name=normalized_skill,
                resource_name=resource_name.strip(),
                folder=folder,
            )
        now = datetime.now(UTC)
        effective_modified_by = modified_by or author
        sv = self.SCHEMA_VERSION_FIELD

        raw = await self._resources_collection.find_one_and_update(
            self._version_filter(
                {
                    "author": author,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                }
            ),
            {
                "$set": {
                    "content": content,
                    "path": path,
                    "modified_by": effective_modified_by,
                    "date_modified": now,
                    **dict(zip(("digest", "size"), self._digest_and_size(content), strict=True)),
                },
                "$setOnInsert": {
                    "author": author,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MongoPluginResourceDocument.from_mongo_dict(raw)

    @_repairs_first
    async def delete_resource(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        result = await self._resources_collection.delete_one(
            self._version_filter(
                {
                    "author": author,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "resource_name": resource_name.strip(),
                }
            )
        )
        return result.deleted_count > 0

    @_repairs_first
    async def read_resource(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> str:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "author": author,
            "skill_name": normalized_skill,
            "resource_name": resource_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        raw = await self._resources_collection.find_one(self._version_filter(query))
        if raw is None:
            raise SkillNotFoundError(
                f"Resource '{resource_name}' not found in skill '{skill_name}' "
                f"of plugin '{plugin_name}' for author '{author}'"
            )
        return str(raw["content"])

    @_repairs_first
    async def list_resource_names(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {"author": author, "skill_name": normalized_skill}
        if plugin_name:
            query["plugin_name"] = plugin_name
        names: list[str] = []
        async for raw in self._resources_collection.find(self._version_filter(query), {"resource_name": 1}):
            names.append(raw["resource_name"])
        return sorted(names)

    @_repairs_first
    async def list_resource_documents(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[MongoPluginResourceDocument]:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {"author": author, "skill_name": normalized_skill}
        if plugin_name:
            query["plugin_name"] = plugin_name
        docs: list[MongoPluginResourceDocument] = []
        async for raw in self._resources_collection.find(self._version_filter(query)):
            docs.append(MongoPluginResourceDocument.from_mongo_dict(raw))
        return sorted(docs, key=lambda d: d.resource_name)

    @_repairs_first
    async def resource_exists(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "author": author,
            "skill_name": normalized_skill,
            "resource_name": resource_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        count = await self._resources_collection.count_documents(self._version_filter(query), limit=1)
        return count > 0

    # --- Script write operations ---------------------------------------------

    @_repairs_first
    async def save_script(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        content: str,
        modified_by: str = "",
        folder: str | None = None,
        path: str | None = None,
    ) -> MongoPluginScriptDocument:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        self._validate_not_empty(plugin_name, "plugin_name")
        self._validate_not_empty(script_name.strip(), "script_name")

        path = path.strip() if isinstance(path, str) and path.strip() else ""
        if not path:
            folder = await self._ensure_skill_folder(
                author=author, plugin_name=plugin_name, skill_name=normalized_skill, folder=folder
            )
            path = build_script_path(
                plugin_name=plugin_name,
                skill_name=normalized_skill,
                script_name=script_name.strip(),
                folder=folder,
            )
        now = datetime.now(UTC)
        effective_modified_by = modified_by or author
        sv = self.SCHEMA_VERSION_FIELD

        raw = await self._scripts_collection.find_one_and_update(
            self._version_filter(
                {
                    "author": author,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                }
            ),
            {
                "$set": {
                    "content": content,
                    "path": path,
                    "modified_by": effective_modified_by,
                    "date_modified": now,
                    **dict(zip(("digest", "size"), self._digest_and_size(content), strict=True)),
                },
                "$setOnInsert": {
                    "author": author,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return MongoPluginScriptDocument.from_mongo_dict(raw)

    @_repairs_first
    async def delete_script(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        result = await self._scripts_collection.delete_one(
            self._version_filter(
                {
                    "author": author,
                    "plugin_name": plugin_name,
                    "skill_name": normalized_skill,
                    "script_name": script_name.strip(),
                }
            )
        )
        return result.deleted_count > 0

    @_repairs_first
    async def read_script(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> str:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "author": author,
            "skill_name": normalized_skill,
            "script_name": script_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        raw = await self._scripts_collection.find_one(self._version_filter(query))
        if raw is None:
            raise SkillNotFoundError(
                f"Script '{script_name}' not found in skill '{skill_name}' "
                f"of plugin '{plugin_name}' for author '{author}'"
            )
        return str(raw["content"])

    @_repairs_first
    async def list_script_names(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {"author": author, "skill_name": normalized_skill}
        if plugin_name:
            query["plugin_name"] = plugin_name
        names: list[str] = []
        async for raw in self._scripts_collection.find(self._version_filter(query), {"script_name": 1}):
            names.append(raw["script_name"])
        return sorted(names)

    @_repairs_first
    async def list_script_documents(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[MongoPluginScriptDocument]:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {"author": author, "skill_name": normalized_skill}
        if plugin_name:
            query["plugin_name"] = plugin_name
        docs: list[MongoPluginScriptDocument] = []
        async for raw in self._scripts_collection.find(self._version_filter(query)):
            docs.append(MongoPluginScriptDocument.from_mongo_dict(raw))
        return sorted(docs, key=lambda d: d.script_name)

    @_repairs_first
    async def script_exists(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> bool:
        self._validate_author(author)
        normalized_skill = self._normalize(skill_name)
        query: dict[str, object] = {
            "author": author,
            "skill_name": normalized_skill,
            "script_name": script_name.strip(),
        }
        if plugin_name:
            query["plugin_name"] = plugin_name
        count = await self._scripts_collection.count_documents(self._version_filter(query), limit=1)
        return count > 0

    # --- Skill read operations -----------------------------------------------

    @_repairs_first
    async def load_snapshot(
        self, *, author: str, plugin_name: str | None = None, include_staging: bool = False
    ) -> SkillSnapshot:
        self._validate_author(author)
        query: dict[str, object] = {"author": author}
        if plugin_name:
            query["plugin_name"] = plugin_name
        if not include_staging:
            query["state"] = {"$ne": "staging"}
        return await self._build_snapshot(query=self._version_filter(query), owner_label=author)

    @_repairs_first
    async def load_shared_snapshot(
        self, *, plugin_name: str | None = None, include_staging: bool = False
    ) -> SkillSnapshot:
        states = ["published", "staging", "in_review"] if include_staging else ["published"]
        query: dict[str, object] = {"state": {"$in": states}}
        if plugin_name:
            query["plugin_name"] = plugin_name
        return await self._build_snapshot(query=self._version_filter(query), owner_label="shared")

    @_repairs_first
    async def get_skill_details(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> SkillDetails:
        self._validate_author(author)
        normalized_name = self._normalize(skill_name)
        query: dict[str, object] = {"author": author, "skill_name": normalized_name}
        if plugin_name:
            query["plugin_name"] = plugin_name
        raw = await self._skills_collection.find_one(self._version_filter(query))
        if raw is None:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in plugin '{plugin_name}' for author '{author}'")
        doc = MongoPluginSkillDocument.from_mongo_dict(raw)
        resources = await self.list_resource_documents(
            author=author, plugin_name=doc.plugin_name, skill_name=normalized_name
        )
        scripts = await self.list_script_documents(
            author=author, plugin_name=doc.plugin_name, skill_name=normalized_name
        )
        manifest = self._build_manifest(doc=doc, resources=resources, scripts=scripts)
        summary = SkillSummary(
            name=doc.skill_name,
            description=doc.description,
            plugin_name=doc.plugin_name,
            folder=doc.folder,
            path=doc.path,
            state=doc.state,
            source_path=Path(f"mongodb://{author}/{doc.plugin_name}/{doc.skill_name}"),
            license=None,
            compatibility=None,
            metadata={"source": "mongodb", "user_id": doc.author, "plugin_name": doc.plugin_name},
            allowed_tools=doc.allowed_tools,
            date_modified=doc.date_modified,
            required_external_servers=doc.required_external_servers,
            manifest=manifest,
        )
        return SkillDetails(
            summary=summary,
            content=doc.content,
            source_path=summary.source_path,
        )

    # --- Usage tracking -------------------------------------------------------

    @_repairs_first
    async def record_skill_usage(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> MongoPluginSkillUsageDocument:
        doc = MongoPluginSkillUsageDocument(
            plugin_name=plugin_name,
            skill_name=skill_name,
            author=user_id,
        )
        data = doc.to_mongo_dict()
        data[self.SCHEMA_VERSION_FIELD] = self._schema_version
        await self._usage_collection.insert_one(data)
        return doc

    @_repairs_first
    async def get_skill_usage_count(self, *, skill_name: str) -> int:
        return int(await self._usage_collection.count_documents(self._version_filter({"skill_name": skill_name})))

    @_repairs_first
    async def get_skill_usage_counts(self, *, skill_names: Sequence[str]) -> Mapping[str, int]:
        if not skill_names:
            return {}
        pipeline: list[dict[str, Any]] = [
            {"$match": {self.SCHEMA_VERSION_FIELD: self._schema_version, "skill_name": {"$in": list(skill_names)}}},
            {"$group": {"_id": "$skill_name", "count": {"$sum": 1}}},
        ]
        counts: dict[str, int] = {name: 0 for name in skill_names}
        async for doc in self._usage_collection.aggregate(pipeline):
            counts[doc["_id"]] = int(doc["count"])
        return counts

    # --- Snapshot builder ----------------------------------------------------

    async def _build_snapshot(self, *, query: dict[str, object], owner_label: str) -> SkillSnapshot:
        docs: list[MongoPluginSkillDocument] = []
        async for raw in self._skills_collection.find(query):
            docs.append(MongoPluginSkillDocument.from_mongo_dict(raw))

        authors = sorted({doc.author for doc in docs})
        skill_names = sorted({doc.skill_name for doc in docs})

        resources_by_key: dict[tuple[str, str, str], list[MongoPluginResourceDocument]] = defaultdict(list)
        async for raw in self._resources_collection.find(
            self._version_filter({"author": {"$in": authors}, "skill_name": {"$in": skill_names}})
        ):
            rdoc = MongoPluginResourceDocument.from_mongo_dict(raw)
            resources_by_key[(rdoc.author, rdoc.plugin_name, rdoc.skill_name)].append(rdoc)

        scripts_by_key: dict[tuple[str, str, str], list[MongoPluginScriptDocument]] = defaultdict(list)
        async for raw in self._scripts_collection.find(
            self._version_filter({"author": {"$in": authors}, "skill_name": {"$in": skill_names}})
        ):
            sdoc = MongoPluginScriptDocument.from_mongo_dict(raw)
            scripts_by_key[(sdoc.author, sdoc.plugin_name, sdoc.skill_name)].append(sdoc)

        details_map: dict[str, SkillDetails] = {}
        summaries: list[SkillSummary] = []

        for doc in docs:
            manifest = self._build_manifest(
                doc=doc,
                resources=resources_by_key.get((doc.author, doc.plugin_name, doc.skill_name), []),
                scripts=scripts_by_key.get((doc.author, doc.plugin_name, doc.skill_name), []),
            )
            summary = SkillSummary(
                name=doc.skill_name,
                description=doc.description,
                plugin_name=doc.plugin_name,
                folder=doc.folder,
                path=doc.path,
                state=doc.state,
                source_path=Path(f"mongodb://{owner_label}/{doc.plugin_name}/{doc.skill_name}"),
                license=None,
                compatibility=None,
                metadata={"source": "mongodb", "user_id": doc.author, "plugin_name": doc.plugin_name},
                allowed_tools=doc.allowed_tools,
                date_modified=doc.date_modified,
                required_external_servers=doc.required_external_servers,
                manifest=manifest,
            )
            detail = SkillDetails(
                summary=summary,
                content=doc.content,
                source_path=summary.source_path,
            )
            details_map[doc.skill_name] = detail
            summaries.append(summary)

        ordered = tuple(sorted(summaries, key=lambda s: s.name))
        return SkillSnapshot(
            details_by_name=MappingProxyType(details_map),
            ordered_summaries=ordered,
        )

    # --- Helpers -------------------------------------------------------------

    async def _resolve_skill_folder(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
    ) -> str | None:
        """Look up the parent skill's stored folder value."""
        doc = await self._skills_collection.find_one(
            self._version_filter({"author": author, "plugin_name": plugin_name, "skill_name": skill_name}),
            {"folder": 1},
        )
        if doc:
            raw_folder = doc.get("folder")
            return normalize_folder(raw_folder if isinstance(raw_folder, str) else None)
        return None

    async def _ensure_skill_folder(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        folder: str | None,
    ) -> str | None:
        """Normalize ``folder``; fall back to the parent skill's stored folder when empty."""
        resolved = normalize_folder(folder)
        if resolved is not None:
            return resolved
        return await self._resolve_skill_folder(author=author, plugin_name=plugin_name, skill_name=skill_name)

    @staticmethod
    def _normalize(value: str) -> str:
        return normalize_skill_name(value=value)

    @staticmethod
    def _validate_author(author: str) -> None:
        if not author or not author.strip():
            raise ValueError("author must be a non-empty string")

    @staticmethod
    def _validate_not_empty(value: str, field_name: str) -> None:
        if not value or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    @staticmethod
    def _digest_and_size(content: str) -> tuple[str, int]:
        return content_digest_and_size(content)

    @staticmethod
    def _build_manifest(
        *,
        doc: MongoPluginSkillDocument,
        resources: Sequence[MongoPluginResourceDocument],
        scripts: Sequence[MongoPluginScriptDocument],
    ) -> tuple[ManifestFileEntry, ...] | Literal["dynamic"] | None:
        if doc.is_dynamic:
            return "dynamic"
        if doc.digest is None or doc.size is None:
            return None

        entries: list[ManifestFileEntry] = [ManifestFileEntry(path="SKILL.md", digest=doc.digest, size=doc.size)]
        for resource in resources:
            if resource.digest is None or resource.size is None:
                return None
            entries.append(
                ManifestFileEntry(
                    path=f"references/{resource.resource_name}", digest=resource.digest, size=resource.size
                )
            )
        for script in scripts:
            if script.digest is None or script.size is None:
                return None
            entries.append(
                ManifestFileEntry(path=f"scripts/{script.script_name}", digest=script.digest, size=script.size)
            )
        return tuple(sorted(entries, key=lambda entry: entry.path))

    @staticmethod
    def _extract_description(content: str) -> str:
        match = _FRONTMATTER_RE.match(content)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1))
                if isinstance(frontmatter, dict):
                    desc = frontmatter.get("description", "")
                    if isinstance(desc, str) and desc.strip():
                        return desc.strip()
            except yaml.YAMLError as e:
                logger.debug("Failed to parse YAML frontmatter in skill content: %s", e)

        for line in content.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:200]

        return "Plugin skill"

    # --- Plugin catalog -------------------------------------------------------

    @_repairs_first
    async def save_plugin(
        self,
        *,
        plugin_name: str,
        description: str,
        skills: Sequence[str],
        mcp_servers: Sequence[dict[str, object]],
        mcp_servers_skipped: Sequence[dict[str, object]] = (),
    ) -> MongoPluginDefinitionDocument:
        """Upsert a plugin definition document."""
        now = datetime.now(UTC)
        sv = self.SCHEMA_VERSION_FIELD
        if mcp_servers_skipped:
            logger.error(
                "save_plugin: upserting plugin '%s' to collection '%s' -- "
                "%d MCP server(s) skipped for unresolved ${ENV_VAR} references: %s",
                plugin_name,
                self._plugins_collection.name,
                len(mcp_servers_skipped),
                [s.get("server_key") for s in mcp_servers_skipped],
            )
        else:
            logger.info(
                "save_plugin: upserting plugin '%s' to collection '%s'",
                plugin_name,
                self._plugins_collection.name,
            )
        raw = await self._plugins_collection.find_one_and_update(
            self._version_filter({"plugin_name": plugin_name}),
            {
                "$set": {
                    "description": description,
                    "skills": list(skills),
                    "mcp_servers": [dict(s) for s in mcp_servers],
                    "mcp_servers_skipped": [dict(s) for s in mcp_servers_skipped],
                    "date_modified": now,
                },
                "$setOnInsert": {
                    "plugin_name": plugin_name,
                    sv: self._schema_version,
                    "date_created": now,
                },
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if raw is None:
            raise SkillLoaderError(f"save_plugin: upsert returned no document for plugin '{plugin_name}'")
        return MongoPluginDefinitionDocument.from_mongo_dict(raw)

    @_repairs_first
    async def plugin_exists(self, *, plugin_name: str) -> bool:
        """Return True if a plugin definition exists for the given name."""
        count = await self._plugins_collection.count_documents(
            self._version_filter({"plugin_name": plugin_name}), limit=1
        )
        return count > 0

    @_repairs_first
    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]:
        """Return all plugin definitions for the current schema version, skipping malformed documents."""
        cursor = self._plugins_collection.find(self._version_filter({})).sort("plugin_name", 1)
        results: list[MongoPluginDefinitionDocument] = []
        async for doc in cursor:
            try:
                results.append(MongoPluginDefinitionDocument.from_mongo_dict(doc))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "list_plugins: skipping malformed plugin document _id=%s: %s",
                    doc.get("_id"),
                    exc,
                )
        return results

    @_repairs_first
    async def has_plugins(self) -> bool:
        """Return True if the plugins collection has at least one document for the current schema version."""
        count = await self._plugins_collection.count_documents(self._version_filter({}), limit=1)
        return count > 0
