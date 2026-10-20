"""PluginSkillStore decorator that records mutation history."""

from __future__ import annotations

import asyncio
import logging
import traceback as tb
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
from langchain_ai_skills_framework.loaders.schema_migrations import content_digest_and_size
from langchain_ai_skills_framework.models.error_record import ErrorRecord
from langchain_ai_skills_framework.models.history_record import HistoryRecord
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginDefinitionDocument,
    MongoPluginResourceDocument,
    MongoPluginScriptDocument,
    MongoPluginSkillDocument,
    MongoPluginSkillUsageDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSnapshot,
)
from langchain_ai_skills_framework.persistence.error_writer import ErrorWriter
from langchain_ai_skills_framework.persistence.history_writer import HistoryWriter
from langchain_ai_skills_framework.persistence.script_content_store import (
    ScriptContentStore,
    build_script_content_key,
    build_script_history_content_ref,
)
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)

logger = logging.getLogger(__name__)

_CONTENT_FIELD = "content"
_CONTENT_REF_FIELD = "content_ref"


class HistoryTrackingPluginSkillStore:
    """Wraps a PluginSkillStore, recording mutation history via HistoryWriter.

    Script history and the external content store: when ``script_content_store`` is given **and**
    ``write_script_history_to_content_store`` is true, a script history snapshot does not carry the script text.
    Its content is written to the store under a ref of its own (``content_ref`` in the snapshot), so a script near
    the 16 MB document limit still gets a history record and the live script's ref can be retired freely. With the
    flag off the records are exactly what they were before. Reads (``get_script_history``) resolve ``content_ref``
    whenever a store is configured, regardless of the flag, and still accept old records with inline ``content``, so
    ``document_snapshot`` always looks the same to callers.
    """

    def __init__(
        self,
        *,
        inner_store: PluginSkillStore,
        history_writer: HistoryWriter,
        error_writer: ErrorWriter | None = None,
        script_content_store: ScriptContentStore | None = None,
        write_script_history_to_content_store: bool = False,
    ) -> None:
        self._inner = inner_store
        self._history = history_writer
        self._errors = error_writer
        self._script_content_store = script_content_store
        self._write_script_history_to_content_store = write_script_history_to_content_store

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        await self._inner.ensure_indexes()
        await self._history.ensure_indexes()
        if self._errors is not None:
            await self._errors.ensure_indexes()

    # ------------------------------------------------------------------
    # Skill mutations
    # ------------------------------------------------------------------

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
    ) -> MongoPluginSkillDocument:
        exists = await self._inner.skill_exists(author=author, plugin_name=plugin_name, skill_name=skill_name)
        action: Literal["created", "updated"] = "updated" if exists else "created"

        try:
            doc = await self._inner.save_skill(
                author=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                content=content,
                modified_by=modified_by,
                folder=folder,
                path=path,
                state=state,
            )
        except Exception as exc:
            await self._record_error(
                operation="save",
                exc=exc,
                user_id=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            raise

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=modified_by or author,
            timestamp=datetime.now(UTC),
            source_collection="plugin_skills",
            user_id=author,
            plugin_name=doc.plugin_name,
            skill_name=doc.skill_name,
        )
        await self._history.write_skill_history(record=record)
        return doc

    async def set_skill_state(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        state: str,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument:
        try:
            doc = await self._inner.set_skill_state(
                author=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                state=state,
                published_branch=published_branch,
            )
        except Exception as exc:
            await self._record_error(
                operation="state_change",
                exc=exc,
                user_id=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            raise

        record = HistoryRecord(
            action="state_changed",
            document_snapshot={**doc.to_mongo_dict(), "state": state},
            changed_by=author,
            timestamp=datetime.now(UTC),
            source_collection="plugin_skills",
            user_id=author,
            plugin_name=doc.plugin_name,
            skill_name=doc.skill_name,
        )
        await self._history.write_skill_history(record=record)
        return doc

    async def delete_skill(self, *, author: str, plugin_name: str, skill_name: str) -> bool:
        normalized_name = normalize_skill_name(value=skill_name)

        snapshot: dict[str, object] = {}
        try:
            details = await self._inner.get_skill_details(author=author, plugin_name=plugin_name, skill_name=skill_name)
            snapshot = {
                "skill_name": details.name,
                "description": details.description,
                "content": details.content,
                "plugin_name": plugin_name,
                "user_id": author,
            }
        except Exception:  # noqa: BLE001 - pre-delete snapshot is best-effort; store backends vary in what they raise for "not found"
            logger.debug("Could not capture pre-delete snapshot for skill %s/%s", plugin_name, skill_name)

        resource_names: Sequence[str] = []
        script_names: Sequence[str] = []
        try:
            resource_names = await self._inner.list_resource_names(
                author=author, plugin_name=plugin_name, skill_name=skill_name
            )
        except Exception:  # noqa: BLE001 - resource enumeration is best-effort before delete; must not block the delete itself
            logger.debug("Could not enumerate resources before skill delete %s/%s", plugin_name, skill_name)
        try:
            script_names = await self._inner.list_script_names(
                author=author, plugin_name=plugin_name, skill_name=skill_name
            )
        except Exception:  # noqa: BLE001 - script enumeration is best-effort before delete; must not block the delete itself
            logger.debug("Could not enumerate scripts before skill delete %s/%s", plugin_name, skill_name)

        deleted = await self._inner.delete_skill(author=author, plugin_name=plugin_name, skill_name=skill_name)

        if deleted:
            now = datetime.now(UTC)
            for resource in resource_names:
                child_record = HistoryRecord(
                    action="deleted",
                    document_snapshot={"resource_name": resource, "plugin_name": plugin_name, "user_id": author},
                    changed_by=author,
                    timestamp=now,
                    source_collection="plugin_references",
                    user_id=author,
                    plugin_name=plugin_name,
                    skill_name=normalized_name,
                    resource_name=resource,
                )
                await self._history.write_reference_history(record=child_record)
            for script in script_names:
                child_record = HistoryRecord(
                    action="deleted",
                    document_snapshot={"script_name": script, "plugin_name": plugin_name, "user_id": author},
                    changed_by=author,
                    timestamp=now,
                    source_collection="plugin_scripts",
                    user_id=author,
                    plugin_name=plugin_name,
                    skill_name=normalized_name,
                    script_name=script,
                )
                await self._history.write_script_history(record=child_record)
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=author,
                timestamp=now,
                source_collection="plugin_skills",
                user_id=author,
                plugin_name=plugin_name,
                skill_name=normalized_name,
            )
            await self._history.write_skill_history(record=record)

        return deleted

    # ------------------------------------------------------------------
    # Resource mutations
    # ------------------------------------------------------------------

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
        exists = await self._inner.resource_exists(
            author=author, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )
        action: Literal["created", "updated"] = "updated" if exists else "created"

        try:
            doc = await self._inner.save_resource(
                author=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=resource_name,
                content=content,
                modified_by=modified_by,
                folder=folder,
                path=path,
            )
        except Exception as exc:
            await self._record_error(
                operation="save",
                exc=exc,
                user_id=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                resource_name=resource_name,
            )
            raise

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=modified_by or author,
            timestamp=datetime.now(UTC),
            source_collection="plugin_references",
            user_id=author,
            plugin_name=doc.plugin_name,
            skill_name=doc.skill_name,
            resource_name=doc.resource_name,
        )
        await self._history.write_reference_history(record=record)
        return doc

    async def delete_resource(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        normalized_skill = normalize_skill_name(value=skill_name)
        normalized_resource = resource_name.strip()

        snapshot: dict[str, object] = {}
        try:
            content = await self._inner.read_resource(
                author=author, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
            )
            snapshot = {
                "resource_name": normalized_resource,
                "content": content,
                "skill_name": normalized_skill,
                "plugin_name": plugin_name,
                "user_id": author,
            }
        except Exception:  # noqa: BLE001 - pre-delete snapshot is best-effort; store backends vary in what they raise for "not found"
            logger.debug(
                "Could not capture pre-delete snapshot for resource %s/%s/%s",
                plugin_name,
                skill_name,
                resource_name,
            )

        deleted = await self._inner.delete_resource(
            author=author, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )

        if deleted:
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=author,
                timestamp=datetime.now(UTC),
                source_collection="plugin_references",
                user_id=author,
                plugin_name=plugin_name,
                skill_name=normalized_skill,
                resource_name=normalized_resource,
            )
            await self._history.write_reference_history(record=record)

        return deleted

    # ------------------------------------------------------------------
    # Script mutations
    # ------------------------------------------------------------------

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
        exists = await self._inner.script_exists(
            author=author, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )
        action: Literal["created", "updated"] = "updated" if exists else "created"

        try:
            doc = await self._inner.save_script(
                author=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
                content=content,
                modified_by=modified_by,
                folder=folder,
                path=path,
            )
        except Exception as exc:
            await self._record_error(
                operation="save",
                exc=exc,
                user_id=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                script_name=script_name,
            )
            raise

        snapshot, history_ref = await self._externalize_script_snapshot(
            snapshot=doc.to_mongo_dict(), author=author, plugin_name=doc.plugin_name
        )
        record = HistoryRecord(
            action=action,
            document_snapshot=snapshot,
            changed_by=modified_by or author,
            timestamp=datetime.now(UTC),
            source_collection="plugin_scripts",
            user_id=author,
            plugin_name=doc.plugin_name,
            skill_name=doc.skill_name,
            script_name=doc.script_name,
        )
        await self._write_script_history(record=record, history_ref=history_ref)
        return doc

    async def delete_script(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
    ) -> bool:
        normalized_skill = normalize_skill_name(value=skill_name)
        normalized_script = script_name.strip()

        snapshot: dict[str, object] = {}
        try:
            content = await self._inner.read_script(
                author=author, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
            )
            snapshot = {
                "script_name": normalized_script,
                "content": content,
                "skill_name": normalized_skill,
                "plugin_name": plugin_name,
                "user_id": author,
            }
        except Exception:  # noqa: BLE001 - pre-delete snapshot is best-effort; store backends vary in what they raise for "not found"
            logger.debug(
                "Could not capture pre-delete snapshot for script %s/%s/%s",
                plugin_name,
                skill_name,
                script_name,
            )

        deleted = await self._inner.delete_script(
            author=author, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )

        if deleted:
            # Only after the delete succeeded, so a failed delete leaves no stray content copy behind.
            snapshot, history_ref = await self._externalize_script_snapshot(
                snapshot=snapshot, author=author, plugin_name=plugin_name
            )
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=author,
                timestamp=datetime.now(UTC),
                source_collection="plugin_scripts",
                user_id=author,
                plugin_name=plugin_name,
                skill_name=normalized_skill,
                script_name=normalized_script,
            )
            await self._write_script_history(record=record, history_ref=history_ref)

        return deleted

    # ------------------------------------------------------------------
    # Plugin catalog mutation
    # ------------------------------------------------------------------

    async def save_plugin(
        self,
        *,
        plugin_name: str,
        description: str,
        skills: Sequence[str],
        mcp_servers: Sequence[dict[str, object]],
        mcp_servers_skipped: Sequence[dict[str, object]] = (),
    ) -> MongoPluginDefinitionDocument:
        exists = await self._inner.plugin_exists(plugin_name=plugin_name)
        action: Literal["created", "updated"] = "updated" if exists else "created"

        try:
            doc = await self._inner.save_plugin(
                plugin_name=plugin_name,
                description=description,
                skills=skills,
                mcp_servers=mcp_servers,
                mcp_servers_skipped=mcp_servers_skipped,
            )
        except Exception as exc:
            await self._record_error(
                operation="save",
                exc=exc,
                plugin_name=plugin_name,
            )
            raise

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by="system",
            timestamp=datetime.now(UTC),
            source_collection="plugins",
            user_id="system",
            plugin_name=doc.plugin_name,
        )
        await self._history.write_plugin_history(record=record)
        return doc

    # ------------------------------------------------------------------
    # Read-only pass-through methods
    # ------------------------------------------------------------------

    async def load_snapshot(
        self, *, author: str, plugin_name: str | None = None, include_staging: bool = False
    ) -> SkillSnapshot:
        return await self._inner.load_snapshot(author=author, plugin_name=plugin_name, include_staging=include_staging)

    async def load_shared_snapshot(
        self, *, plugin_name: str | None = None, include_staging: bool = False
    ) -> SkillSnapshot:
        return await self._inner.load_shared_snapshot(plugin_name=plugin_name, include_staging=include_staging)

    async def get_skill_details(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> SkillDetails:
        return await self._inner.get_skill_details(author=author, plugin_name=plugin_name, skill_name=skill_name)

    async def skill_exists(self, *, author: str, plugin_name: str | None = None, skill_name: str) -> bool:
        return await self._inner.skill_exists(author=author, plugin_name=plugin_name, skill_name=skill_name)

    async def read_resource(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> str:
        return await self._inner.read_resource(
            author=author, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )

    async def list_resource_names(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        return await self._inner.list_resource_names(author=author, plugin_name=plugin_name, skill_name=skill_name)

    async def list_resource_documents(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[MongoPluginResourceDocument]:
        return await self._inner.list_resource_documents(author=author, plugin_name=plugin_name, skill_name=skill_name)

    async def resource_exists(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        resource_name: str,
    ) -> bool:
        return await self._inner.resource_exists(
            author=author, plugin_name=plugin_name, skill_name=skill_name, resource_name=resource_name
        )

    async def read_script(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> str:
        return await self._inner.read_script(
            author=author, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )

    async def list_script_names(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[str]:
        return await self._inner.list_script_names(author=author, plugin_name=plugin_name, skill_name=skill_name)

    async def list_script_documents(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
    ) -> Sequence[MongoPluginScriptDocument]:
        return await self._inner.list_script_documents(author=author, plugin_name=plugin_name, skill_name=skill_name)

    async def script_exists(
        self,
        *,
        author: str,
        plugin_name: str | None = None,
        skill_name: str,
        script_name: str,
    ) -> bool:
        return await self._inner.script_exists(
            author=author, plugin_name=plugin_name, skill_name=skill_name, script_name=script_name
        )

    async def record_skill_usage(
        self,
        *,
        plugin_name: str,
        skill_name: str,
        user_id: str,
    ) -> MongoPluginSkillUsageDocument:
        return await self._inner.record_skill_usage(plugin_name=plugin_name, skill_name=skill_name, user_id=user_id)

    async def get_skill_usage_count(self, *, skill_name: str) -> int:
        return await self._inner.get_skill_usage_count(skill_name=skill_name)

    async def get_skill_usage_counts(self, *, skill_names: Sequence[str]) -> Mapping[str, int]:
        return await self._inner.get_skill_usage_counts(skill_names=skill_names)

    async def plugin_exists(self, *, plugin_name: str) -> bool:
        return await self._inner.plugin_exists(plugin_name=plugin_name)

    async def list_plugins(self) -> Sequence[MongoPluginDefinitionDocument]:
        return await self._inner.list_plugins()

    async def has_plugins(self) -> bool:
        return await self._inner.has_plugins()

    # ------------------------------------------------------------------
    # History query pass-through
    # ------------------------------------------------------------------

    async def get_skill_history(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_skill_history(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=normalize_skill_name(value=skill_name),
            limit=limit,
            offset=offset,
        )

    async def get_resource_history(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        resource_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_resource_history(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=normalize_skill_name(value=skill_name),
            resource_name=resource_name.strip(),
            limit=limit,
            offset=offset,
        )

    async def get_script_history(
        self,
        *,
        user_id: str,
        plugin_name: str,
        skill_name: str,
        script_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        records = await self._history.get_script_history(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=normalize_skill_name(value=skill_name),
            script_name=script_name.strip(),
            limit=limit,
            offset=offset,
        )
        return await asyncio.gather(*(self._resolve_script_history_content(record=record) for record in records))

    async def get_plugin_history(
        self,
        *,
        plugin_name: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[HistoryRecord]:
        return await self._history.get_plugin_history(plugin_name=plugin_name, limit=limit, offset=offset)

    # ------------------------------------------------------------------
    # Error query pass-through
    # ------------------------------------------------------------------

    async def get_errors(
        self,
        *,
        user_id: str | None = None,
        plugin_name: str | None = None,
        skill_name: str | None = None,
        operation: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ErrorRecord]:
        if self._errors is None:
            return []
        normalized_skill = normalize_skill_name(value=skill_name) if skill_name else None
        return await self._errors.get_errors(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=normalized_skill,
            operation=operation,
            limit=limit,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Script history content (external store)
    # ------------------------------------------------------------------

    async def _externalize_script_snapshot(
        self, *, snapshot: dict[str, object], author: str, plugin_name: str
    ) -> tuple[dict[str, object], str | None]:
        """Move a script snapshot's content into the store; return ``(snapshot, ref written or None)``.

        Does nothing (returns ``snapshot`` unchanged) unless a store is configured and history writes to it are on,
        or when the snapshot has no text content (e.g. a pre-delete read that failed). If the store put fails, the
        snapshot keeps its inline content and the failure is logged: the record is then written exactly as it was
        before this feature, so a small script still gets its history and an oversized one hits the existing
        "failed to write script history" warning. The user's save or delete is never failed by history.
        """
        store = self._script_content_store
        content = snapshot.get(_CONTENT_FIELD)
        if store is None or not self._write_script_history_to_content_store or not isinstance(content, str):
            return snapshot, None
        script_key = build_script_content_key(
            author=author,
            plugin_name=plugin_name,
            skill_name=str(snapshot.get("skill_name", "")),
            script_name=str(snapshot.get("script_name", "")),
        )
        digest = snapshot.get("digest")
        if not isinstance(digest, str) or not digest:
            digest, _ = content_digest_and_size(content)
        ref = build_script_history_content_ref(script_key=script_key, digest=digest)
        try:
            await store.put(ref=ref, content=content)
        except Exception:  # history must never fail the user's save/delete; fall back to the inline record
            logger.warning(
                "Could not store script history content %s; recording the snapshot with inline content",
                ref,
                exc_info=True,
            )
            return snapshot, None
        externalized = {key: value for key, value in snapshot.items() if key != _CONTENT_FIELD}
        externalized[_CONTENT_REF_FIELD] = ref
        return externalized, ref

    async def _write_script_history(self, *, record: HistoryRecord, history_ref: str | None) -> None:
        """Write a script history record; if it was not stored, drop the content copy made for it."""
        written = await self._history.write_script_history(record=record)
        if written is False and history_ref is not None and self._script_content_store is not None:
            try:
                await self._script_content_store.delete(ref=history_ref)
            except Exception:  # best-effort cleanup of an unreferenced leftover
                logger.warning("Could not delete script history content %s", history_ref, exc_info=True)

    async def _resolve_script_history_content(self, *, record: HistoryRecord) -> HistoryRecord:
        """Return ``record`` with ``content`` restored from the store when its snapshot holds a ``content_ref``.

        Records with inline ``content`` (everything written before this feature, or with the flag off) are returned
        untouched. A resolved snapshot has ``content`` and no ``content_ref``, the shape callers always saw. If the
        content cannot be fetched (no store configured, the entry is gone, or the store raises) the record is still
        returned, with ``content`` left absent and ``content_ref`` kept, and a warning is logged: never an empty
        string standing in for lost content, and one unreadable entry does not fail the whole history listing.
        """
        snapshot = record.document_snapshot
        ref = snapshot.get(_CONTENT_REF_FIELD)
        if snapshot.get(_CONTENT_FIELD) is not None or not isinstance(ref, str):
            return record
        store = self._script_content_store
        if store is None:
            logger.warning("Script history content %s is in an external store but none is configured", ref)
            return record
        try:
            content = await store.get(ref=ref)
        except Exception:
            logger.warning("Could not read script history content %s", ref, exc_info=True)
            return record
        if content is None:
            logger.warning("Script history content %s is missing from the content store", ref)
            return record
        resolved = {key: value for key, value in snapshot.items() if key != _CONTENT_REF_FIELD}
        resolved[_CONTENT_FIELD] = content
        return record.model_copy(update={"document_snapshot": resolved})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _record_error(
        self,
        *,
        operation: str,
        exc: Exception,
        user_id: str = "",
        plugin_name: str = "",
        skill_name: str = "",
        resource_name: str | None = None,
        script_name: str | None = None,
    ) -> None:
        if self._errors is None:
            return
        error_record = ErrorRecord(
            operation=operation,  # type: ignore[arg-type]
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback="".join(tb.format_exception(exc)),
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=normalize_skill_name(value=skill_name) if skill_name else "",
            resource_name=resource_name,
            script_name=script_name,
        )
        await self._errors.write_error(record=error_record)
