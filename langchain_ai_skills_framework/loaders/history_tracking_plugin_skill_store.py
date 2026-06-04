"""PluginSkillStore decorator that records mutation history."""

from __future__ import annotations

import logging
import traceback as tb
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence

from langchain_ai_skills_framework.loaders.plugin_skill_store import PluginSkillStore
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
from langchain_ai_skills_framework.models.error_record import ErrorRecord
from langchain_ai_skills_framework.persistence.error_writer import ErrorWriter
from langchain_ai_skills_framework.persistence.history_writer import HistoryWriter
from langchain_ai_skills_framework.utilities.skill_name_normalizer import (
    normalize_skill_name,
)

logger = logging.getLogger(__name__)


class HistoryTrackingPluginSkillStore:
    """Wraps a PluginSkillStore, recording mutation history via HistoryWriter."""

    def __init__(
        self,
        *,
        inner_store: PluginSkillStore,
        history_writer: HistoryWriter,
        error_writer: ErrorWriter | None = None,
    ) -> None:
        self._inner = inner_store
        self._history = history_writer
        self._errors = error_writer

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
        in_testing: bool | None = None,
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
                in_testing=in_testing,
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
            timestamp=datetime.now(timezone.utc),
            source_collection="plugin_skills",
            user_id=author,
            plugin_name=doc.plugin_name,
            skill_name=doc.skill_name,
        )
        await self._history.write_skill_history(record=record)
        return doc

    async def set_skill_published(
        self,
        *,
        author: str,
        plugin_name: str,
        skill_name: str,
        published: bool,
        published_branch: str | None = None,
    ) -> MongoPluginSkillDocument:
        try:
            doc = await self._inner.set_skill_published(
                author=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
                published=published,
                published_branch=published_branch,
            )
        except Exception as exc:
            await self._record_error(
                operation="publish",
                exc=exc,
                user_id=author,
                plugin_name=plugin_name,
                skill_name=skill_name,
            )
            raise

        action: Literal["published", "unpublished"] = "published" if published else "unpublished"
        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=author,
            timestamp=datetime.now(timezone.utc),
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
        except Exception:
            logger.debug("Could not capture pre-delete snapshot for skill %s/%s", plugin_name, skill_name)

        resource_names: Sequence[str] = []
        script_names: Sequence[str] = []
        try:
            resource_names = await self._inner.list_resource_names(
                author=author, plugin_name=plugin_name, skill_name=skill_name
            )
        except Exception:
            logger.debug("Could not enumerate resources before skill delete %s/%s", plugin_name, skill_name)
        try:
            script_names = await self._inner.list_script_names(
                author=author, plugin_name=plugin_name, skill_name=skill_name
            )
        except Exception:
            logger.debug("Could not enumerate scripts before skill delete %s/%s", plugin_name, skill_name)

        deleted = await self._inner.delete_skill(author=author, plugin_name=plugin_name, skill_name=skill_name)

        if deleted:
            now = datetime.now(timezone.utc)
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
            timestamp=datetime.now(timezone.utc),
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
        except Exception:
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
                timestamp=datetime.now(timezone.utc),
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

        record = HistoryRecord(
            action=action,
            document_snapshot=doc.to_mongo_dict(),
            changed_by=modified_by or author,
            timestamp=datetime.now(timezone.utc),
            source_collection="plugin_scripts",
            user_id=author,
            plugin_name=doc.plugin_name,
            skill_name=doc.skill_name,
            script_name=doc.script_name,
        )
        await self._history.write_script_history(record=record)
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
        except Exception:
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
            record = HistoryRecord(
                action="deleted",
                document_snapshot=snapshot,
                changed_by=author,
                timestamp=datetime.now(timezone.utc),
                source_collection="plugin_scripts",
                user_id=author,
                plugin_name=plugin_name,
                skill_name=normalized_skill,
                script_name=normalized_script,
            )
            await self._history.write_script_history(record=record)

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
    ) -> MongoPluginDefinitionDocument:
        exists = await self._inner.plugin_exists(plugin_name=plugin_name)
        action: Literal["created", "updated"] = "updated" if exists else "created"

        try:
            doc = await self._inner.save_plugin(
                plugin_name=plugin_name,
                description=description,
                skills=skills,
                mcp_servers=mcp_servers,
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
            timestamp=datetime.now(timezone.utc),
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
        self, *, author: str, plugin_name: str | None = None, include_testing: bool = False
    ) -> SkillSnapshot:
        return await self._inner.load_snapshot(author=author, plugin_name=plugin_name, include_testing=include_testing)

    async def load_shared_snapshot(self, *, plugin_name: str | None = None) -> SkillSnapshot:
        return await self._inner.load_shared_snapshot(plugin_name=plugin_name)

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
        author: str,
    ) -> MongoPluginSkillUsageDocument:
        return await self._inner.record_skill_usage(plugin_name=plugin_name, skill_name=skill_name, author=author)

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
        return await self._history.get_script_history(
            user_id=user_id,
            plugin_name=plugin_name,
            skill_name=normalize_skill_name(value=skill_name),
            script_name=script_name.strip(),
            limit=limit,
            offset=offset,
        )

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
