from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    AllAuthorsSummaryPageLoader,
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.skill_sync import SYSTEM_USER_ID as _SYSTEM_AUTHOR
from langchain_ai_skills_framework.loaders.summary_page_cursor import DEFAULT_PAGE_LIMIT
from langchain_ai_skills_framework.models.skills_model import SkillSummary
from langchain_ai_skills_framework.services.skill_operation_error import require_non_empty
from langchain_ai_skills_framework.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["SKILLS"])


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """Lightweight summary returned by :class:`ListSkillsService`."""

    name: str
    description: str
    plugin_name: str | None = None
    folder: str | None = None
    state: str = "published"
    date_modified: datetime | None = None
    author: str | None = None


@dataclass(frozen=True, slots=True)
class SkillInfoPage:
    """One page of ``SkillInfo`` from :meth:`ListSkillsService.execute_all_authors_page`.

    ``next_cursor`` is opaque; ``None`` means this is the last page.
    """

    items: tuple[SkillInfo, ...]
    next_cursor: str | None = None


def _to_skill_info(summary: SkillSummary) -> SkillInfo:
    return SkillInfo(
        name=summary.name,
        description=summary.description,
        plugin_name=summary.plugin_name,
        folder=summary.folder,
        state=summary.state,
        date_modified=summary.date_modified,
        author=(
            str(summary.metadata["user_id"])
            if summary.metadata and summary.metadata.get("user_id") not in (None, _SYSTEM_AUTHOR)
            else None
        ),
    )


class ListSkillsService:
    """List skills available to a user."""

    def __init__(self, *, skill_loader: SkillLoaderProtocol) -> None:
        self._loader = skill_loader

    async def execute(
        self,
        *,
        user_id: str,
        plugin_name: str | None = None,
        folder: str | None = None,
        include_staging: bool = False,
        exclude_states: set[str] | None = None,
        allow_states: set[str] | None = None,
        all_authors: bool = False,
    ) -> Sequence[SkillInfo]:
        """Return available skills as a sequence of ``SkillInfo``.

        ``allow_states`` is a positive whitelist applied after merging.
        Use it when callers need strict state enforcement (e.g. list_skills
        showing only published skills). Takes precedence over ``exclude_states``.

        ``all_authors=True`` returns every author's skills (including other users'
        drafts) as per-author entries. This service performs no authorization: the caller
        must verify the user may list other users' skills. Raises ``SkillOperationError``
        if ``all_authors`` is set without a ``user_id``.
        """
        if all_authors:
            require_non_empty(value=user_id, label="user_id")
        if user_id:
            if all_authors:
                summaries = await self._loader.list_all_summaries(
                    user_id=user_id, allowed_skills=set(), include_staging=include_staging, all_authors=True
                )
            else:
                summaries = await self._loader.list_all_summaries(
                    user_id=user_id, allowed_skills=set(), include_staging=include_staging
                )
        else:
            summaries = self._loader.list_skill_summaries(allowed_skills=set())

        if plugin_name:
            summaries = [s for s in summaries if s.plugin_name == plugin_name]

        if folder is not None:
            summaries = [s for s in summaries if s.folder == folder]

        if allow_states is not None:
            summaries = [s for s in summaries if s.state in allow_states]
        elif exclude_states:
            summaries = [s for s in summaries if s.state not in exclude_states]

        results = sorted((_to_skill_info(s) for s in summaries), key=lambda s: s.name)
        logger.debug("ListSkillsService: found %d skills for user_id=%s", len(results), user_id)
        return results

    async def execute_all_authors_page(
        self,
        *,
        user_id: str,
        include_staging: bool = False,
        limit: int = DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> SkillInfoPage:
        """Return one bounded page of every author's skills as per-author ``SkillInfo`` entries.

        The bounded alternative to ``execute(all_authors=True)``: ordered by ``(plugin_name, skill_name, author)``,
        built without reading resources or scripts. Pass ``next_cursor`` back as ``cursor`` for the next page. This
        service performs no authorization: the caller must verify the user may list other users' skills. It applies
        no ``plugin_name``/``folder``/state filters, because filtering after paging would make page sizes uneven.

        Raises:
            SkillOperationError: If ``user_id`` is empty.
            TypeError: If the loader does not implement ``list_all_authors_summaries_page``.
            ValueError: If ``limit`` is below 1.
            InvalidPageCursorError: If ``cursor`` is malformed or from an incompatible version.
        """
        require_non_empty(value=user_id, label="user_id")
        if not isinstance(self._loader, AllAuthorsSummaryPageLoader):
            raise TypeError(
                f"{type(self._loader).__name__} does not implement list_all_authors_summaries_page; "
                "paged all_authors listing is unavailable."
            )
        page = await self._loader.list_all_authors_summaries_page(
            include_staging=include_staging, limit=limit, cursor=cursor
        )
        return SkillInfoPage(items=tuple(_to_skill_info(s) for s in page.summaries), next_cursor=page.next_cursor)

    @staticmethod
    def format_as_text(skills: Sequence[SkillInfo]) -> str:
        """Format the skill list as an XML string."""
        if not skills:
            return "<available_skills>\n</available_skills>"
        skill_elements = []
        for s in skills:
            plugin_tag = f"<plugin_name>{s.plugin_name}</plugin_name>\n" if s.plugin_name else ""
            folder_tag = f"<folder>{s.folder}</folder>\n" if s.folder else ""
            date_tag = f"<last_updated>{s.date_modified.isoformat()}</last_updated>\n" if s.date_modified else ""
            skill_elements.append(
                f"<skill>\n{plugin_tag}{folder_tag}<name>{s.name}</name>\n"
                f"<description>{s.description}</description>\n{date_tag}</skill>"
            )
        return "<available_skills>\n" + "\n".join(skill_elements) + "\n</available_skills>"
