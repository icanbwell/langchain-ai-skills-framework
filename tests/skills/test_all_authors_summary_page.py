"""Paged, summaries-only all-authors listing (BAI-1509)."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.loaders.composite_skill_loader import CompositeSkillLoader
from langchain_ai_skills_framework.loaders.history_tracking_plugin_skill_store import HistoryTrackingPluginSkillStore
from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.loaders.null_plugin_skill_store import NullPluginSkillStore
from langchain_ai_skills_framework.loaders.plugin_skill_store import (
    AllAuthorsSnapshotStore,
    AllAuthorsSummaryPageStore,
)
from langchain_ai_skills_framework.loaders.skill_loader_protocol import (
    AllAuthorsSummaryPageLoader,
    SkillLoaderProtocol,
)
from langchain_ai_skills_framework.loaders.summary_page_cursor import (
    MAX_CURSOR_LENGTH,
    MAX_PAGE_LIMIT,
    InvalidPageCursorError,
    SummaryPageKey,
    decode_cursor,
    encode_cursor,
    filters_fingerprint,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginSkillDocument
from langchain_ai_skills_framework.services.list_skills_service import ListSkillsService
from langchain_ai_skills_framework.services.skill_operation_error import SkillOperationError

_KEY_FIELDS = ("plugin_name", "skill_name", "author")


def _matches_clause(doc: Mapping[str, Any], clause: Mapping[str, Any]) -> bool:
    for field, expected in clause.items():
        if isinstance(expected, dict):
            if "$gt" in expected and not doc[field] > expected["$gt"]:
                return False
            if "$ne" in expected and doc[field] == expected["$ne"]:
                return False
            if "$in" in expected and doc.get(field) not in expected["$in"]:
                return False
        elif doc.get(field) != expected:
            return False
    return True


def _matches(doc: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    plain = {k: v for k, v in query.items() if k != "$or"}
    if not _matches_clause(doc, plain):
        return False
    return "$or" not in query or any(_matches_clause(doc, c) for c in query["$or"])


class _FakeCursor:
    limits: ClassVar[list[int]] = []  # every limit() argument seen, to assert the read is bounded

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    def sort(self, spec: list[tuple[str, int]]) -> _FakeCursor:
        assert [k for k, _ in spec] == list(_KEY_FIELDS)
        return _FakeCursor(sorted(self._docs, key=lambda d: tuple(d[k] for k in _KEY_FIELDS)))

    def limit(self, n: int) -> _FakeCursor:
        _FakeCursor.limits.append(n)
        return _FakeCursor(self._docs[:n])

    def __aiter__(self) -> _FakeCursor:
        self._it = iter(self._docs)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._it)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.fixture(autouse=True)
def _reset_recorded_limits() -> None:
    _FakeCursor.limits = []


def _skill(author: str, plugin: str, name: str, state: str = "published", folder: str | None = None) -> dict[str, Any]:
    return {
        "folder": folder,
        "author": author,
        "plugin_name": plugin,
        "skill_name": name,
        "description": "d",
        "content": "SECRET-BODY",
        "state": state,
        "digest": "sha256:" + "a" * 64,
        "size": 1,
        MongoPluginSkillLoader.SCHEMA_VERSION_FIELD: MongoPluginSkillDocument.SCHEMA_VERSION,
    }


def _loader(database: MagicMock, skills: list[dict[str, Any]]) -> MongoPluginSkillLoader:
    def find(query: dict[str, Any], projection: dict[str, int] | None = None) -> _FakeCursor:
        rows = [
            {k: v for k, v in d.items() if not (projection and projection.get(k) == 0)}
            for d in skills
            if _matches(d, query)
        ]
        return _FakeCursor(rows)

    database["plugin_skills"].find = MagicMock(side_effect=find)
    return MongoPluginSkillLoader(database=database)


def _fixture_skills() -> list[dict[str, Any]]:
    # Inserted out of order on purpose; includes the same skill name under two authors and two plugins.
    return [
        _skill("b@example.com", "p2", "alpha"),
        _skill("a@example.com", "p1", "zeta"),
        _skill("b@example.com", "p1", "same", "draft"),
        _skill("a@example.com", "p1", "same", "draft"),
        _skill("system", "p1", "alpha"),
        _skill("a@example.com", "p3", "staged", "staging"),
    ]


EXPECTED_ORDER = [
    ("p1", "alpha", "system"),
    ("p1", "same", "a@example.com"),
    ("p1", "same", "b@example.com"),
    ("p1", "zeta", "a@example.com"),
    ("p2", "alpha", "b@example.com"),
]


def _keys(page_summaries: Any) -> list[tuple[str | None, str, str]]:
    return [(s.plugin_name, s.name, str(s.metadata["user_id"])) for s in page_summaries]


async def _collect(loader: MongoPluginSkillLoader, *, limit: int, **kwargs: Any) -> list[Any]:
    pages = []
    cursor = None
    while True:
        page = await loader.list_all_summaries_page(limit=limit, cursor=cursor, **kwargs)
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            return pages
        assert len(pages) < 50, "paging did not terminate"


@pytest.mark.parametrize(("limit", "expected_sizes"), [(1, [1, 1, 1, 1, 1]), (2, [2, 2, 1]), (5, [5]), (100, [5])])
async def test_pages_are_bounded_ordered_and_cover_everything_once(
    mock_mongo_database: MagicMock, limit: int, expected_sizes: list[int]
) -> None:
    loader = _loader(mock_mongo_database, _fixture_skills())

    pages = await _collect(loader, limit=limit)

    assert [len(p.summaries) for p in pages] == expected_sizes
    assert all(len(p.summaries) <= limit for p in pages)
    assert [k for p in pages for k in _keys(p.summaries)] == EXPECTED_ORDER
    assert [p.next_cursor is None for p in pages] == [False] * (len(pages) - 1) + [True]


async def test_exact_multiple_of_limit_ends_without_dangling_cursor(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, [s for s in _fixture_skills() if s["state"] != "staging"][:4])

    pages = await _collect(loader, limit=2)

    assert [len(p.summaries) for p in pages] == [2, 2]
    assert pages[-1].next_cursor is None


async def test_empty_result_has_no_cursor(mock_mongo_database: MagicMock) -> None:
    page = await _loader(mock_mongo_database, []).list_all_summaries_page(limit=10)

    assert page.summaries == ()
    assert page.next_cursor is None


@pytest.mark.parametrize(("include_staging", "expected_count"), [(False, 5), (True, 6)])
async def test_include_staging_controls_which_states_are_listed(
    mock_mongo_database: MagicMock, include_staging: bool, expected_count: int
) -> None:
    pages = await _collect(_loader(mock_mongo_database, _fixture_skills()), limit=2, include_staging=include_staging)

    assert sum(len(p.summaries) for p in pages) == expected_count


async def test_summaries_carry_author_and_no_content_or_manifest(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, _fixture_skills())

    page = await loader.list_all_summaries_page(limit=10)

    assert all(s.manifest is None for s in page.summaries)
    assert {s.metadata["user_id"] for s in page.summaries} >= {"a@example.com", "b@example.com", "system"}
    assert "SECRET-BODY" not in repr(page)
    query, projection = mock_mongo_database["plugin_skills"].find.call_args.args
    assert projection == {"content": 0}
    assert query[MongoPluginSkillLoader.SCHEMA_VERSION_FIELD] == loader._schema_version


async def test_summaries_carry_digest_and_frontmatter_metadata(mock_mongo_database: MagicMock) -> None:
    skill = _skill("a@example.com", "p1", "alpha")
    skill["metadata"] = {"version": "1.2", "user_id": "spoofed"}
    loader = _loader(mock_mongo_database, [skill])

    page = await loader.list_all_summaries_page(limit=10)

    (summary,) = page.summaries
    assert summary.digest == "sha256:" + "a" * 64
    assert summary.metadata["version"] == "1.2"
    assert summary.metadata["user_id"] == "a@example.com"


async def test_never_reads_resource_or_script_collections(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, _fixture_skills())

    await _collect(loader, limit=2)

    mock_mongo_database["plugin_references"].find.assert_not_called()
    mock_mongo_database["plugin_scripts"].find.assert_not_called()


async def test_reads_at_most_limit_plus_one_documents(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, _fixture_skills())

    await loader.list_all_summaries_page(limit=2)

    assert _FakeCursor.limits == [3]


async def test_cursor_survives_deletion_of_its_own_row(mock_mongo_database: MagicMock) -> None:
    skills = _fixture_skills()
    loader = _loader(mock_mongo_database, skills)
    first = await loader.list_all_summaries_page(limit=2)
    skills[:] = [
        s for s in skills if (s["plugin_name"], s["skill_name"], s["author"]) != ("p1", "same", "a@example.com")
    ]

    second = await loader.list_all_summaries_page(limit=2, cursor=first.next_cursor)

    assert _keys(second.summaries) == [("p1", "same", "b@example.com"), ("p1", "zeta", "a@example.com")]


async def test_limit_above_maximum_is_clamped(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, _fixture_skills())

    await loader.list_all_summaries_page(limit=MAX_PAGE_LIMIT * 10)

    assert _FakeCursor.limits == [MAX_PAGE_LIMIT + 1]


@pytest.mark.parametrize("limit", [0, -1])
async def test_non_positive_limit_is_rejected(mock_mongo_database: MagicMock, limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        await _loader(mock_mongo_database, []).list_all_summaries_page(limit=limit)


@pytest.mark.parametrize(
    "bad_cursor",
    [
        "",
        "not-base64!!",
        base64.urlsafe_b64encode(b"not json").decode(),
        base64.urlsafe_b64encode(json.dumps({"v": 99, "k": ["a", "b", "c"]}).encode()).decode(),
        base64.urlsafe_b64encode(json.dumps({"v": 1, "k": ["a", "b"]}).encode()).decode(),
        base64.urlsafe_b64encode(json.dumps({"v": 1, "k": ["a", "b", 3]}).encode()).decode(),
        base64.urlsafe_b64encode(json.dumps([1, 2]).encode()).decode(),
    ],
)
async def test_unknown_or_malformed_cursor_is_rejected_before_querying(
    mock_mongo_database: MagicMock, bad_cursor: str
) -> None:
    loader = _loader(mock_mongo_database, _fixture_skills())

    with pytest.raises(InvalidPageCursorError):
        await loader.list_all_summaries_page(limit=2, cursor=bad_cursor)

    mock_mongo_database["plugin_skills"].find.assert_not_called()


def test_cursor_round_trips_non_ascii_keys() -> None:
    key = SummaryPageKey(plugin_name="plügin", skill_name="skill-é", author="ä@example.com")

    assert decode_cursor(cursor=encode_cursor(key=key)) == key


async def test_null_store_returns_empty_page() -> None:
    page = await NullPluginSkillStore().list_all_summaries_page(limit=5)

    assert page.summaries == ()
    assert page.next_cursor is None


async def test_null_store_validates_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        await NullPluginSkillStore().list_all_summaries_page(limit=0)


async def test_history_wrapper_delegates_without_writing_history() -> None:
    inner = AsyncMock(spec=AllAuthorsSummaryPageStore)
    writer = AsyncMock()
    store = HistoryTrackingPluginSkillStore(inner_store=inner, history_writer=writer)

    await store.list_all_summaries_page(include_staging=True, limit=7, cursor="c")

    inner.list_all_summaries_page.assert_awaited_once_with(
        include_staging=True, limit=7, cursor="c", plugin_name=None, folder=None, allow_states=None
    )
    assert writer.method_calls == []


async def test_history_wrapper_raises_type_error_when_inner_store_lacks_it() -> None:
    class LegacyInner:
        pass

    store = HistoryTrackingPluginSkillStore(inner_store=LegacyInner(), history_writer=AsyncMock())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="list_all_summaries_page"):
        await store.list_all_summaries_page()


def test_capability_is_separate_from_snapshot_capability() -> None:
    class SnapshotOnly:
        async def load_all_snapshot(self, *, include_staging: bool = False) -> None: ...

    assert isinstance(SnapshotOnly(), AllAuthorsSnapshotStore)
    assert not isinstance(SnapshotOnly(), AllAuthorsSummaryPageStore)
    assert isinstance(NullPluginSkillStore(), AllAuthorsSummaryPageStore)


async def test_composite_delegates_to_store_page() -> None:
    store = AsyncMock(spec=AllAuthorsSummaryPageStore)
    composite = CompositeSkillLoader(shared_loader=AsyncMock(spec=SkillLoaderProtocol), user_loader=store)

    await composite.list_all_authors_summaries_page(
        include_staging=True, limit=3, cursor="c", plugin_name="p", folder="f", allow_states={"draft"}
    )

    store.list_all_summaries_page.assert_awaited_once_with(
        include_staging=True, limit=3, cursor="c", plugin_name="p", folder="f", allow_states={"draft"}
    )


async def test_composite_raises_type_error_when_store_lacks_paging() -> None:
    class LegacyStore:
        pass

    composite = CompositeSkillLoader(
        shared_loader=AsyncMock(spec=SkillLoaderProtocol),
        user_loader=LegacyStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="list_all_summaries_page"):
        await composite.list_all_authors_summaries_page()


async def test_only_loaders_with_the_page_method_satisfy_the_page_capability() -> None:
    store = AsyncMock(spec=AllAuthorsSummaryPageStore)
    composite = CompositeSkillLoader(shared_loader=AsyncMock(spec=SkillLoaderProtocol), user_loader=store)

    assert isinstance(composite, AllAuthorsSummaryPageLoader)
    assert not isinstance(AsyncMock(spec=SkillLoaderProtocol), AllAuthorsSummaryPageLoader)


async def test_service_pages_end_to_end_with_authors(mock_mongo_database: MagicMock) -> None:
    store = _loader(mock_mongo_database, _fixture_skills())
    composite = CompositeSkillLoader(shared_loader=AsyncMock(spec=SkillLoaderProtocol), user_loader=store)
    service = ListSkillsService(skill_loader=composite)

    first = await service.execute_all_authors_page(user_id="admin@example.com", limit=3)
    second = await service.execute_all_authors_page(user_id="admin@example.com", limit=3, cursor=first.next_cursor)

    assert [(i.name, i.author) for i in first.items] == [
        ("alpha", None),
        ("same", "a@example.com"),
        ("same", "b@example.com"),
    ]
    assert [(i.name, i.author) for i in second.items] == [("zeta", "a@example.com"), ("alpha", "b@example.com")]
    assert first.next_cursor is not None
    assert second.next_cursor is None


async def test_service_requires_user_id() -> None:
    service = ListSkillsService(skill_loader=AsyncMock(spec=SkillLoaderProtocol))

    with pytest.raises(SkillOperationError):
        await service.execute_all_authors_page(user_id="")


async def test_service_raises_type_error_for_loader_without_paging() -> None:
    service = ListSkillsService(skill_loader=AsyncMock(spec=SkillLoaderProtocol))

    with pytest.raises(TypeError, match="list_all_authors_summaries_page"):
        await service.execute_all_authors_page(user_id="admin@example.com")


def _index_calls(database: MagicMock, collection: str) -> dict[str, tuple[list[tuple[str, int]], bool]]:
    return {
        c.kwargs["name"]: (c.args[0], c.kwargs["unique"]) for c in database[collection].create_index.await_args_list
    }


async def test_ensure_indexes_creates_listing_index_serving_the_page_query(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)

    await loader.ensure_indexes()

    keys, unique = _index_calls(mock_mongo_database, "plugin_skills")[MongoPluginSkillLoader.LISTING_INDEX_NAME]
    assert unique is False
    # equality field first, then exactly the sort the query uses, so the index order satisfies the sort
    assert keys == [
        (MongoPluginSkillLoader.SCHEMA_VERSION_FIELD, 1),
        ("plugin_name", 1),
        ("skill_name", 1),
        ("author", 1),
    ]
    state_keys, state_unique = _index_calls(mock_mongo_database, "plugin_skills")[
        MongoPluginSkillLoader.LISTING_STATE_INDEX_NAME
    ]
    assert state_unique is False
    assert state_keys == [
        (MongoPluginSkillLoader.SCHEMA_VERSION_FIELD, 1),
        ("state", 1),
        ("plugin_name", 1),
        ("skill_name", 1),
        ("author", 1),
    ]


async def test_page_query_filter_and_sort_match_the_listing_index(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, [])
    cursor = encode_cursor(
        key=SummaryPageKey("p", "s", "a"),
        fingerprint=filters_fingerprint(include_staging=False, plugin_name=None, folder=None, allow_states=None),
    )
    sorts: list[list[tuple[str, int]]] = []

    def record_sort(spec: list[tuple[str, int]]) -> _FakeCursor:
        sorts.append(spec)
        return _FakeCursor([])

    mock_mongo_database["plugin_skills"].find.side_effect = None
    mock_mongo_database["plugin_skills"].find.return_value = MagicMock(sort=record_sort)

    await loader.list_all_summaries_page(limit=5, cursor=cursor)

    query = mock_mongo_database["plugin_skills"].find.call_args.args[0]
    assert sorts == [list(MongoPluginSkillLoader.LISTING_SORT_KEYS)]
    equality_fields = {k for k, v in query.items() if k not in ("$or", "state") and not isinstance(v, dict)}
    assert equality_fields == {MongoPluginSkillLoader.SCHEMA_VERSION_FIELD}


async def test_ensure_indexes_is_idempotent_and_leaves_existing_indexes_alone(mock_mongo_database: MagicMock) -> None:
    loader = MongoPluginSkillLoader(database=mock_mongo_database)

    await loader.ensure_indexes()
    first = _index_calls(mock_mongo_database, "plugin_skills")
    await loader.ensure_indexes()

    skills = mock_mongo_database["plugin_skills"]
    assert _index_calls(mock_mongo_database, "plugin_skills") == first
    assert skills.create_index.await_count == 2 * len(first)
    assert {MongoPluginSkillLoader.INDEX_NAME, MongoPluginSkillLoader.PATH_INDEX_NAME} <= set(first)
    assert {MongoPluginSkillLoader.LISTING_INDEX_NAME, MongoPluginSkillLoader.LISTING_STATE_INDEX_NAME} <= set(first)
    assert first[MongoPluginSkillLoader.INDEX_NAME] == (
        [(MongoPluginSkillLoader.SCHEMA_VERSION_FIELD, 1), ("author", 1), ("plugin_name", 1), ("skill_name", 1)],
        True,
    )
    skills.drop_index.assert_not_called()


def _filtered_skills() -> list[dict[str, Any]]:
    return [
        _skill("a@example.com", "p1", "one", "published", "f1"),
        _skill("b@example.com", "p1", "one", "draft", "f1"),
        _skill("a@example.com", "p1", "two", "draft", "f2"),
        _skill("a@example.com", "p1", "three", "staging", "f1"),
        _skill("a@example.com", "p2", "one", "published", "f1"),
        _skill("a@example.com", "p2", "two", "in_review", None),
        _skill("system", "p3", "one", "draft", "f1"),
    ]


@pytest.mark.parametrize(
    ("filters", "include_staging", "expected"),
    [
        (
            {"plugin_name": "p1"},
            False,
            [("p1", "one", "a@example.com"), ("p1", "one", "b@example.com"), ("p1", "two", "a@example.com")],
        ),
        (
            {"folder": "f1"},
            False,
            [
                ("p1", "one", "a@example.com"),
                ("p1", "one", "b@example.com"),
                ("p2", "one", "a@example.com"),
                ("p3", "one", "system"),
            ],
        ),
        (
            {"allow_states": {"draft"}},
            False,
            [("p1", "one", "b@example.com"), ("p1", "two", "a@example.com"), ("p3", "one", "system")],
        ),
        (
            {"allow_states": {"draft", "in_review"}},
            False,
            [
                ("p1", "one", "b@example.com"),
                ("p1", "two", "a@example.com"),
                ("p2", "two", "a@example.com"),
                ("p3", "one", "system"),
            ],
        ),
        (
            {"plugin_name": "p1", "allow_states": {"draft"}},
            False,
            [("p1", "one", "b@example.com"), ("p1", "two", "a@example.com")],
        ),
        ({"plugin_name": "p1", "folder": "f1", "allow_states": {"draft"}}, False, [("p1", "one", "b@example.com")]),
        (
            {"plugin_name": "p1", "folder": "f1"},
            False,
            [("p1", "one", "a@example.com"), ("p1", "one", "b@example.com")],
        ),
        ({"allow_states": {"staging"}}, False, []),
        ({"allow_states": {"staging"}}, True, [("p1", "three", "a@example.com")]),
        (
            {"allow_states": {"staging", "draft"}},
            False,
            [("p1", "one", "b@example.com"), ("p1", "two", "a@example.com"), ("p3", "one", "system")],
        ),
        ({"allow_states": set()}, True, []),
        ({"plugin_name": "nope"}, False, []),
        (
            {},
            False,
            [
                ("p1", "one", "a@example.com"),
                ("p1", "one", "b@example.com"),
                ("p1", "two", "a@example.com"),
                ("p2", "one", "a@example.com"),
                ("p2", "two", "a@example.com"),
                ("p3", "one", "system"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("limit", [1, 2, 100])
async def test_filters_are_applied_in_the_query_with_exact_pages(
    mock_mongo_database: MagicMock,
    filters: dict[str, Any],
    include_staging: bool,
    limit: int,
    expected: list[tuple[str, str, str]],
) -> None:
    loader = _loader(mock_mongo_database, _filtered_skills())

    pages = await _collect(loader, limit=limit, include_staging=include_staging, **filters)

    assert [k for p in pages for k in _keys(p.summaries)] == expected
    assert all(len(p.summaries) == limit for p in pages[:-1])
    assert all(len(p.summaries) <= limit for p in pages)


async def test_empty_permitted_states_does_not_query(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, _filtered_skills())

    page = await loader.list_all_summaries_page(allow_states={"staging"})

    assert page.summaries == ()
    mock_mongo_database["plugin_skills"].find.assert_not_called()


@pytest.mark.parametrize(
    ("issued", "reused"),
    [
        ({"plugin_name": "p1"}, {"plugin_name": "p2"}),
        ({"plugin_name": "p1"}, {}),
        ({}, {"folder": "f1"}),
        ({"folder": "f1"}, {"folder": "f2"}),
        ({"allow_states": {"draft"}}, {"allow_states": {"published"}}),
        ({"allow_states": {"draft"}}, {}),
        ({"include_staging": False}, {"include_staging": True}),
    ],
)
async def test_cursor_is_rejected_under_different_filters(
    mock_mongo_database: MagicMock, issued: dict[str, Any], reused: dict[str, Any]
) -> None:
    loader = _loader(mock_mongo_database, _filtered_skills())
    first = await loader.list_all_summaries_page(limit=1, **issued)
    assert first.next_cursor is not None
    mock_mongo_database["plugin_skills"].find.reset_mock()

    with pytest.raises(InvalidPageCursorError):
        await loader.list_all_summaries_page(limit=1, cursor=first.next_cursor, **reused)

    mock_mongo_database["plugin_skills"].find.assert_not_called()


async def test_cursor_accepts_equivalent_filters_in_any_state_order(mock_mongo_database: MagicMock) -> None:
    loader = _loader(mock_mongo_database, _filtered_skills())
    first = await loader.list_all_summaries_page(limit=1, allow_states=["draft", "in_review"])

    second = await loader.list_all_summaries_page(
        limit=1, cursor=first.next_cursor, allow_states=("in_review", "draft")
    )

    assert len(second.summaries) == 1


async def test_service_passes_filters_through(mock_mongo_database: MagicMock) -> None:
    store = _loader(mock_mongo_database, _filtered_skills())
    service = ListSkillsService(
        skill_loader=CompositeSkillLoader(shared_loader=AsyncMock(spec=SkillLoaderProtocol), user_loader=store)
    )

    page = await service.execute_all_authors_page(
        user_id="admin@example.com", plugin_name="p1", folder="f1", allow_states={"draft"}
    )

    assert [(i.plugin_name, i.name, i.author, i.folder) for i in page.items] == [("p1", "one", "b@example.com", "f1")]


def test_oversized_cursor_is_rejected_before_decoding() -> None:
    oversized = "A" * (MAX_CURSOR_LENGTH + 1)

    with pytest.raises(InvalidPageCursorError):
        decode_cursor(cursor=oversized)


def test_bare_string_allow_states_is_rejected() -> None:
    with pytest.raises(TypeError):
        filters_fingerprint(include_staging=False, plugin_name=None, folder=None, allow_states="draft")


def test_deeply_nested_cursor_is_rejected_as_invalid_cursor() -> None:
    nested = base64.urlsafe_b64encode(b"[" * 200_000).decode("ascii")

    with pytest.raises(InvalidPageCursorError):
        decode_cursor(cursor=nested)


@pytest.mark.asyncio
async def test_null_store_rejects_invalid_cursor() -> None:
    with pytest.raises(InvalidPageCursorError):
        await NullPluginSkillStore().list_all_summaries_page(cursor="not-a-cursor")


def test_cursor_with_huge_key_part_is_rejected() -> None:
    huge = encode_cursor(key=SummaryPageKey(plugin_name="p" * 5_000_000, skill_name="s", author="a"))

    with pytest.raises(InvalidPageCursorError):
        decode_cursor(cursor=huge)
