"""The one rule for passing the acting user to stores and publishers."""

import pytest

from langchain_ai_skills_framework.services.skill_operation_error import OwnerAndActor, acting_kwargs


@pytest.mark.parametrize(
    ("owner", "actor", "key", "expected"),
    [
        ("alice", "alice", "modified_by", {}),
        ("alice", "bob", "modified_by", {"modified_by": "bob"}),
        ("alice", "alice", "acting_user_id", {}),
        ("alice", "bob", "acting_user_id", {"acting_user_id": "bob"}),
    ],
)
def test_acting_kwargs_only_when_acting_for_another_user(
    owner: str, actor: str, key: str, expected: dict[str, str]
) -> None:
    assert acting_kwargs(OwnerAndActor(owner=owner, actor=actor), key=key) == expected
