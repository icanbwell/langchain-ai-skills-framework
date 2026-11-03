"""MongoPluginSkillLoader.set_skill_state persists ``modified_by`` only when given (BAI-1500)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.loaders.mongo_plugin_skill_loader import MongoPluginSkillLoader
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import MongoPluginSkillDocument


def _doc() -> MongoPluginSkillDocument:
    return MongoPluginSkillDocument(
        plugin_name="p",
        author="author-a@example.com",
        skill_name="my-skill",
        path="p/skills/my-skill/SKILL.md",
        description="d",
        content="# c",
        modified_by="author-a@example.com",
    )


@pytest.mark.parametrize(
    ("modified_by", "expect_field"),
    [("admin@example.com", True), ("", False)],
)
async def test_modified_by_is_persisted_only_when_given(
    mock_mongo_database: MagicMock, modified_by: str, expect_field: bool
) -> None:
    mock_mongo_database["plugin_skills"].find_one_and_update = AsyncMock(return_value=_doc().to_mongo_dict())
    loader = MongoPluginSkillLoader(database=mock_mongo_database)

    await loader.set_skill_state(
        author="author-a@example.com",
        plugin_name="p",
        skill_name="my-skill",
        state="draft",
        modified_by=modified_by,
    )

    update_fields = mock_mongo_database["plugin_skills"].find_one_and_update.call_args.args[1]["$set"]
    assert ("modified_by" in update_fields) is expect_field
    if expect_field:
        assert update_fields["modified_by"] == modified_by
