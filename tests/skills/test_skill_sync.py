from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_ai_skills_framework.loaders.skill_sync import (
    SYSTEM_USER_ID,
    SkillSync,
)
from langchain_ai_skills_framework.models.mongo_plugin_skill_document import (
    MongoPluginSkillDocument,
)
from langchain_ai_skills_framework.models.skills_model import (
    SkillDetails,
    SkillSummary,
)


def _make_summary(name: str, source_path: Path | None = None) -> SkillSummary:
    return SkillSummary(
        name=name,
        description=f"Description for {name}",
        source_path=source_path or Path(f"/skills/{name}/SKILL.md"),
    )


def _make_details(name: str, content: str = "# Content") -> SkillDetails:
    summary = _make_summary(name)
    return SkillDetails(
        summary=summary,
        content=content,
        source_path=summary.source_path,
    )


def _make_shared_loader(
    summaries: list[SkillSummary] | None = None,
    details: dict[str, SkillDetails] | None = None,
    resource_names: dict[str, list[str]] | None = None,
    script_names: dict[str, list[str]] | None = None,
) -> MagicMock:
    loader = MagicMock()
    loader.list_skill_summaries.return_value = summaries or []
    _details = details or {}
    loader.get_skill_details.side_effect = lambda skill_name, *, plugin_name="": _details.get(
        skill_name, _make_details(skill_name)
    )
    _resources = resource_names or {}
    loader.list_skill_resource_names.side_effect = lambda skill_name, *, plugin_name="": _resources.get(skill_name, [])
    _scripts = script_names or {}
    loader.list_skill_script_names.side_effect = lambda skill_name, *, plugin_name="": _scripts.get(skill_name, [])
    loader.read_skill_resource.side_effect = lambda skill_name, resource_name, *, plugin_name="": (
        f"content of {resource_name}"
    )
    return loader


def _make_store() -> AsyncMock:
    store = AsyncMock()
    store.save_skill.return_value = MongoPluginSkillDocument(
        plugin_name="test-plugin",
        user_id=SYSTEM_USER_ID,
        skill_name="test",
        path="test-plugin/skills/test/SKILL.md",
        description="test",
        content="test",
        modified_by=SYSTEM_USER_ID,
    )
    store.set_skill_published.return_value = store.save_skill.return_value
    return store


class TestSkillSync:
    @pytest.mark.asyncio
    async def test_syncs_missing_skill(self) -> None:
        summaries = [_make_summary("my-skill")]
        shared = _make_shared_loader(summaries=summaries)
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.skills_added == 1
        store.save_skill.assert_awaited_once()
        call_kwargs = store.save_skill.call_args.kwargs
        assert call_kwargs["user_id"] == SYSTEM_USER_ID
        assert call_kwargs["skill_name"] == "my-skill"
        assert call_kwargs["modified_by"] == SYSTEM_USER_ID
        # Should be marked as shared
        store.set_skill_published.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upserts_existing_skill(self) -> None:
        """Sync always upserts — existing skills get replaced with latest content."""
        summaries = [_make_summary("my-skill")]
        shared = _make_shared_loader(summaries=summaries)
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.skills_added == 1
        store.save_skill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_syncs_missing_resources(self) -> None:
        summaries = [_make_summary("my-skill")]
        shared = _make_shared_loader(
            summaries=summaries,
            resource_names={"my-skill": ["FORMS.md", "REF.md"]},
        )
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.resources_added == 2
        assert store.save_resource.await_count == 2

    @pytest.mark.asyncio
    async def test_upserts_existing_resources(self) -> None:
        """Sync always upserts — existing resources get replaced with latest content."""
        summaries = [_make_summary("my-skill")]
        shared = _make_shared_loader(
            summaries=summaries,
            resource_names={"my-skill": ["FORMS.md"]},
        )
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.resources_added == 1
        store.save_resource.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_syncs_missing_scripts(self, tmp_path: Path) -> None:
        """Scripts in the ``scripts/`` subdirectory are found and synced."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# My Skill")
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        script_file = scripts_dir / "analyze.py"
        script_file.write_text("print('hello')")

        summary = _make_summary("my-skill", source_path=skill_md)
        details = SkillDetails(summary=summary, content="# My Skill", source_path=skill_md)
        shared = _make_shared_loader(
            summaries=[summary],
            details={"my-skill": details},
            script_names={"my-skill": ["analyze"]},
        )
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.scripts_added == 1
        call_kwargs = store.save_script.call_args.kwargs
        assert call_kwargs["script_name"] == "analyze"
        assert call_kwargs["content"] == "print('hello')"
        assert call_kwargs["modified_by"] == SYSTEM_USER_ID

    @pytest.mark.asyncio
    async def test_syncs_scripts_from_skill_root_fallback(self, tmp_path: Path) -> None:
        """Scripts at the skill root (legacy layout) are found via fallback."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# My Skill")
        script_file = skill_dir / "analyze.py"
        script_file.write_text("print('legacy')")

        summary = _make_summary("my-skill", source_path=skill_md)
        details = SkillDetails(summary=summary, content="# My Skill", source_path=skill_md)
        shared = _make_shared_loader(
            summaries=[summary],
            details={"my-skill": details},
            script_names={"my-skill": ["analyze"]},
        )
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.scripts_added == 1
        call_kwargs = store.save_script.call_args.kwargs
        assert call_kwargs["content"] == "print('legacy')"

    @pytest.mark.asyncio
    async def test_upserts_existing_scripts(self, tmp_path: Path) -> None:
        """Sync always upserts — existing scripts get replaced with latest content."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# My Skill")
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "analyze.py").write_text("print('hello')")

        summary = _make_summary("my-skill", source_path=skill_md)
        details = SkillDetails(summary=summary, content="# My Skill", source_path=skill_md)
        shared = _make_shared_loader(
            summaries=[summary],
            details={"my-skill": details},
            script_names={"my-skill": ["analyze"]},
        )
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.scripts_added == 1
        store.save_script.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_empty_skills(self) -> None:
        shared = _make_shared_loader(summaries=[])
        store = _make_store()
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.skills_added == 0

    @pytest.mark.asyncio
    async def test_continues_on_skill_error(self) -> None:
        summaries = [_make_summary("bad-skill"), _make_summary("good-skill")]
        shared = _make_shared_loader(summaries=summaries)
        store = _make_store()

        # First save_skill call raises, second succeeds
        store.save_skill.side_effect = [RuntimeError("db error"), store.save_skill.return_value]
        sync = SkillSync(shared_loader=shared, user_store=store)

        result = await sync.sync()

        assert result.errors == 1
        assert result.skills_added == 1
