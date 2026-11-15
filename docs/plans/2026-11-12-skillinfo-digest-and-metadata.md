# SkillInfo digest and metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** `SkillInfo` (returned by `ListSkillsService`) carries each skill's `digest` and frontmatter `metadata`, so baileyai-skills-service can expose them on `/api/skills`.

**Architecture:** `SkillSummary` already has both fields (BAI-1514, released in 2.5.0). `ListSkillsService.execute` builds `SkillInfo` from summaries and drops them. Add two optional fields and copy them across. Additive, defaults `None`/empty.

**Tech Stack:** Python 3.12, frozen dataclasses, pytest.

**Spec:** baileyai `docs/superpowers/specs/2026-11-03-skill-result-cache-design.md` (section 2a). Master plan: baileyai `docs/superpowers/plans/2026-11-12-skill-result-cache.md`.

**Scope boundary:** this repo only. The service projection is in baileyai-skills-service; the cache is in baileyai.

## Global Constraints

- `metadata` stays internal to the framework and the skills service (it contains `user_id`); consumers must not forward it to clients.

## Review Focus

1. A skill with no stored digest (marketplace-directory skills, old content): `SkillInfo.digest` is `None`, not an error.
2. A skill with no frontmatter metadata: `SkillInfo.metadata` is an empty mapping.
3. The existing `author` derivation from `metadata["user_id"]` is unchanged.

---

### Task 1: Add the fields and populate them

**Files:** Modify `langchain_ai_skills_framework/services/list_skills_service.py` (`SkillInfo` at line 20, `execute` at line 87); Test `tests/services/test_list_skills_service.py`.

**Produces:** `SkillInfo.digest: str | None = None`, `SkillInfo.metadata: Mapping[str, object] = field(default_factory=dict)`.

- [ ] **Step 1: Failing tests:** `test_skill_info_carries_digest_and_metadata_from_summary`; `test_skill_without_digest_has_none` ; `test_skill_without_metadata_has_empty_mapping`; `test_author_still_derived_from_user_id_metadata`. Build summaries through the existing fake loader used in that file.
- [ ] **Step 2:** Run `make tests` (or the single file in the dev container) — expect FAIL.
- [ ] **Step 3:** Add the two fields to `SkillInfo`; in `execute` pass `digest=s.digest, metadata=s.metadata`. Check `list_all_authors_skills`-style callers and `tests/services/test_list_skills_all_authors.py` still pass.
- [ ] **Step 4:** PASS. Commit: `BAI-1499 expose digest and metadata on SkillInfo`.

### Task 2: Release

- [ ] **Step 1:** Merge via PR, then release a new minor version through the repo's release flow (the 2.5.0 tag is the precedent; `VERSION` is `0.0.1`, so confirm how releases are cut before tagging).
- [ ] **Step 2:** Confirm the new version is available from the JFrog virtual PyPI before baileyai-skills-service bumps to it.
