# langchain-ai-skills-framework

## Overview
- langchain-ai-skills-framework loads Agent Skills from `SKILL.md` files and serves them via LangChain middleware and tools.
- `SkillDirectoryLoader` now uses `pydantic-ai-skills` registries for local filesystem and GitHub sources with TTL-based reload, and is surfaced through `SkillMiddleware` and `LoadSkillTool`.

## Skill authoring
- See `docs/skill-authoring.md` for required frontmatter, naming rules, and examples.

## GitHub authentication for remote skills
- When `SKILLS_DIRECTORY` uses `github://...`, provide a token via `SKILLS_GITHUB_TOKEN` (preferred) or `GITHUB_TOKEN` (fallback).
- Expected GitHub directory format: `github://<owner>:<repo>[@branch]/<path>` (for example: `github://my-org:private-skills/skills`).
- Supported token types: fine-grained Personal Access Token (PAT) and GitHub App installation token.
- Recommended usage:
  - Local development: fine-grained PAT scoped to the required repositories.
  - CI (GitHub Actions): workflow `GITHUB_TOKEN` when permissions are sufficient.
  - Long-running services: short-lived GitHub App installation tokens.

## Quick start
- `make init` – initialize the local dev environment.
- `make up` – start the dev container.
- `make run-pre-commit` – run lint/type/security suite.
- `make tests` – run dockerized pytest.
- `make build` – build sdist/wheel packages.
