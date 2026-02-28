# langchain-ai-skills-framework

## Overview
- langchain-ai-skills-framework loads Agent Skills from `SKILL.md` files and serves them via LangChain middleware and tools.
- Skills are parsed and validated by `SkillDirectoryLoader`, cached with `SkillCache`, and surfaced through `SkillMiddleware` and `LoadSkillTool`.

## Skill authoring
- See `docs/skill-authoring.md` for required frontmatter, naming rules, and examples.

## Quick start
- `make init` – initialize the local dev environment.
- `make up` – start the dev container.
- `make run-pre-commit` – run lint/type/security suite.
- `make tests` – run dockerized pytest.
- `make build` – build sdist/wheel packages.
