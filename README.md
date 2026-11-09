# langchain-ai-skills-framework

## Overview
- langchain-ai-skills-framework loads Agent Skills from `SKILL.md` files and serves them via LangChain tools.
- Skills come from two sources:
  - **Plugin marketplace** (`MarketplaceDirectoryLoader`) — shared skills loaded from filesystem or GitHub, organized as `plugins/<plugin>/skills/<skill>/SKILL.md`.
  - **User-persisted skills** (`MongoPluginSkillLoader`) — stored in MongoDB across three collections (`plugin_skills`, `plugin_references`, `plugin_scripts`).
- `CompositeSkillLoader` merges both sources with precedence: user → shared DB → marketplace.
- All skill tools are scoped by `plugin_name`.

## Skill authoring
- See `docs/skill-authoring.md` for required frontmatter, naming rules, and examples.

## GitHub authentication for remote plugins
- When `PLUGINS_MARKETPLACE` uses `github://...`, provide a token via `SKILLS_GITHUB_TOKEN` (preferred) or `GITHUB_TOKEN` (fallback).
- `${ENV_VAR}` in a plugin's `.mcp.json` `url`, `headers`, `command`, `args` and `env` is expanded from the process environment at sync. Set `PLUGINS_MARKETPLACE_ENV_EXPANSION_INCLUDE` (comma-separated plugin names) to limit that to the listed plugins: any other plugin keeps the placeholder verbatim, so it cannot read a service secret by naming it. Unset means every plugin may expand (the default, unchanged); set to an empty string means none may. Plugins are matched by the name in the marketplace's `marketplace.json` (or the plugin directory name), so the allowlist is only as strong as write access to the marketplace: keep it operator-controlled. Names are matched the way `PLUGINS_MARKETPLACE_INCLUDE` matches them (case-insensitive; `_` and spaces equal `-`). Turning the allowlist on does not clear values already expanded and stored by earlier syncs (the `plugins` collection and the marketplace snapshot cache): call `reload_plugins` after changing it, and rotate any secret an unlisted plugin could have read.
- Expected GitHub directory format: `github://<owner>/<repo>/<path>?ref=<branch>` (for example: `github://my-org/ai-plugin-marketplace/plugins?ref=main`).
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
