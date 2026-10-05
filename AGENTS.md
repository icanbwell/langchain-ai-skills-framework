# AGENTS.md — langchain-ai-skills-framework

`CLAUDE.md` is a symlink to this file. Edit this file only.

## What this package is

A Python library (>=3.12) that loads Agent Skills (`SKILL.md` files) and serves them as LangChain tools. It is a
pinned dependency of other services, notably `baileyai` and `baileyai-skills-service`, which also share its MongoDB
collections. See `README.md` for the overview and `docs/skill-authoring.md` for skill frontmatter and naming rules.

Skills come from two sources, merged by `CompositeSkillLoader` with precedence user → shared DB → marketplace:

- **Plugin marketplace** (`MarketplaceDirectoryLoader`): shared skills from the filesystem or GitHub, laid out as
  `plugins/<plugin>/skills/<skill>/SKILL.md`.
- **User-persisted skills** (`MongoPluginSkillLoader`): MongoDB collections `plugin_skills`, `plugin_references`,
  `plugin_scripts`, plus `plugins` and `plugin_skill_usage`.

## Layout

| Path | Contents |
|---|---|
| `langchain_ai_skills_framework/loaders/` | Skill loaders, `PluginSkillStore` and `SkillLoaderProtocol`, `SkillSync` |
| `.../models/` | Pydantic models, including the Mongo document models and `SCHEMA_VERSION` |
| `.../services/` | One service per operation (save, load, list, delete, publish, run script, ...) |
| `.../langchain/tools/` | LangChain tool wrappers over the services; `tool_factory.py` builds them |
| `.../executors/` | Script executors (local, shell, AgentCore) |
| `.../persistence/` | Mongo database factory, history and error writers |
| `.../publishing/` | `GitHubMarketplacePublisher` |
| `.../container/`, `.../environment/` | Dependency-injection container and environment-variable settings |
| `.../startup.py` | `initialize_skills` and `reload_plugins` |
| `tests/` | Unit tests (Mongo is mocked); `tests_integration/` needs a real environment |

## Commands

Everything runs in Docker:

```bash
make init              # one-time local setup
make up / make down    # start / stop the dev container
make tests             # pytest tests + package
make tests-integration # pytest tests_integration
make run-pre-commit    # ruff, mypy (strict), bandit, detect-secrets, etc.
make uv.lock           # re-lock dependencies (also: make update-fast)
make build             # build sdist and wheel
```

The pre-commit hook runs in Docker and cannot see a git worktree's `.git` file, so it fails with
`FatalError: git failed` inside a worktree. Run `make run-pre-commit` from the main checkout, or run ruff and
pytest directly, and let CI run the full set.

## Conventions

- Line length is 120 (ruff). mypy runs in strict mode; the package ships `py.typed`.
- Commit messages and PR titles must start with a JIRA key (for example `BAI-965 feat: ...`), or `Bump`, `Merge`,
  `Revert`, or `Reapply`. Conventional-commit prefixes alone are rejected by `check-commit-message.yml`.
- Every PR needs one `risk:*`, one `type:*`, and one `semver:*` label before it can merge.
- A breaking change uses `feat!:` in the PR title and describes the consumer migration in the body.
- Releases are published to PyPI when a GitHub release is created. The release tag becomes the package version
  (`VERSION` in the repo is a placeholder).
- Tests are plain async pytest (`asyncio_mode = "auto"`). Mongo is faked or mocked in unit tests; use the
  `mock_mongo_database` fixture in `tests/conftest.py`.

## Version upgrades must be backwards compatible

A release must never make data or code written for the previous version stop working. Before opening a PR, check
each of these:

- **Schema bumps need a migration.** Reads filter by exact `schema_version`, so a bump to
  `MongoPluginSkillDocument.SCHEMA_VERSION` silently hides every user-authored document saved under the old version
  unless those documents are migrated. Ship the migration with the bump; if the new fields can't be defaulted by the
  model, compute them from stored data (as v3→v4 does for `digest` and `size`, derived from `content`).
- **Prefer additive changes.** New document fields get a default and the models keep `extra="ignore"`. Do not
  rename, remove, or change the type or meaning of a stored field. If you must, add the new field, migrate, and keep
  reading the old one for a release.
- **Never leave old and new pods incompatible during a rolling deploy.** Old pods keep writing the previous
  `schema_version` while new pods run. A change that makes either side's writes unreadable to the other needs a
  staged release.
- **Don't break public interfaces.** Adding a required method to `SkillLoaderProtocol`, `PluginSkillStore`, or
  another protocol, or changing a public signature, breaks external implementers under `mypy --strict` (as #67
  did). Add optional methods or defaulted parameters instead. If a break is unavoidable, make it a major version.
- **Unique indexes include `schema_version`.** Changing index keys or names needs a rollout-safe path through
  `_ensure_index`, and a migration must not violate them (never re-tag a document over a same-identity document
  already at the current version).
- **Test the upgrade, not just the new state.** Include a test that data stored under the previous version is still
  readable or migrated after your change.
