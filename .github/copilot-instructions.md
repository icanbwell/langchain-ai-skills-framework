# langchain_ai_skills_framework – Copilot Code Review Instructions

## Objectives
- Keep every change aligned with the langchain_ai_skills_framework library for loading and serving Agent Skills (SKILL.md) through LangChain tools and middleware.
- Preserve strict typing (mypy --strict), Ruff compliance, formatting, and pre-commit rules while using absolute imports across the repo.
- Protect skill content from accidental leakage in logs or examples; avoid committing secrets in test fixtures or sample skills.
- Maintain Docker Compose and Makefile workflows (`make up`, `make tests`, `make run-pre-commit`, `make build`, etc.).
- Provide direct, prioritized feedback for contributors with blocking issues called out before suggestions.

## Repository Context Summary
- **Stack**: Python 3.12, LangChain + langchain-core, Pydantic v2, PyYAML, Docker/Compose, uv.
- **Key modules**:
  - `langchain_ai_skills_framework/loaders/marketplace_directory_loader.py` – Plugin marketplace SKILL.md discovery, parsing, validation, and caching.
  - `langchain_ai_skills_framework/loaders/composite_skill_loader.py` – Merges marketplace (shared) skills with user-persisted MongoDB skills.
  - `langchain_ai_skills_framework/loaders/mongo_plugin_skill_loader.py` – MongoDB-backed store for user-persisted skills, resources, and scripts.
  - `langchain_ai_skills_framework/loaders/plugin_skill_store.py` – Protocol for the plugin skill store.
  - `langchain_ai_skills_framework/models/skills_model.py` – SkillSummary/SkillDetails/SkillSnapshot dataclasses.
  - `langchain_ai_skills_framework/tools/` – LangChain tools (load_skill, save_skill, list_skills, etc.) scoped by `plugin_name`.
  - `langchain_ai_skills_framework/utilities/logger/log_levels.py` – logging defaults and per-source levels.
- **Tests & fixtures**: `tests/skills/` directory.
- **Tooling**: uv (`pyproject.toml`), Ruff/mypy/bandit via pre-commit, Docker Compose in `docker-compose.yml`, pytest config in `pyproject.toml`.

## Code Style and Quality Rules
- Absolute imports only (e.g., `from langchain_ai_skills_framework.loaders.composite_skill_loader import CompositeSkillLoader`). No relative imports within the project.
- Provide full type annotations (functions, class attrs, module-level constants). Avoid `Any`; use Protocols/dataclasses/TypedDicts when needed.
- Keep mypy strict and Ruff clean; do not add unchecked `# type: ignore` or blanket `noqa`.
- Use `MarketplaceDirectoryLoader`'s snapshot caching for shared skills. User-persisted skills in MongoDB (`MongoPluginSkillLoader`) are read directly without caching.
- Logging uses the standard library with per-source levels from `langchain_ai_skills_framework.utilities.logger.log_levels.SRC_LOG_LEVELS`.
- Do not log full skill content or any secrets; inside `except` blocks prefer `logger.exception("context message")` to preserve stack traces.

## Review Focus Areas (in priority order)
1. **Skill Spec Validation (blocking)**
   - SKILL.md YAML frontmatter required (`name`, `description`), proper terminator, valid types.
   - Name normalization rules enforced (lowercase, hyphenated, matches directory).
   - `allowed-tools` is a space-delimited string; metadata is a string-to-string mapping.
2. **Architectural Consistency (blocking)**
   - Skill discovery stays in `MarketplaceDirectoryLoader` (shared) and `MongoPluginSkillLoader` (user-persisted); no ad hoc parsing elsewhere.
   - `CompositeSkillLoader` merges both sources with precedence: user → shared DB → marketplace.
   - All skill tools require `plugin_name` to scope operations to a specific plugin.
   - `LoadSkillTool` remains the supported entry point for loading skill content in agents.
3. **Type Safety & Linting (blocking)**
   - mypy strict and Ruff clean; no new unchecked `type: ignore`.
4. **Testing & Reliability (blocking)**
   - Tests runnable via `make tests`; new logic covered in `tests/skills/`.
   - Cache refresh logic remains deterministic and thread-safe.
5. **Performance & Resource Use (block if severe)**
   - Skill scans are cached; no redundant directory walks in hot paths.
6. **Documentation & DX (non-blocking but expected when workflows change)**
   - Update `README.md` when public APIs, skill format, or workflows change.

## Blocking Issues (must fix before merge)
- Relative imports, missing type hints, or mypy/Ruff failures.
- Bypassing `MarketplaceDirectoryLoader` or `CompositeSkillLoader` for direct file/DB access.
- Weakening skill validation rules (frontmatter, naming, metadata, allowed-tools).
- Tests not runnable via `make tests`, or new logic lacking coverage.
- Secrets committed to the repo (credentials, tokens, private keys).

## Non-Blocking Suggestions (nice to have)
- Refactors that improve clarity of skill parsing, prompt formatting, or cache usage.
- Additional tests for edge cases in frontmatter parsing or middleware insertion order.
- Small documentation improvements for skill authoring or tool usage.

## Security & Privacy Guidelines
- Avoid logging skill bodies or user-provided content that may include sensitive data.
- Mask identifiers in logs when possible; keep logs to metadata and counts.
- Never commit real credentials or private keys in examples or tests.

## Performance Guidelines
- Reuse cached snapshots where possible; only refresh when needed.
- Keep skill summaries compact when building prompt additions.

## Testing Guidelines
- Run `make tests` (dockerized pytest) before submitting.
- Prefer pytest fixtures and stubs in `tests/skills/`; avoid global monkeypatching.
- Keep async middleware tests deterministic with `pytest-asyncio`.

## Dependencies & Build
- Manage Python deps via uv. If `pyproject.toml` changes, regenerate lockfile using `make uv.lock`.
- Pre-commit hooks live in `pre-commit-hook`; run `make setup-pre-commit` once per clone.
- Build and publish tasks use `make build`, `make testpackage`, and `make package`.

## Documentation & Examples
- Update `README.md` when adding public APIs or changing the SKILL.md schema.
- Docstrings for tools/middleware should describe purpose, inputs, outputs, and example usage.

## Tone & Feedback Style
- Highlight blocking issues first, referencing file paths and line numbers where possible.
- Follow up with suggestions only after blockers.
- Keep feedback concise, specific, and actionable; include command/file hints but avoid large code dumps.

## Decision Authority & Constraints
- Absolute imports, strict typing, and pre-commit workflows are non-negotiable.
- Keep skill validation strict to avoid malformed metadata entering runtime usage.
- When uncertain, choose smaller, well-tested changes and request clarification.

## Quick Start & Common Commands
- Initial setup:
  - `make init`
- Daily workflow:
  - `make up` – start the dev container.
  - `make run-pre-commit` – run lint/type/security suite locally.
  - `make tests` – run dockerized pytest.
  - `make shell` – drop into the dev container shell.
- Packaging:
  - `make build` – build sdist/wheel.
  - `make testpackage` / `make package` – upload to TestPyPI/PyPI.

## Integration Points
- **Skill format**: SKILL.md with YAML frontmatter, organized in plugin marketplace structure (`plugins/<plugin>/skills/<skill>/SKILL.md`).
- **Shared skills**: `MarketplaceDirectoryLoader` loads from filesystem/GitHub with L1/L2 caching.
- **User-persisted skills**: `MongoPluginSkillLoader` stores in three MongoDB collections (`plugin_skills`, `plugin_references`, `plugin_scripts`).
- **Composite loader**: `CompositeSkillLoader` merges both sources and provides tools.
- **LangChain tools**: `LoadSkillTool`, `SaveSkillTool`, `ListSkillsTool`, etc. — all scoped by `plugin_name`.

## Enforcement Checklist for Reviewers
- Imports use absolute `langchain_ai_skills_framework.*` paths; typing is complete and mypy-clean.
- Pre-commit hooks (Ruff, formatting, bandit, mypy) run successfully.
- Skill parsing/validation stays centralized in `MarketplaceDirectoryLoader` and `MongoPluginSkillLoader`.
- Cache usage is preserved and thread-safe.
- Tests run via `make tests`; new behavior has coverage.
- Logging avoids sensitive content and uses per-source log levels.
