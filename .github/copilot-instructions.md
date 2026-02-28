# langchain_ai_skills_framework – Copilot Code Review Instructions

## Objectives
- Keep every change aligned with the langchain_ai_skills_framework library for loading and serving Agent Skills (SKILL.md) through LangChain tools and middleware.
- Preserve strict typing (mypy --strict), Ruff compliance, formatting, and pre-commit rules while using absolute imports across the repo.
- Protect skill content from accidental leakage in logs or examples; avoid committing secrets in test fixtures or sample skills.
- Maintain Docker Compose and Makefile workflows (`make up`, `make tests`, `make run-pre-commit`, `make build`, etc.).
- Provide direct, prioritized feedback for contributors with blocking issues called out before suggestions.

## Repository Context Summary
- **Stack**: Python 3.12, LangChain + langchain-core, Pydantic v2, PyYAML, Docker/Compose, Pipenv.
- **Key modules**:
  - `langchain_ai_skills_framework/skills/skill_loader.py` – SKILL.md discovery, parsing, validation, and caching.
  - `langchain_ai_skills_framework/skills/skills_model.py` – SkillSummary/SkillDetails dataclasses.
  - `langchain_ai_skills_framework/skills/skills_middleware.py` – LangChain middleware that injects skill summaries into system prompts.
  - `langchain_ai_skills_framework/skills/skills_tool.py` – LangChain tool to load a full skill on demand.
  - `langchain_ai_skills_framework/utilities/cache/skill_cache.py` – TTL-aware cache for skill snapshots.
  - `langchain_ai_skills_framework/utilities/logger/log_levels.py` – logging defaults and per-source levels.
- **Tests & fixtures**: `tests/skills/test_skill_loader.py`, `tests/skills/test_skills_middleware.py`, `tests/skills/test_skills_tool.py`.
- **Tooling**: Pipenv (`Pipfile`), Ruff/mypy/bandit via pre-commit, Docker Compose in `docker-compose.yml`, pytest config in `setup.cfg`.

## Code Style and Quality Rules
- Absolute imports only (e.g., `from langchain_ai_skills_framework.skills.skill_loader import SkillDirectoryLoader`). No relative imports within the project.
- Provide full type annotations (functions, class attrs, module-level constants). Avoid `Any`; use Protocols/dataclasses/TypedDicts when needed.
- Keep mypy strict and Ruff clean; do not add unchecked `# type: ignore` or blanket `noqa`.
- Use `SkillCache` for shared snapshots and avoid repeated filesystem scans in request paths.
- Logging uses the standard library with per-source levels from `langchain_ai_skills_framework.utilities.logger.log_levels.SRC_LOG_LEVELS`.
- Do not log full skill content or any secrets; inside `except` blocks prefer `logger.exception("context message")` to preserve stack traces.

## Review Focus Areas (in priority order)
1. **Skill Spec Validation (blocking)**
   - SKILL.md YAML frontmatter required (`name`, `description`), proper terminator, valid types.
   - Name normalization rules enforced (lowercase, hyphenated, matches directory).
   - `allowed-tools` is a space-delimited string; metadata is a string-to-string mapping.
2. **Architectural Consistency (blocking)**
   - Skill discovery stays in `SkillDirectoryLoader`; no ad hoc parsing elsewhere.
   - Middleware continues injecting skill summaries as a system message.
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
- Bypassing `SkillDirectoryLoader` or `SkillCache` for direct file access.
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
- Manage Python deps via Pipenv. If `Pipfile` changes, regenerate lockfile using `make Pipfile.lock`.
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
- **Skill format**: SKILL.md with YAML frontmatter parsed by `SkillDirectoryLoader`.
- **LangChain middleware**: `SkillMiddleware` injects `<available_skills>` into system prompts.
- **LangChain tool**: `LoadSkillTool` loads full skill content for agent use.

## Enforcement Checklist for Reviewers
- Imports use absolute `langchain_ai_skills_framework.*` paths; typing is complete and mypy-clean.
- Pre-commit hooks (Ruff, formatting, bandit, mypy) run successfully.
- Skill parsing/validation stays centralized in `SkillDirectoryLoader`.
- Cache usage is preserved and thread-safe.
- Tests run via `make tests`; new behavior has coverage.
- Logging avoids sensitive content and uses per-source log levels.
