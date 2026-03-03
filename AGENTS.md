# AGENTS.md

## Scope
- This file applies to the whole repository: `E:\1_LLM_PROJECT\multi-agent-design-patterns`.

## Communication
- Default response language: Chinese.
- Keep explanations concise and practical.
- For code changes, provide: what changed, why, and how to verify.

## Environment
- Python version: `>=3.13` (see `pyproject.toml`).
- Package/tooling preference: `uv`.

## Common Commands
- Sync dependencies: `uv sync`
- Run an example module: `uv run python <path_to_script.py>`
- Run tests (if tests exist): `uv run pytest -q`
- Lint/format (if configured): `uv run ruff check .` and `uv run ruff format .`

## Coding Rules
- Prefer minimal, targeted changes; avoid broad refactors unless requested.
- Follow existing project structure and naming conventions.
- Do not introduce secrets, tokens, or hardcoded credentials.
- Add comments only for non-obvious logic.
- Keep files ASCII unless non-ASCII is already required.

## Safety Boundaries
- Do not modify lockfiles or dependencies unless task requires it.
- Do not change CI/release/deployment files unless explicitly requested.
- Do not delete files or run destructive git commands unless explicitly requested.

## Validation
- For functional code changes:
  1. Run the smallest relevant check first.
  2. If available, run targeted tests before full test suite.
  3. Report commands run and key results.
- If local validation cannot run, clearly state why.

## Collaboration Preferences
- When requirements are ambiguous, make a reasonable assumption and proceed.
- Surface risks/tradeoffs early if they can affect correctness.
- Keep PR-style summaries short and actionable.
