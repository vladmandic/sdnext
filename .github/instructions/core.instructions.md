---
description: "Use when editing Python core runtime code, startup flow, model loading, API internals, backend/device logic, or shared state in modules and pipelines."
name: "Core Runtime Guidelines"
applyTo: "launch.py, webui.py, installer.py, modules/**/*.py, pipelines/**/*.py, scripts/**/*.py, extensions-builtin/**/*.py, cli/**/*.py"
---
## Agent Guidelines

1. Verify the user instruction against relevant guidelines in this file and linked instruction files before proceeding.
2. If the instruction conflicts with any guideline, do not proceed. Explain which guideline(s) it conflicts with and how to adjust the instruction to comply.
3. If the instruction is valid but unclear or incomplete, ask targeted follow-up questions before implementation. Do not assume user intent or requirements.
4. When giving feedback, name the applicable guideline(s) and explain how each one applies.
5. If the instruction is clear and compliant, proceed and keep resulting changes aligned with project coding style, conventions, and structure.

## Language Guidelines

- Use clear and concise language when communicating with users, providing feedback, and explaining guidelines.
- Avoid unnecessary pleasantries or filler language; focus on the technical content and actionable feedback.
- When asking follow-up questions for clarification, be direct and specific about the information needed to proceed with the instruction while ensuring that the questions are relevant to the project guidelines and conventions.

# Core Runtime Guidelines

1. Preserve startup ordering and import timing in `launch.py` and `webui.py`; avoid moving initialization steps unless required to fix a critical startup bug or implement a new startup feature.
2. Treat `modules/shared.py` as the source of truth for global runtime state (`shared.opts`, model references, backend/device flags).
3. Prefer narrow changes (changes scoped to a single function or module when feasible) with explicit side effects; avoid introducing new cross-module mutable globals.
4. Keep platform paths neutral: do not assume CUDA-only behavior and preserve ROCm/IPEX/DirectML/OpenVINO compatibility branches.
5. Keep extension and script loading resilient: when adding startup scans/hooks, preserve partial-failure tolerance and logging.
6. Follow existing API/server patterns under `modules/api/` and reuse shared queue/state helpers rather than ad-hoc request handling.
7. Reuse established model-loading and pipeline patterns (`modules/sd_*`, `pipelines/`) instead of creating parallel abstractions.
8. For substantial Python changes, run at least relevant checks: `pnpm run ruff` and `pnpm run pylint` (or narrower equivalents when appropriate).

## Tools

- `venv` for Python environment management, activated with `source venv/bin/activate` (Linux) or `venv\Scripts\activate` (Windows).  
  venv MUST be activated before running any Python commands or scripts to ensure correct dependencies and environment variables.  
- `python` 3.10+.
- `pyproject.toml` for Python configuration, including linting and type checking settings.
- `pnpm` for managing JavaScript dependencies and scripts, with key commands defined in `package.json`.
- `ruff` and `pylint` for Python linting, with configurations in `pyproject.toml` and executed via `pnpm ruff` and `pnpm pylint`.
- `pre-commit` hooks which also check line-endings and other formatting issues, configured in `.pre-commit-config.yaml`.
- When writing helper scripts or capturing temporary output/files for a task, always use the repository-local `tmp/` folder.

## Build And Test

- Activate environment: `source venv/bin/activate` (always ensure this is active when working with Python code).
- Test startup: `python launch.py --test`
- Full startup: `python launch.py`
- Full lint sequence: `pnpm lint`
- Python checks individually: `pnpm ruff`, `pnpm pylint`

## Pitfalls

- Initialization order matters: startup paths in `launch.py` and `webui.py` are sensitive to import/load timing.
- Shared mutable global state can create subtle regressions; prefer narrow, explicit changes.
- Device/backend-specific code paths (**CUDA/ROCm/IPEX/DirectML/OpenVINO**) should not assume one platform.
- Scripts and extension loading is dynamic; failures may appear only when specific extensions or models are present.
