---
description: "Use when editing Python core runtime code, startup flow, model loading, API internals, backend/device logic, or shared state in modules and pipelines."
name: "Core Runtime Guidelines"
applyTo: "launch.py, webui.py, installer.py, modules/**/*.py, pipelines/**/*.py, scripts/**/*.py, extensions-builtin/**/*.py"
---
# Core Runtime Guidelines

1. Preserve startup ordering and import timing in `launch.py` and `webui.py`; avoid moving initialization steps unless required to fix a critical startup bug or implement a new startup feature.
2. Treat `modules/shared.py` as the source of truth for global runtime state (`shared.opts`, model references, backend/device flags).
3. Prefer narrow changes (changes scoped to a single function or module when feasible) with explicit side effects; avoid introducing new cross-module mutable globals.
4. Keep platform paths neutral: do not assume CUDA-only behavior and preserve ROCm/IPEX/DirectML/OpenVINO compatibility branches.
5. Keep extension and script loading resilient: when adding startup scans/hooks, preserve partial-failure tolerance and logging.
6. Follow existing API/server patterns under `modules/api/` and reuse shared queue/state helpers rather than ad-hoc request handling.
7. Reuse established model-loading and pipeline patterns (`modules/sd_*`, `pipelines/`) instead of creating parallel abstractions.
8. For substantial Python changes, run at least relevant checks: `npm run ruff` and `npm run pylint` (or narrower equivalents when appropriate).

## Build And Test

- Activate environment: `source venv/bin/activate` (always ensure this is active when working with Python code).
- Test startup: `python launch.py --test`
- Full startup: `python launch.py`
- Full lint sequence: `pnpm lint`
- Python checks individually: `pnpm ruff`, `pnpm pylint`
- TypeScript checks: `pnpm eslint`, `pnpm tsc`

## Pitfalls

- Initialization order matters: startup paths in `launch.py` and `webui.py` are sensitive to import/load timing.
- Shared mutable global state can create subtle regressions; prefer narrow, explicit changes.
- Device/backend-specific code paths (**CUDA/ROCm/IPEX/DirectML/OpenVINO**) should not assume one platform.
- Scripts and extension loading is dynamic; failures may appear only when specific extensions or models are present.
