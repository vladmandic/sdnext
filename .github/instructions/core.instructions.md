---
description: "Use when editing Python core runtime code, startup flow, model loading, API internals, backend/device logic, or shared state in modules and pipelines."
name: "Core Runtime Guidelines"
applyTo: "launch.py, webui.py, installer.py, modules/**/*.py, pipelines/**/*.py, scripts/**/*.py, extensions-builtin/**/*.py"
---
# Core Runtime Guidelines

- Preserve startup ordering and import timing in `launch.py` and `webui.py`; avoid moving initialization steps unless required.
- Treat `modules/shared.py` as the source of truth for global runtime state (`shared.opts`, model references, backend/device flags).
- Prefer narrow changes with explicit side effects; avoid introducing new cross-module mutable globals.
- Keep platform paths neutral: do not assume CUDA-only behavior and preserve ROCm/IPEX/DirectML/OpenVINO compatibility branches.
- Keep extension and script loading resilient: when adding startup scans/hooks, preserve partial-failure tolerance and logging.
- Follow existing API/server patterns under `modules/api/` and reuse shared queue/state helpers rather than ad-hoc request handling.
- Reuse established model-loading and pipeline patterns (`modules/sd_*`, `pipelines/`) instead of creating parallel abstractions.
- For substantial Python changes, run at least relevant checks: `npm run ruff` and `npm run pylint` (or narrower equivalents when appropriate).
