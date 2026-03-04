# Project Guidelines

## Tools
- `pyproject.toml` for Python configuration, including linting and type checking settings.
- `venv` for Python environment management, activated with `source venv/bin/activate`.
- `python` 3.10+.
- `eslint` configured for both core and UI code.
- `pnpm` for managing JavaScript dependencies and scripts, with key commands defined in `package.json`.
- `ruff` and `pylint` for Python linting, with configurations in `pyproject.toml` and executed via `pnpm ruff` and `pnpm pylint`.
- `pre-commit` hooks which also check line-endings and other formatting issues, configured in `.pre-commit-config.yaml`.

## Project Structure
- Entry/startup flow: `webui.sh` -> `launch.py` -> `webui.py` -> modules under `modules/`.
- Core runtime state is centralized in `modules/shared.py` (shared.opts, model state, backend/device state).
- API/server routes are under `modules/api/`.
- Model and pipeline logic is split between `modules/sd_*` and `pipelines/`.
- Additional plug-ins live in `scripts/` and are used only when specified.
- Extensions live in `extensions-builtin/` and `extensions/` and are loaded dynamically.
- Tests and CLI scripts are under `test/` and `cli/`, with some API smoke checks in `test/full-test.sh`.

## Code Style
- Prefer existing project patterns over strict generic style rules;  
  this codebase intentionally allows patterns often flagged in default linters such as allowing long lines, etc.

## Build And Test
- Activate environment: `source venv/bin/activate` (always ensure this is active when working with Python code).
- Test startup: `python launch.py --test`
- Full startup: `python launch.py`
- Full lint sequence: `pnpm lint`
- Python checks individually: `pnpm ruff`, `pnpm pylint`
- JS checks: `pnpm eslint` and `pnpm eslint-ui`

## Conventions
- Keep PR-ready changes targeted to `dev` branch conventions from `CONTRIBUTING`.
- Do not include unrelated edits or submodule changes when preparing contributions.
- Use existing CLI/API tool patterns in `cli/` and `test/` when adding automation scripts.
- Respect environment-driven behavior (`SD_*` flags and options) instead of hardcoding platform/model assumptions.
- For startup/init edits, preserve error handling and partial-failure tolerance in parallel scans and extension loading.

## Pitfalls
- Initialization order matters: startup paths in `launch.py` and `webui.py` are sensitive to import/load timing.
- Shared mutable global state can create subtle regressions; prefer narrow, explicit changes.
- Device/backend-specific code paths (CUDA/ROCm/IPEX/DirectML/OpenVINO) should not assume one platform.
- Extension loading is dynamic; failures may appear only when specific extensions or models are present.
