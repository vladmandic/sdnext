# SD.Next: AGENTS.md Project Guidelines

SD.Next is a complex codebase with specific patterns and conventions.
General app structure is:
- Python backend server  
  Uses Torch for model inference, FastAPI for API routes and Gradio for creation of UI components.  
- JavaScript/CSS frontend  

## Tools

- `venv` for Python environment management, activated with `source venv/bin/activate` (Linux) or `venv\Scripts\activate` (Windows).  
  venv MUST be activated before running any Python commands or scripts to ensure correct dependencies and environment variables.  
- `python` 3.10+.
- `pyproject.toml` for Python configuration, including linting and type checking settings.
- `eslint` configured for both core and UI code.
- `pnpm` for managing JavaScript dependencies and scripts, with key commands defined in `package.json`.
- `ruff` and `pylint` for Python linting, with configurations in `pyproject.toml` and executed via `pnpm ruff` and `pnpm pylint`.
- `pre-commit` hooks which also check line-endings and other formatting issues, configured in `.pre-commit-config.yaml`.

## Project Structure

- Entry/startup flow: `webui.sh` -> `launch.py` -> `webui.py` -> modules under `modules/`.
- Install: `installer.py` takes care of installing dependencies and setting up the environment.
- Core runtime state is centralized in `modules/shared.py` (shared.opts, model state, backend/device state).
- API/server routes are under `modules/api/`.
- UI codebase is split between base JS in `javascript/` and actual UI in `extensions-builtin/sdnext-modernui/`.
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

- Keep PR-ready changes targeted to `dev` branch.
- Use conventions from `CONTRIBUTING`.
- Do not include unrelated edits or submodule changes when preparing contributions.
- Use existing CLI/API tool patterns in `cli/` and `test/` when adding automation scripts.
- Respect environment-driven behavior (`SD_*` flags and options) instead of hardcoding platform/model assumptions.
- For startup/init edits, preserve error handling and partial-failure tolerance in parallel scans and extension loading.

## Pitfalls

- Initialization order matters: startup paths in `launch.py` and `webui.py` are sensitive to import/load timing.
- Shared mutable global state can create subtle regressions; prefer narrow, explicit changes.
- Device/backend-specific code paths (**CUDA/ROCm/IPEX/DirectML/OpenVINO**) should not assume one platform.
- Scripts and extension loading is dynamic; failures may appear only when specific extensions or models are present.
