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

## File Creation

- Any temporary scripts or markdown reports must be stored in `tmp/` folder
- Any reusable test scripts must be stored in `test/` folder

## Repo-Local Skills

All skills are defined in `.github/skills/` and indexed in `.github/skills/README.md`.
Use these repo-local skills for recurring SD.Next model integration work:

- `port-model`  
  File: `.github/skills/port-model/SKILL.md`  
  Use when adding a new model family, porting a standalone script into a Diffusers pipeline, or wiring an upstream Diffusers model into SD.Next.

- `debug-model`  
  File: `.github/skills/debug-model/SKILL.md`  
  Use when a new or existing SD.Next/Diffusers model integration fails during detection, loading, prompt encoding, sampling, or output handling.

- `check-api`  
  File: `.github/skills/check-api/SKILL.md`  
  Use when auditing API routes in `modules/api/api.py` and validating endpoint parameters plus request/response signatures.

- `check-schedulers`  
  File: `.github/skills/check-schedulers/SKILL.md`  
  Use when auditing scheduler registrations in `modules/sd_samplers_diffusers.py` for class loadability, config validity, and `SamplerData` mapping correctness.

- `check-models`  
  File: `.github/skills/check-models/SKILL.md`  
  Use when running end-to-end model integration audits across loaders, detect/routing parity, reference catalogs, and pipeline API contracts.

- `check-processing`  
  File: `.github/skills/check-processing/SKILL.md`  
  Use when validating txt2img/img2img/control processing workflows from UI submit definitions through backend processing and Diffusers execution, including parameter, type, and initialization checks.

- `check-scripts`  
  File: `.github/skills/check-scripts/SKILL.md`  
  Use when auditing scripts in `scripts/*.py` for standard `Script` overrides (`__init__`, `title`, `show`) and validating `ui()` output against `run()` or `process()` parameters.

- `github-issues`  
  File: `.github/skills/github-issues/SKILL.md`  
  Use when reading SD.Next GitHub issues with `[Issues]` in title and producing a markdown report with short summary, status, and suggested next steps for each issue.

- `github-features`  
  File: `.github/skills/github-features/SKILL.md`  
  Use when reading SD.Next GitHub issues with `[Feature]` in title and producing a markdown report with short summary, status, and suggested next steps for each issue.

- `analyze-model`  
  File: `.github/skills/analyze-model/SKILL.md`  
  Use when analyzing an external model URL to identify implementation style and estimate how difficult it is to port into SD.Next.

- `diffusers-code`  
  File: `.github/skills/diffusers-code/SKILL.md`  
  Use when creating or editing code that must comply with Hugging Face diffusers conventions, including preparing PR-ready changes targeting diffusers.

- `reference-catalog`  
  File: `.github/skills/reference-catalog/SKILL.md`  
  Use when maintaining and validating model reference catalogs in `data/reference*.json`, including duplicate checks and thumbnail alignment.

- `fix-lint`  
  File: `.github/skills/fix-lint/SKILL.md`  
  Use when running the full lint workflow in required order (`pre-commit`, `eslint`, `ruff`, `pylint`) and fixing findings as needed, while ignoring lint issues explicitly marked with `TODO`.

- `todo`  
  File: `.github/skills/todo/SKILL.md`  
  Use when searching the full codebase for `TODO` markers and producing a markdown document with proposed next steps for each item.

- `update-docs`  
  File: `.github/skills/update-docs/SKILL.md`  
  Use when reading markdown files from `wiki/` to correct markdown syntax, improve readability, and optionally normalize structure, links, and terminology while preserving technical meaning.

When creating and updating skills, update this file and the index in `.github/skills/README.md` accordingly.
