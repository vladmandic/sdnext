# SD.Next: Project Guidelines

*NOTE: this document is provided in two copies as `AGENTS.md` and `.github/copilot-instructions.md`*.  

All instructions files are in `.github/instructions/`.  
All skill files are in `.github/skills`.  

General app structure is:
- **Python** backend server  
  Uses **Torch** for model inference, **FastAPI** for API routes and **Gradio** for creation of UI components.  
- **TypeScript**/**CSS** frontend  

## Instructions

This file contains general guidelines for contributing to the SD.Next codebase, including conventions, tools, and project structure. For more specific guidance on working with particular areas of the codebase, please refer to the instructions files linked below:
- [Core Runtime Guidelines](instructions/core.instructions.md): Use when editing Python core runtime code, startup flow, model loading, API internals, backend/device logic, or shared state in modules and pipelines. Keep logic changes narrow; end-to-end parameter changes may span modules, and each affected module and its reason should be listed in the final summary.
- [Processing Workflow Guidelines](instructions/processing.instructions.md): Use when changing generation processing parameters, processing objects, workflow dispatch, or UI-to-backend processing paths.
- [Model Loading Guidelines](instructions/model-loading.instructions.md): Use when changing model detection, loading, component downloads, cache routing, quantization, or offload behavior.
- [LoRA Integration Guidelines](instructions/lora.instructions.md): Use when changing LoRA discovery, metadata detection, prompt parsing, loading strategy, activation, weight application, or UI integration.
- [Scheduler Integration Guidelines](instructions/scheduler.instructions.md): Use when changing scheduler classes, sampler registration or configuration, scheduler compatibility, or scheduler option wiring.
- [Chat Feature Guidelines](instructions/chat.instructions.md): Use when changing the optional chat feature, including its API, command parsing, persistence, operations, or artifacts.
- [API Development Guidelines](instructions/api.instructions.md): Use when adding or changing FastAPI routes, request/response contracts, route registration, authentication, or API handlers.
- [UI And Frontend Guidelines](instructions/ui.instructions.md): Use when editing frontend UI code, JavaScript, HTML, CSS, localization files, or built-in UI extensions including modernui and kanvas.
- [Hint Typography Guidelines](instructions/hints.instructions.md): Use when editing hint text or other UI strings in the English source file `ui/locale/locale_en.json`; manual locale overrides are out of scope.
- [SDNQ Quantization Guidelines](instructions/sdnq.instructions.md): Use when changing SDNQ quantization, pre-quantized model loading, SDNQ attention, device placement, or quantized component behavior.

## Agent Guidelines

1. Verify the user instruction against relevant guidelines in this file and linked instruction files before proceeding.
2. If the instruction conflicts with any guideline, do not proceed. Explain which guideline(s) it conflicts with and how to adjust the instruction to comply.
3. Ask a follow-up only when unresolved ambiguity materially affects behavior, scope, or safety; otherwise make a reasonable assumption and state it.
4. When giving feedback, name the applicable guideline(s) and explain how each one applies.
5. If the instruction is clear and compliant, proceed and keep resulting changes aligned with project coding style, conventions, and structure.

## Language Guidelines

- Use clear and concise language when communicating with users, providing feedback, and explaining guidelines.
- Avoid unnecessary pleasantries or filler language; focus on the technical content and actionable feedback.
- When asking follow-up questions for clarification, be direct and specific about the information needed to proceed with the instruction while ensuring that the questions are relevant to the project guidelines and conventions.

## Tools

- `venv` for Python environment management, activated with `source venv/bin/activate` (Linux) or `venv\Scripts\activate` (Windows).  
  venv MUST be activated before running any Python commands or scripts to ensure correct dependencies and environment variables.  
- `python` 3.11+.
- `pyproject.toml` for Python configuration, including linting and type checking settings.
- `pnpm` for managing JavaScript dependencies and scripts, with key commands defined in `package.json`.

## Checks

- `pre-commit` hooks which checks line-endings and other formatting issues, configured in `.pre-commit-config.yaml` and executed with `pnpm precommit`.
- `codespell` for checking spelling errors in code and documentation, executed via `pnpm codespell`.
- `eslint` configured for both core and UI code and executed via `pnpm eslint:core`, `pnpm eslint:modernui` or `pnpm eslint:kanvas` (depending on module).
- `ruff` for Python linting, with configurations in `pyproject.toml` and executed via `pnpm ruff`.
- `pylint` for Python linting, with configurations in `pyproject.toml` and executed via `pnpm pylint`.
- `ty` for Python type checking, with configurations in `pyproject.toml` and executed via `pnpm ty`.
- `pyright` for Python type checking, with configurations in `pyproject.toml` and executed via `pnpm pyright`.

Full linting and typechecking suite is executed using `test/lint.sh`.  
SDNext initialization and startup is tested using `./webui.sh --test`.  

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

## Conventions

- Keep PR-ready changes targeted to `dev` branch.
- Use conventions from `CONTRIBUTING`.
- Do not include unrelated edits or submodule changes when preparing contributions.
- Do not inspect or modify files under `extensions/` unless the user explicitly asks; this restriction does not apply to `extensions-builtin/`.
- `ui/locale/locale_en.json` is the editable source of truth. Leave generated translated locale files (`ui/locale/locale_*.json`, except the English source) and manual overrides (`ui/locale/override_*.json`) untouched.
- Use existing CLI/API tool patterns in `cli/` and `test/` when adding automation scripts.
- Respect environment-driven behavior (`SD_*` flags and options) instead of hardcoding platform/model assumptions.
- For startup/init edits, preserve error handling and partial-failure tolerance in parallel scans and extension loading.

## File Creation

- Any temporary scripts or markdown reports must be stored in `tmp/` folder
- Any helper scripts, task execution scripts, or output capture that need temporary files should always use the repository-local `tmp/` folder
- Any reusable test scripts must be stored in `test/` folder

## Repo-Local Skills

Repo-local skills are indexed in `skills/README.md`; use that file as the canonical list. Update the index when adding or changing a skill.
