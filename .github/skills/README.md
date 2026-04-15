# Repo Skills

This folder contains repo-local Copilot skills for recurring SD.Next tasks.

## Available Skills

- `port-model`
  File: `port-model/SKILL.md`
  Use when adding or porting a model family into SD.Next and Diffusers.

- `debug-model`
  File: `debug-model/SKILL.md`
  Use when a new or existing SD.Next/Diffusers model integration fails during detect, load, prompt encode, sample, or output handling.

- `check-api`
  File: `check-api/SKILL.md`
  Use when auditing API endpoints in `modules/api/api.py` and delegated API modules for route parameter correctness plus request/response signature consistency.

- `check-schedulers`
  File: `check-schedulers/SKILL.md`
  Use when auditing scheduler registrations from `modules/sd_samplers_diffusers.py` to verify class loadability, config validity against scheduler capabilities, and `SamplerData` correctness.

- `check-models`
  File: `check-models/SKILL.md`
  Use when running an end-to-end model integration audit covering loaders, detect/routing parity, reference catalogs, and custom pipeline API contracts.

- `check-processing`
  File: `check-processing/SKILL.md`
  Use when validating txt2img/img2img/control processing workflows from UI submit definitions to backend execution with parameter, type, and initialization checks.

- `check-scripts`
  File: `check-scripts/SKILL.md`
  Use when auditing `scripts/*.py` for correct Script overrides (`__init__`, `title`, `show`) and verifying `ui()` output compatibility with `run()` or `process()` parameters.

- `github-issues`
  File: `github-issues/SKILL.md`
  Use when reading SD.Next GitHub issues with `[Issues]` in title and producing a markdown summary with status and suggested next steps for each issue.

- `github-features`
  File: `github-features/SKILL.md`
  Use when reading SD.Next GitHub issues with `[Feature]` in title and producing a markdown summary with status and suggested next steps for each issue.

- `analyze-model`
  File: `analyze-model/SKILL.md`
  Use when analyzing an external model URL to classify implementation style and estimate SD.Next porting difficulty before coding.

- `diffusers-code`
  File: `diffusers-code/SKILL.md`
  Use when creating or editing code to be compliant with Hugging Face diffusers conventions, including PR-ready change preparation for diffusers.

- `reference-catalog`
  File: `reference-catalog/SKILL.md`
  Use when maintaining and validating model reference catalogs in `data/reference*.json`, including duplicate checks and thumbnail alignment.

- `fix-lint`
  File: `fix-lint/SKILL.md`
  Use when running the full lint workflow in strict order and fixing issues as needed (`pre-commit`, `eslint`, `ruff`, `pylint`), while ignoring findings explicitly marked with `TODO`.

- `todo`
  File: `todo/SKILL.md`
  Use when scanning the full codebase for `TODO` markers and producing a markdown document with proposed next steps for each item.

## Notes

- Keep skills narrowly task-oriented and reusable.
- Prefer referencing existing repo patterns over generic framework advice.
- Update this index when adding new repo-local skills.
