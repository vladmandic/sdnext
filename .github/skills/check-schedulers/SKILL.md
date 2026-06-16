---
name: check-schedulers
description: "Run a phased scheduler audit from modules/sd_samplers_diffusers.py and scheduler UI definitions: verify class loadability first, then config validity against scheduler capabilities, then SamplerData correctness and UI option alignment."
argument-hint: "Optionally focus on a scheduler subset, such as flow-matching, res4lyf, or parallel schedulers"
---

# Check Scheduler Registry And Contracts

Use `modules/sd_samplers_diffusers.py` as the starting point and verify that scheduler classes, scheduler config, and `SamplerData` mappings are coherent and executable.

## Required Guarantees

The audit must explicitly verify all four:

1. All scheduler classes can be loaded and compiled.
2. All scheduler config entries are valid and match scheduler capabilities in `__init__`.
3. Scheduler-related UI option values are valid and consistent with the runtime scheduler path.
4. All scheduler classes have valid associated `SamplerData` entries and mapping correctness.

## Scope

Primary file:

- `modules/sd_samplers_diffusers.py`

Related files:

- `modules/ui_sections.py` for scheduler UI definitions in `create_sampler_and_steps_selection` and `create_sampler_options`
- `scripts/xyz/xyz_grid_classes.py` for mirrored scheduler UI values used by the XYZ grid scripts
- `modules/sd_samplers_common.py` for `SamplerData` definition and sampler expectations
- `modules/sd_samplers.py` for sampler selection flow and runtime wiring
- `modules/schedulers/**/*.py` for custom scheduler implementations
- `modules/res4lyf/**/*.py` for Res4Lyf scheduler classes (if installed/enabled)

## Guidance

- Consult `.github/instructions/core.instructions.md` for relevant core runtime guidance before proceeding.

## What "Loaded And Compiled" Means

Treat this as a two-level check:

1. Import and class resolution:
   every scheduler class referenced by `SamplerData(... DiffusionSampler(..., SchedulerClass, ...))` resolves without missing symbol errors.
2. Construction and compile sanity:
   scheduler instances can be created from their intended config path and survive a lightweight execution-level check (including compile path when available).

Notes:

- For non-`torch.nn.Module` schedulers, "compiled" means the scheduler integration path is executable in runtime checks (not necessarily `torch.compile`).
- If the environment cannot run compile checks, explicitly state this in the findings summary and proceed with static validation only.

## Procedure

### 1. Build Scheduler Inventory

- Enumerate scheduler classes imported in `modules/sd_samplers_diffusers.py`.
- Enumerate all entries in `samplers_data_diffusers`.
- Enumerate all config keys in `config`.

Create a joined table by sampler name with:

- sampler name
- scheduler class
- config key used
- custom scheduler category (diffusers, SD.Next custom, Res4Lyf)

### 2. Validate Scheduler UI Definitions and Class Resolution

Before validating runtime scheduler classes, inspect `modules/ui_sections.py` and confirm that the UI definitions for scheduler options are consistent with the later scheduler code.

- verify `create_sampler_and_steps_selection` presents sampler names that exist in the sampler registry and are valid for the downstream selection flow
- verify `create_sampler_options` option lists and values match the scheduler runtime option names and accepted values used later in code
- verify UI displayed values such as `default`, `karras`, `betas`, `exponential`, `flowmatch`, `linspace`, `leading`, `trailing`, and checkbox option labels are consumed correctly by scheduler configuration handling
- detect mismatches where a UI option can be selected but would later be rejected, ignored, or misrouted by scheduler code

For each mapped scheduler class:

- confirm symbol exists and is importable
- confirm class object is callable for construction paths used in SD.Next

Flag missing imports, dead entries, or stale class names.

### 3. Validate Config Against Scheduler __init__ Capabilities

For each sampler config entry:

- inspect scheduler class `__init__` signature and accepted config fields
- flag config keys that are unsupported or misspelled
- flag required scheduler init/config fields that are absent
- verify defaults and overrides are compatible with scheduler family behavior

Special attention:

- flow-matching schedulers: shift/base_shift/max_shift/use_dynamic_shifting
- DPM families: algorithm_type/solver_order/solver_type/final_sigmas_type
- compatibility-only keys that are intentionally ignored should be documented, not silently assumed
- detect false positives from runtime config pruning such as `if 'EDM' in name` or `name in {'IPNDM', 'CMSI', 'VDM Solver'}` in `DiffusionSampler`: verify whether unsupported keys are removed intentionally before constructor invocation

### 4. Validate SamplerData Mapping Correctness

For each `SamplerData` entry:

- scheduler label matches config key intent
- callable builds `DiffusionSampler` with the expected scheduler class
- mapping is not accidentally pointing to a different named preset
- no duplicate names with conflicting class/config behavior
- if UI sampler names are displayed in multiple contexts, verify the same name resolves to the same scheduler class and behavior across tabs

Flag mismatches such as wrong display name, wrong class wired to name, or stale aliasing.

### 5. Runtime Smoke Checks (Preferred)

If feasible, run lightweight checks:

- instantiate each scheduler from config
- execute minimal scheduler setup path (`set_timesteps` and a dummy step where possible)
- verify no immediate runtime contract errors

If runtime checks are not feasible for some schedulers, mark those explicitly as unverified-at-runtime.

### 6. Compile Path Validation

Where scheduler runtime path supports compile-related checks in SD.Next:

- verify scheduler integration path remains compatible with compile options
- detect obvious compile blockers introduced by signature/config mismatches

Do not mark compile as passed if only static checks were done.

## Reporting Format

Return findings ordered by severity:

1. Blocking scheduler load failures
2. Config/signature contract mismatches
3. `SamplerData` mapping inconsistencies
4. Non-blocking improvements

For each finding include:

- sampler name
- scheduler class
- file location
- mismatch reason
- minimal corrective action

Also include summary counts:

- total scheduler classes discovered
- total `SamplerData` entries checked
- total config entries checked
- runtime-validated count
- compile-path validated count

## Pass Criteria

The check passes only if all are true:

- all referenced scheduler classes resolve
- each scheduler config entry is compatible with scheduler capabilities
- each `SamplerData` entry is correctly mapped and usable
- no blocking runtime or compile-path failures in validated scope

If scope is partial due to environment limitations, report pass with explicit limitations, not a full pass.
