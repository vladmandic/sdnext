---
name: check-schedulers
description: "Audit scheduler registrations starting from modules/sd_samplers_diffusers.py and verify class loadability, config validity against scheduler capabilities, and SamplerData correctness."
argument-hint: "Optionally focus on a scheduler subset, such as flow-matching, res4lyf, or parallel schedulers"
---

# Check Scheduler Registry And Contracts

Use `modules/sd_samplers_diffusers.py` as the starting point and verify that scheduler classes, scheduler config, and `SamplerData` mappings are coherent and executable.

## Required Guarantees

The audit must explicitly verify all three:

1. All scheduler classes can be loaded and compiled.
2. All scheduler config entries are valid and match scheduler capabilities in `__init__`.
3. All scheduler classes have valid associated `SamplerData` entries and mapping correctness.

## Scope

Primary file:

- `modules/sd_samplers_diffusers.py`

Related files:

- `modules/sd_samplers_common.py` for `SamplerData` definition and sampler expectations
- `modules/sd_samplers.py` for sampler selection flow and runtime wiring
- `modules/schedulers/**/*.py` for custom scheduler implementations
- `modules/res4lyf/**/*.py` for Res4Lyf scheduler classes (if installed/enabled)

## What "Loaded And Compiled" Means

Treat this as a two-level check:

1. Import and class resolution:
   every scheduler class referenced by `SamplerData(... DiffusionSampler(..., SchedulerClass, ...))` resolves without missing symbol errors.
2. Construction and compile sanity:
   scheduler instances can be created from their intended config path and survive a lightweight execution-level check (including compile path when available).

Notes:

- For non-`torch.nn.Module` schedulers, "compiled" means the scheduler integration path is executable in runtime checks (not necessarily `torch.compile`).
- If the environment cannot run compile checks, report this explicitly and still complete static validation.

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

### 2. Validate Scheduler Class Resolution

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

### 4. Validate SamplerData Mapping Correctness

For each `SamplerData` entry:

- scheduler label matches config key intent
- callable builds `DiffusionSampler` with the expected scheduler class
- mapping is not accidentally pointing to a different named preset
- no duplicate names with conflicting class/config behavior

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