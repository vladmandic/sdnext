---
name: check-scripts
description: "Audit scripts/*.py and verify Script override contracts (init/title/show) plus ui() output compatibility with run() or process() parameters."
argument-hint: "Optionally focus on a subset of scripts or only run-vs-ui or process-vs-ui checks"
---

# Check Script Class Contracts

Audit all Python scripts in `scripts/*.py` and validate that script class overrides and UI-to-execution parameter contracts are correct.

## When To Use

- New or changed files were added under `scripts/*.py`
- A script crashes when selected or executed from UI
- A script UI was changed and runtime args no longer match
- You want a pre-PR quality gate for script API compatibility

## Scope

Primary audit scope:

- `scripts/*.py`

Contract references:

- `modules/scripts_manager.py` (`Script` base class contracts for `title`, `show`, `ui`, `run`, `process`)
- `modules/scripts_postprocessing.py` (`ScriptPostprocessing` contracts for `ui` and `process`)

## Required Checks

### A. Standard Overrides: `__init__`, `title`, `show`

For each class in `scripts/*.py` that subclasses `scripts.Script` or `scripts_manager.Script`:

1. `title`:
- method exists
- callable signature is valid
- returns non-empty string value

2. `show`:
- method exists
- signature is compatible with script runner usage (`show(is_img2img)` or permissive `*args/**kwargs`)
- return behavior is compatible (`bool` or `scripts.AlwaysVisible` / equivalent)

3. `__init__` (if overridden):
- does not require mandatory constructor args that would break loader instantiation
- avoids side effects that require runtime-only globals at import time
- leaves class in a usable state before `ui()`/`run()`/`process()` are called

Notes:
- `__init__` is optional; do not fail scripts that rely on inherited constructor.
- For dynamic patterns, flag as warning with rationale instead of hard fail.

### B. `ui()` Output vs `run()`/`process()` Parameters

For each script class:

1. Determine execution target:
- Prefer `run()` if present for generation scripts
- Use `process()` if present and `run()` is absent or script is postprocessing-oriented

2. Compare `ui()` output shape to target method parameter expectations:
- `ui()` list/tuple output count should match target positional argument capacity after the first processing arg (`p` or `pp`), unless target uses `*args`
- if target is strict positional (no `*args`/`**kwargs`), detect missing/extra UI values
- if target uses keyword-driven processing, ensure UI dict keys map to accepted params or `**kwargs`

3. Validate ordering assumptions:
- UI control order should align with positional parameter order when positional binding is used
- detect obvious drift when new UI control was added but method signature was not updated

4. Validate optionality/defaults:
- required target parameters should be satisfiable by UI outputs
- defaulted target params are acceptable even if UI omits them

### C. Runner Compatibility

Confirm script methods align with runner expectations in `modules/scripts_manager.py`:

- `ui()` return type is compatible with runner collection (`list/tuple` or recognized mapping pattern where used)
- `run()`/`process()` receive args in expected form from runner slices
- no obvious mismatch between `args_from/args_to` assumptions and script method arity

For postprocessing-style scripts in `scripts/*.py`:

- verify compatibility with `modules/scripts_postprocessing.py` conventions (`ui()` list/dict, `process(pp, *args, **kwargs)`)

## Procedure

1. Enumerate all classes in `scripts/*.py` and classify by base class type.
2. For each generation script class, validate `title`, `show`, optional `__init__`, and `ui` -> `run/process` contracts.
3. For each postprocessing script class under `scripts/*.py`, validate `ui` -> `process` mapping semantics.
4. Cross-check ambiguous cases against script runner behavior from `modules/scripts_manager.py` and `modules/scripts_postprocessing.py`.
5. Report concrete mismatches with minimal fixes.

## Reporting Format

Return findings by severity:

1. Blocking script contract failures
2. Runtime- likely arg/arity mismatches
3. Signature/type compatibility warnings
4. Style/consistency improvements

For each finding include:

- script file
- class name
- failing contract area (`init`, `title`, `show`, `ui->run`, `ui->process`)
- mismatch summary
- minimal fix

Also include summary counts:

- total `scripts/*.py` files checked
- total script classes checked
- classes with `run` contract checked
- classes with `process` contract checked
- override issues found (`init/title/show`)

## Pass Criteria

A full pass requires all of the following across audited `scripts/*.py` classes:

- `title` and `show` overrides are valid and runner-compatible for generation scripts
- overridden `__init__` methods are safely instantiable
- `ui()` output contracts are compatible with `run()` or `process()` args
- no blocking arity/signature mismatch remains

If a class uses highly dynamic argument routing that cannot be proven statically, mark as conditional pass with explicit runtime validation recommendation.
