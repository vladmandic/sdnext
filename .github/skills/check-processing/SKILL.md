---
name: check-processing
description: "Validate txt2img/img2img/control/caption processing workflows from UI submit bindings to backend processing execution and confirm parameter/type/init correctness."
argument-hint: "Optionally focus on txt2img, img2img, control, caption, or process-only and include changed files"
---

# Check Processing Workflow Contracts

Trace generation workflows from UI definitions and submit bindings to backend execution, then validate that parameters are passed, typed, and initialized correctly.

## When To Use

- A change touched UI submit wiring for txt2img, img2img, control, or caption workflows
- Processing code changed and regressions are suspected in argument ordering or defaults
- A new parameter was added to UI or processing classes/functions and needs end-to-end validation
- You want a pre-PR contract audit for generation flow integrity

## Required Workflow Coverage

Start from UI definitions and follow each workflow to final implementation:

1. `txt2img`: `modules/ui_txt2img.py` -> `modules/txt2img.py` -> `modules/processing.py:process_images` -> `modules/processing_diffusers.py:process_diffusers`
2. `img2img`: `modules/ui_img2img.py` -> `modules/img2img.py` -> `modules/processing.py:process_images` -> `modules/processing_diffusers.py:process_diffusers`
3. `control/process`: `modules/ui_control.py` -> `modules/control/run.py` (and related control processing entrypoints) -> `modules/processing.py:process_images` -> `modules/processing_diffusers.py:process_diffusers`
4. `caption/process`: `modules/ui_caption.py` -> caption handler module(s) -> `modules/processing.py:process_images` and/or postprocess/caption execution module(s), depending on selected caption backend

Also validate script hooks when present:

- `modules/scripts_manager.py` (`run`, `before_process`, `process`, `process_images`, `after`)

## Primary Files

- `modules/ui_txt2img.py`
- `modules/txt2img.py`
- `modules/ui_img2img.py`
- `modules/img2img.py`
- `modules/ui_control.py`
- `modules/ui_caption.py` (and `modules/ui_captions.py` if present)
- `modules/control/run.py`
- `modules/processing.py`
- `modules/processing_diffusers.py`
- `modules/scripts_manager.py`

## Audit Goals

For each covered workflow, verify all three dimensions:

1. Parameter pass-through correctness (name, order, semantic meaning)
2. Type correctness (UI component output shape vs function signature expectations)
3. Initialization correctness (defaults, `None` handling, fallback logic, and object state)

## Procedure

### 1. Build End-To-End Call Graph

For each workflow (`txt2img`, `img2img`, `control`, `caption`):

- Locate submit/click bindings in UI modules.
- Capture the exact `inputs=[...]` list order and target function (`fn=...`).
- Resolve wrappers (`call_queue.wrap_gradio_gpu_call`, queued wrappers) to actual function signatures.
- Follow function flow through processing class construction and execution (`processing.process_images`, then `process_diffusers` when applicable).

Produce a normalized mapping table per workflow:

- UI input component name
- UI expected output type
- receiving argument in submit target
- receiving processing-object field (if applicable)
- downstream consumption point

### 2. Validate Argument Order And Arity

For each submit path:

- Compare UI `inputs` order against function positional parameter order.
- Validate handling of `*args` and script arguments.
- Confirm `state`, task id, mode flags, and tab selections align with function signatures.
- Flag positional drift where adding/removing an argument in one layer is not propagated.

### 3. Validate Name And Semantic Parity

Check that semantically related parameters remain coherent across layers:

- sampler fields (`sampler_index`, `hr_sampler_index`, sampler name conversion)
- guidance/cfg fields
- size/resize fields
- denoise/refiner/hires fields
- detailer, hdr, grading fields
- override settings fields

Flag mismatches such as:

- same concept with divergent naming and wrong destination
- field sent but never consumed
- required field consumed but never provided

### 4. Validate Type Contracts

Audit type compatibility from UI component to processing target:

- `gr.Image`/`gr.File`/`gr.Video` outputs vs expected Python types (`PIL.Image`, bytes, list, path-like, etc.)
- radios returning index/value and expected downstream representation
- sliders/number inputs (`int` vs `float`) and conversion points
- optional objects (`None`) and `.name`/attribute access safety

Flag ambiguous or unsafe assumptions, especially for optional file inputs and mixed scalar/list values.

### 5. Validate Initialization And Defaults

In target modules (`txt2img.py`, `img2img.py`, `control/run.py`, processing classes):

- verify defaults/fallbacks for invalid or missing inputs
- verify guards for unset model/state and expected error paths
- verify object fields are initialized before first use
- verify flow-specific defaults are not leaking across workflows

Include checks for common regressions:

- `None` passed into required processing fields
- missing fallback for sampler/seed/size values
- stale fields retained from prior job state

### 6. Validate Script Hook Contracts

Where script systems are involved:

- verify `scripts_*.run(...)` fallback behavior to `processing.process_images(...)`
- verify `scripts_*.after(...)` receives compatible processed object
- ensure script args wiring matches `setup_ui(...)` order

### 7. Runtime Spot Check (Preferred)

If feasible, run lightweight smoke validation for each workflow:

- one minimal txt2img run
- one minimal img2img run
- one minimal control run
- one minimal caption run

Use very small dimensions/steps to limit runtime.
If runtime checks are not feasible, explicitly report static-only limitations.

## Reporting Format

Return findings by severity:

1. Blocking contract failures (will break execution)
2. High-risk mismatches (likely wrong behavior)
3. Type/init safety issues
4. Non-blocking consistency issues

For each finding include:

- workflow (`txt2img` | `img2img` | `control` | `caption`)
- layer transition (`ui -> handler`, `handler -> processing`, `processing -> diffusers`)
- file location
- mismatch summary
- minimal fix

Also include summary counts:

- workflows checked
- UI bindings checked
- function signatures checked
- parameter mappings validated
- runtime checks executed vs skipped

## Pass Criteria

A full pass requires all of the following:

- UI submit input order matches target function signatures
- all required parameters are passed end-to-end with correct semantics
- type expectations are explicit and compatible at each boundary
- initialization/default logic prevents unset/invalid state usage
- scripts fallback path to `processing.process_images` is coherent

If only part of the workflow scope was checked, report partial pass with explicit exclusions.
