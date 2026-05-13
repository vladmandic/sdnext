---
name: port-pipeline
description: "Port custom model pipeline implementations to Diffusers using phased priorities: preserve behavior first, avoid new dependencies second, and keep device/attention handling configurable throughout. Use when migrating custom or non-Diffusers pipeline code into SD.Next repo-local pipeline files such as pipelines/model_<name>.py or pipelines/<model>/pipeline.py."
argument-hint: "Provide source pipeline path, target SD.Next destination path, and target pipeline class name"
---

# Port Custom Pipeline To Diffusers

Port an existing custom model pipeline implementation into a Diffusers-compatible pipeline class with behavior parity and SD.Next-friendly conventions.
This skill targets SD.Next repo-local pipeline ports only.

## When To Use

- A user has a custom pipeline implementation and wants it ported to Diffusers
- Existing model code is runnable but not structured as a Diffusers pipeline
- The destination is SD.Next pipeline code under `pipelines/model_*.py` or `pipelines/<model>/`
- The task requires preserving generation behavior without introducing new dependencies
- The task requires removing hard-coded runtime assumptions (device or attention backend)

## Mandatory Clarification Gate

Before implementation, confirm these required inputs with the user:

1. Path to the source custom pipeline implementation
2. Destination path in this SD.Next repository (typically under `pipelines/`)
3. Target pipeline class name

If any of the above are missing or ambiguous, stop and ask concise clarification questions before writing code.
If the user input is invalid (for example, nonexistent path, non-Python source file, or invalid class name), report the specific validation error and request corrected input before writing code.

## Constraints

Priority 1 - behavior constraints:

- Preserve externally visible behavior of the source pipeline unless the user asks for intentional changes

Priority 2 - dependency constraints:

- Do not add new dependencies

Priority 3 - runtime configurability constraints:

- Do not hard-code device type (`cpu`, `cuda`, `mps`, etc.)
- Do not hard-code attention type or backend assumptions

## Workflow

1. Collect Inputs
- Ask for source path, destination path, and target pipeline name.
- Confirm destination is an SD.Next repo-local pipeline location, not an upstream Diffusers repository path.
- Confirm runtime assumptions, including device configuration, memory constraints, and expected task type (text-to-image, image-to-image, inpaint, etc.).

2. Analyze Source Pipeline
- Inspect model loading, prompt processing, denoising or sampling loop, scheduler interactions, and output post-processing.
- Identify all components that must be ported: models, tokenizers or processors, schedulers, adapters, preprocessors, postprocessors, callbacks, and output dataclasses.
- Note any hidden global state, side effects, or implicit defaults that must become explicit parameters.

3. Map To Diffusers Interfaces
- Choose the most appropriate Diffusers base class and output type.
- Define `__init__`, module registration, `from_pretrained` and `__call__` signatures aligned with existing Diffusers patterns.
- Keep parameter names and behavior as close as possible to upstream conventions.
- Identify any custom classes needed beyond the pipeline itself: transformer blocks, attention processors, custom schedulers, or output types. Plan a separate module file for each.

4. Implement Supporting Classes
- If the pipeline requires custom model classes (e.g., a custom transformer block, attention module, or other model component), implement each in a **separate module** located in the **same directory** as the main pipeline file (e.g., `pipelines/<model>/transformer.py`, `pipelines/<model>/scheduler.py`).
- If the pipeline requires a custom scheduler class, implement it in its own module (e.g., `pipelines/<model>/scheduler_<name>.py`) following Diffusers scheduler conventions (`step`, `add_noise`, `scale_model_input`, etc.).
- Each supporting class module must be self-contained: no circular imports, no hidden global state, and no hard-coded device or attention assumptions.
- Import supporting classes into the main pipeline module from their respective sibling modules.

5. Implement Pipeline Class
- Create the destination pipeline classes at the user-provided path.
- Port logic in small, testable sections: initialization, input validation, prompt encoding, latent preparation, denoising loop, decoding, and output packaging.
- Replace hard-coded device and attention logic with runtime-configurable behavior.
- Keep imports limited to existing project and Diffusers dependencies.

6. Lint And Fix
- Activate the project venv: `source venv/bin/activate`
- Run `ruff` on all newly written files: `pnpm ruff` (or `ruff check <file> --fix` for targeted runs).
- Run `pylint` on all newly written files: `pnpm pylint` (or `pylint <file>` for targeted runs).
- Fix every reported error or warning that is not explicitly marked with a `TODO` suppression comment in the source.
- Re-run both linters after fixes to confirm a clean result before proceeding.

7. Validate Behavior Parity
- Compare source and ported implementations for input-output shape handling, dtype flow, scheduler step ordering, and guidance behavior.
- Run focused checks or smoke tests if available in the workspace.
- Call out any known differences that were required for Diffusers compatibility.

8. Report Results
- Summarize what was ported and where.
- List any unresolved assumptions, risks, or TODOs.
- Provide minimal follow-up steps for integration and testing.

## Review Checklist

- Required inputs were collected before edits
- No new dependency was introduced
- No hard-coded device or attention backend remains
- Core components from source pipeline were fully mapped
- Pipeline class is in requested destination with requested name
- Each custom supporting class (transformer, scheduler, etc.) is in its own sibling module
- Supporting modules have no circular imports or hidden global state
- `ruff` and `pylint` both pass cleanly on all newly written files (venv activated)
- Main inference path behavior matches the source implementation

## Output Expectations

Final response should include:
- Source path, destination path, and final pipeline class name
- Brief parity summary of key components ported
- Validation performed and any gaps
- Explicit note of any assumptions requiring user confirmation
