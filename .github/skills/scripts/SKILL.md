---
name: scripts
description: "Use when creating or changing SD.Next script files under scripts/*, including generation scripts, postprocessing scripts, callbacks, and script utilities. Guides implementation against the script runner contracts; use check-scripts for audits."
argument-hint: "Optionally provide the scripts/* path and whether it is a generation script, postprocessing script, or standalone utility"
---

# Develop SD.Next Scripts

Use this skill for implementation work in `scripts/*`. For a static contract audit, use the separate `check-scripts` skill.

## Workflow

1. Classify the target before editing:
   - Generation or always-on hooks subclass `modules.scripts_manager.Script`.
   - Postprocessing scripts subclass `modules.scripts_postprocessing.ScriptPostprocessing`.
   - Standalone utilities do not need a script class; follow nearby script and CLI conventions instead.
2. Read a neighboring implementation and the owning runner in `modules/scripts_manager.py` or `modules/scripts_postprocessing.py`. Confirm when the file is loaded and which workflows discover it.
3. For generation scripts, preserve the runner contract across `title()`, `show()`, `ui()`, and `run()` or `process()`. Keep UI output order and `args_from`/`args_to` consistent with the receiving signature. For always-on hooks, use the narrowest lifecycle callback that meets the need and preserve the `StableDiffusionProcessing` object contract.
4. For postprocessing scripts, keep `ui()` controls aligned with the `process(pp, ...)` arguments. Preserve the postprocessing result and image/video handling expected by the runner.
5. Keep imports safe during dynamic discovery: avoid generation, model loading, downloads, or other runtime-only side effects at module import or script construction. Defer optional heavy dependencies until the feature is used.
6. Reuse shared processing, queue, state, and callback helpers. Keep script behavior isolated to its selected workflow and avoid leaking mutable state across requests.
7. Validate a changed script with `check-scripts`. If it changes a generation workflow or processing parameter path, also use `check-processing` for that affected path.

## Key References

- `scripts/*`: implementations and standalone utilities
- `modules/scripts_manager.py`: discovery, generation-script lifecycle, UI argument slicing, and hooks
- `modules/scripts_postprocessing.py`: postprocessing discovery and execution contracts
- `.github/instructions/core.instructions.md`: runtime, shared-state, and failure-handling guidance
- `.github/instructions/processing.instructions.md`: UI-to-backend processing and parameter-flow guidance
