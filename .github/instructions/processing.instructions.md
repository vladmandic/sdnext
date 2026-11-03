---
description: "Use when changing generation processing parameters, processing objects, workflow dispatch, or UI-to-backend processing paths."
name: "Processing Workflow Guidelines"
applyTo: "modules/processing*.py, modules/txt2img.py, modules/img2img.py, modules/control/run.py, modules/ui_txt2img.py, modules/ui_img2img.py, modules/ui_control.py, modules/ui_caption.py, modules/ui_video.py, modules/video_models/video_run.py, modules/scripts_manager.py"
---
# Processing Workflow Guidelines

1. Treat processing parameters as an end-to-end contract. When adding or changing a field, trace it through the submitting UI or API, request/wrapper arguments, processing object initialization, and the final consumer.
2. Preserve positional argument order and arity across Gradio bindings and their target functions. Update all affected call sites together when a signature changes; avoid silently reinterpreting existing arguments.
3. Keep workflow-specific defaults and optional-value handling explicit. Check `None`, empty media, and fallback behavior at the boundary, and ensure txt2img, img2img, control, caption, and video paths do not inherit stale values from another request.
4. Keep per-generation state on the processing/request object. Use `modules/shared.py` only for intentional shared runtime state, and avoid leaking request values through shared objects or mutable defaults.
5. Preserve the existing processing lifecycle, including queueing, callbacks, script hooks, cleanup, output saving, and metadata propagation. Route work through the established processing entrypoints rather than bypassing them with direct pipeline calls.
6. When a change crosses a workflow boundary or adds a parameter, use the `check-processing` skill to trace the affected UI-to-backend path and validate parameter names, order, types, and initialization.
