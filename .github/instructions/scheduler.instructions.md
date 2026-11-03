---
description: "Use when changing scheduler classes, sampler registration or configuration, scheduler compatibility, or scheduler option wiring."
name: "Scheduler Integration Guidelines"
applyTo: "modules/sd_samplers*.py, modules/sd_hijack_schedulers.py, modules/schedulers/**/*.py, modules/ui_sections.py, scripts/xyz/xyz_grid_classes.py, modules/scheduler.py, modules/scheduler_db.py, modules/api/scheduler.py, modules/ui_control.py, modules/control/run.py, wiki/Scheduler.md"
---
# Scheduler Integration Guidelines

1. Treat `modules/sd_samplers_diffusers.py` as the central integration point for scheduler imports, per-scheduler configuration, and `SamplerData` registrations. Keep custom classes in `modules/schedulers/` aligned with that registry.
2. Keep each scheduler's accepted constructor/config fields consistent with its class signature, model-provided defaults, and the overrides applied by `DiffusionSampler`. Preserve model defaults unless an override is supported by that scheduler.
3. Preserve the scheduler interfaces and input types expected by the existing Diffusers and SD.Next sampling paths. Check the caller before changing methods such as `set_timesteps`, `step`, `add_noise`, or `scale_noise`; use existing compatibility helpers where applicable.
4. Keep sampler labels and option values consistent across runtime registration, `modules/ui_sections.py`, and mirrored choices such as `scripts/xyz/xyz_grid_classes.py`.
5. Keep sampler configuration instance-local. Avoid mutating shared model or global scheduler configuration when applying per-request or UI overrides.
6. For sampler audits, use the `check-schedulers` skill. Validate changes against the affected scheduler construction and sampling path, not only importability.

## Scheduled Task Database And API

7. Keep scheduled-task persistence in `modules/scheduler_db.py` and lifecycle/business logic in `modules/scheduler.py`. API and UI code should call those service functions rather than issue SQL or create a parallel task store.
8. Reuse the database connection context manager so transactions commit or roll back and connections close consistently. Preserve foreign-key behavior, parameterize SQL values, and do not hold database transactions open during generation or other slow work.
9. Treat schema, status, mode, and media storage as persistent contracts. Add an explicit migration when changing an existing schema, preserve queued task data, constraints, and indexes, and keep schema-version and diagnostics logic aligned.
10. Respect configured database paths, including `SD_SCHEDULER_DB` and the scheduler directory option; do not hardcode a database location.
11. Use `modules/api/scheduler.py` as the HTTP contract for task management. Keep the scheduler service and database independent of Gradio. Preserve the existing generation-submit callback path through `modules/ui_control.py` and `modules/control/run.py`, which hands scheduled generation to the scheduler service; do not require that path to become API-only.
12. Reflect each new or behavior-changing task-scheduler feature in `wiki/Scheduler.md`. Update only the relevant documentation and avoid unrelated rewrites or cleanup.
