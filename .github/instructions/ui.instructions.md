---
description: "Use when editing frontend UI code, JavaScript, HTML, CSS, localization files, or built-in UI extensions including modernui and kanvas."
name: "UI And Frontend Guidelines"
applyTo: "javascript/**/*.js, html/**/*.html, html/**/*.css, html/**/*.js, extensions-builtin/sdnext-modernui/**/*, extensions-builtin/sdnext-kanvas/**/*"
---
# UI And Frontend Guidelines

- Preserve existing UI behavior and wiring between Gradio/Python endpoints and frontend handlers; do not change payload shapes without backend alignment.
- Follow existing project lint and style patterns; prefer consistency with nearby files over introducing new frameworks or architecture.
- Keep localization-friendly UI text changes synchronized with locale resources in `html/locale_*.json` when user-facing strings are added or changed.
- Avoid bundling unrelated visual refactors with functional fixes; keep UI PRs scoped and reviewable.
- For extension UI work, respect each extension's boundaries and avoid cross-extension coupling.
- Validate JavaScript changes with `pnpm eslint`; for modern UI extension changes also run `pnpm eslint-ui`.
- Maintain mobile compatibility when touching layout or interaction behavior.
