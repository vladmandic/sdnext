---
description: "Use when editing frontend UI code, JavaScript, HTML, CSS, localization files, or built-in UI extensions including modernui and kanvas."
name: "UI And Frontend Guidelines"
applyTo: "javascript/**/*.js, html/**/*.html, html/**/*.css, html/**/*.js, extensions-builtin/sdnext-modernui/**/*, extensions-builtin/sdnext-kanvas/**/*"
---
# UI And Frontend Guidelines

Apply these rules in priority order:

1. Preserve the current event-handling logic and data flow between Gradio/Python endpoints and frontend handlers; do not change payload shapes without backend alignment.
2. Follow existing project lint and style patterns; prefer consistency with nearby files over introducing new frameworks or architecture.
3. Keep localization-friendly UI text changes synchronized with locale resources in `html/locale_*.json` when user-facing strings are added or changed.
4. Avoid bundling unrelated visual refactors with functional fixes; keep UI PRs scoped and reviewable.
5. For extension UI work, respect each extension's boundaries and avoid cross-extension coupling.
6. Validate JavaScript changes with `pnpm eslint`; for modern UI extension changes also run `pnpm eslint-ui`.
7. Maintain mobile compatibility when touching layout or interaction behavior.
