---
description: "Use when editing frontend UI code, TypeScript, JavaScript, HTML, CSS, localization files, or built-in UI extensions including modernui and kanvas."
name: "UI And Frontend Guidelines"
applyTo: "ui/**/*, extensions-builtin/sdnext-modernui/**/*, extensions-builtin/sdnext-kanvas/**/*"
---
# UI And Frontend Guidelines

Apply these rules in priority order:

If rules conflict, prioritize earlier rules over later ones unless explicitly stated otherwise.

1. Preserve the current event-handling logic and data flow between Gradio/Python endpoints and frontend handlers. Do not modify payload shapes or event-handling mechanisms unless explicitly aligned with the backend team.
2. Follow existing project lint and style patterns; prefer consistency with nearby files over introducing new frameworks or architecture.
3. Keep localization-friendly UI text changes synchronized with locale resources in `ui/locale/locale_*.json` when user-facing strings are added or changed. For dynamically generated UI text, ensure that localization keys are pre-defined and referenced appropriately in the codebase.
4. Avoid combining visual changes that do not directly support the functional fixes being implemented; ensure UI PRs are scoped to a single purpose and remain reviewable.
5. For extension UI work, respect each extension's boundaries and avoid cross-extension coupling.
6. Validate TypeScript and JavaScript changes with `pnpm run eslint` and `pnpm run tsc` from the repository root, or the equivalent extension-level command when working inside an extension.
7. Maintain mobile compatibility when touching layout or interaction behavior.

## UI code locations

- Core UI source: `ui/`
- Core build config: `ui/.build.json`
- ModernUI source: `extensions-builtin/sdnext-modernui/src/`
- Kanvas source: `extensions-builtin/sdnext-kanvas/src/`
- ModernUI built output: `extensions-builtin/sdnext-modernui/javascript/`
- Kanvas built output: `extensions-builtin/sdnext-kanvas/javascript/` and `extensions-builtin/sdnext-kanvas/dist/`
- Core built output: `ui/dist/`

> Do not edit built files directly. Always change source files and use the build commands to regenerate UI artifacts.

## Build and development commands

Ensure `pnpm` is installed:

- `npm install -g pnpm`

Install dependencies from the repository root:

- `pnpm install`

Build UI components:

- `pnpm run build:core` - build the core UI
- `pnpm run build:modernui` - build ModernUI
- `pnpm run build:kanvas` - build Kanvas
- `pnpm run build` - build all UI components

Run development builds with watch mode:

- `pnpm run dev:core` - development build for core UI
- `pnpm run dev:modernui` - development build for ModernUI
- `pnpm run dev:kanvas` - development build for Kanvas

## Lint and validation

UI checks are required for all frontend contributions:

- `pnpm run eslint:core` - lint core UI files
- `pnpm run eslint:modernui` - lint ModernUI files
- `pnpm run eslint:kanvas` - lint Kanvas files
- `pnpm run eslint` - lint all UI code
- `pnpm run tsc:core` - type-check core UI files
- `pnpm run tsc:modernui` - type-check ModernUI files
- `pnpm run tsc:kanvas` - type-check Kanvas files
- `pnpm run tsc` - type-check all UI code
- `pnpm run precommit` - run pre-commit checks across the repository
- `pnpm run ui` - run full UI lint/type/build sequence

## Notes

- UI changes require rebuilding before they are visible in the running application.
- If you are changing UI text, update localization resources in `ui/locale/` as needed.
- If a build command fails, check the error logs and ensure all dependencies are installed. Refer to the repository troubleshooting guide for common issues.
- Use the existing code patterns in the current UI folders rather than introducing parallel frontend frameworks.
