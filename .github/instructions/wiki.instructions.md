---
description: "Use when cleaning, validating, or improving wiki Markdown documentation, or another Markdown file explicitly named by the user."
name: "Wiki Documentation Guidelines"
applyTo: "wiki/**/*.md"
---
# Wiki Documentation Guidelines

## Scope

- The default scope is Markdown files under `wiki/**/*.md`. This instruction can also be used for other Markdown files when the user explicitly names them.
- Skip Markdown files whose names start with `_` (for example, `_footer.md`) and skip all non-Markdown files.
- The `wiki/` directory is a Git submodule, not part of the SD.Next core repository. Keep wiki edits scoped to that submodule and distinguish wiki changes from core-repository changes. Preserve unrelated existing changes in the submodule.

## Baseline Actions

- Correct Markdown syntax.
- Improve general readability without changing technical intent.
- Check links and anchors for obvious errors.
- Normalize heading hierarchy and section flow.
- Add or correct fenced code block language tags when the language is known.

## Style Rules

- Write in concise technical language that remains approachable to non-expert users.
- Prefer short to medium sentences; avoid unnecessarily long sentences.
- Briefly explain terms on first use when readers may not know them.
- Avoid unexplained jargon and acronym-heavy phrasing.
- Keep wording direct, specific, and neutral.

## Safety Rules

- Preserve technical meaning and factual content.
- Do not invent commands, API behavior, or version claims.
- Keep edits narrow and reversible.
- Preserve project-specific terminology unless it is clearly inconsistent.
- If uncertainty is high, ask for clarification rather than guessing.

## Procedure

1. Confirm the target Markdown files and any scope constraints. Ask for paths if the targets are missing.
2. Confirm the requested depth: syntax-only, readability, or full pass. If unspecified, use readability and state that assumption.
3. For broad or unclear work, run `test/check-docs` first to establish a baseline. Inspect syntax, readability, links, and sections where edits could change technical meaning.
4. Normalize heading levels before making broader wording changes. Keep one logical top-level heading where appropriate, avoid skipped levels, and keep sibling sections consistent.
5. Fix syntax and rendering issues first, including malformed lists, fences, tables, links, and accidental Markdown/HTML mixing.
6. Improve readability by splitting dense paragraphs, tightening repetition, and using lists where they make enumerations easier to scan. Avoid broad restructuring unless requested.
7. Check inline links, reference links, and anchors for malformed syntax and obvious relative-path or heading mismatches. If a target cannot be verified, preserve it and report the uncertainty.
8. Add language tags to fenced code blocks when the language is clear. Correct clearly wrong tags; leave the tag blank when it cannot be inferred safely.
9. Run `test/check-docs` for the affected file or files when possible. Use `test/check-docs --fix <file>` only for safe Markdown lint fixes.
10. Report the edited files, the types of changes made, and any unresolved ambiguity or factual follow-up.

## Completion Checks

- The affected files pass `test/check-docs` when the checker is available.
- Heading hierarchy and code fences are consistent and render correctly.
- Links and anchors are internally consistent where their targets can be verified.
- Changes preserve technical meaning and do not introduce unsupported claims.

For minimal-edit requests, keep wording changes small but still check heading hierarchy, links, and code block language tags. For a full pass, also review section flow and terminology consistency. Flag potentially stale claims when they cannot be verified; do not rewrite them as fact.
