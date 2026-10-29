---
name: reference-catalog
description: "Maintain and validate SD.Next model reference catalogs in data/reference*.json, including schema consistency, deduplication, link checks, and thumbnail alignment."
argument-hint: "Describe which catalog files to audit (or use all), whether to only report or also fix, and whether to include thumbnail sync in models/Reference"
---

# Reference Catalog Maintenance

Use this skill to audit and update SD.Next model reference catalogs using a phased approach: validate structure first, then resolve duplicates/conflicts, then apply minimal deterministic edits.

## When To Use

- Adding or updating model entries in `data/reference*.json`
- Cleaning duplicates, stale entries, or inconsistent metadata
- Verifying category placement across `base/cloud/quant/distilled/nunchaku/community`
- Syncing catalog entries with thumbnail files in `models/Reference`

## Guidance

- Consult `.github/instructions/core.instructions.md` for relevant core runtime guidance before proceeding.

## Catalog Files In Scope

- `data/reference-base.json` (base)
- `data/reference-cloud.json`
- `data/reference-quantized.json`
- `data/reference-distilled.json`
- `data/reference-nunchaku.json`
- `data/reference-community.json`

## Base Model Wiki Reconciliation

Whenever `data/reference-base.json` is in scope, reconcile every catalog record against both wiki pages. Do not limit this check to newly added or changed records.

1. Use each record's display-name key, `path`, and `variant` (when present) to identify its model family and variant.
2. In `wiki/Model-Support.md`, ensure the family is listed and the relevant version or variant is represented. A family-level entry may cover multiple catalog records only when its listed variants make that coverage explicit.
3. In `wiki/Models.md`, ensure every catalog record maps to a row that names its model and version. A grouped row is acceptable only when it explicitly covers each mapped record.
4. Add missing family entries or rows, and update incomplete or stale wiki entries instead of creating duplicates. Keep a record-to-wiki mapping during the audit so no `reference-base` entry is skipped.
5. Use official links and verified metadata. The catalog's `size` and `date` can inform the matching row; verify architecture, parameter counts, text encoders, autoencoder, and license from reliable sources. Do not invent unavailable details.

When `data/reference-base.json` is not in scope, these wiki checks are not required.

## Core Rules

Priority 1 - data safety and category stability:

- Verify category placement across catalogs and report conflicts first.
- Move entries between categories only when explicitly requested, or when placement is supported by at least two independent metadata sources.
- Keep changes targeted to only affected records.

Priority 2 - schema and formatting consistency:

- Preserve existing field names and conventions used by neighboring entries.
- Prefer deterministic normalization (stable key order, consistent value style).

Priority 3 - assets and size backfill:

- Do not overwrite real thumbnails with placeholders.
- For `size` backfill, use `cli/hf-info.py` -> section `info` -> field `size` as the primary source of truth.

## Validation Checklist

1. Structural validity
- Confirm JSON parses cleanly.
- Ensure top-level structure matches existing catalog conventions.

2. Entry integrity
- Required identifiers exist and are non-empty.
- URLs/repo references are syntactically valid.
- No malformed numeric/string fields compared with peer entries.

3. Cross-catalog consistency
- Detect likely duplicates across `reference*.json` files.
- Flag conflicting metadata for the same model key/name.
- Resolve duplicates by keeping the most complete record in the correct category, then merge missing non-conflicting metadata from duplicate records.
- Report category conflicts; only auto-fix when rules are explicit.

4. Base model wiki alignment (when `reference-base` is in scope)
- Reconcile every record against both `wiki/Model-Support.md` and `wiki/Models.md`.
- Confirm family and variant coverage, update missing or stale entries, and avoid duplicate entries.
- Verify wiki metadata; report details that cannot be confirmed rather than guessing.

5. Thumbnail alignment
- Check expected thumbnail presence under `models/Reference`.
- If missing and requested, create zero-byte placeholder only.
- Never replace existing non-empty image assets with placeholders.

6. Deterministic formatting
- Keep formatting style consistent with nearby file conventions.
- Avoid broad reformatting unrelated to edited records.

7. Fields checks
- Detect missing or extra fields compared to similar entries.
- Validate field value formats (e.g. size in GB, date format).
- Ensure that all fields are consistent and not null, empty or contain zero values.  

8. Size backfill checks (`size: 0`)
- Enumerate all entries with `"size": 0` across `data/reference*.json`.
- For each Hugging Face repo-style path (`owner/name`), run `cli/hf-info.py`.
- Parse `info.data.size` from tool output when present (format is MB string, e.g. `"23933.4MB"`).
- Convert MB to GB using deterministic rounding: `gb = round(mb / 1024, 2)`.
- Update only the `size` field for resolvable records; do not modify unrelated fields.
- If `cli/hf-info.py` returns `ok: false`, missing `data.size`, or non-repo paths, leave `size` unchanged and report as unresolved.
- Do not invent fallback sizes unless explicitly requested.

## Safe Edit Workflow

1. Identify target entries and category intent.
2. Audit only relevant catalog files first.
3. When `reference-base` is in scope, map every record to its family/variant coverage on both wiki pages and note any missing or stale documentation.
4. Run size backfill using `cli/hf-info.py`.
5. Propose minimal edits (or apply when asked), including the required wiki corrections.
6. Re-validate JSON, duplicate checks, and base-model wiki coverage.
7. Summarize exact changed records and rationale.

## Common Failure Modes To Prevent

- Adding a model to wrong category file
- Duplicating near-identical entries under different names
- Breaking JSON structure while editing by hand
- Inconsistent key naming across similar entries
- Creating placeholder thumbnail over an existing asset
- Running `cli/hf-info.py` with the wrong Python environment/interpreter
- Treating `subfolder` variants as unsupported when the repo path itself is valid
- Writing guessed `size` values when `cli/hf-info.py` returns no size

## Output Contract

When using this skill, provide:

- Files audited
- Validation findings grouped by severity
- Exact records changed (before/after summary)
- For `reference-base`: record-to-entry coverage for `wiki/Model-Support.md` and `wiki/Models.md`, including wiki edits or unresolved metadata
- Duplicate/conflict report across catalogs
- Thumbnail sync result for `models/Reference`
- `size: 0` backfill report: total candidates, updated count, unresolved count, unresolved reasons
- Residual risks or follow-up items
