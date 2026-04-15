---
name: reference-catalog
description: "Maintain and validate SD.Next model reference catalogs in data/reference*.json, including schema consistency, deduplication, link checks, and thumbnail alignment."
argument-hint: "Describe which catalog files to audit (or use all), whether to only report or also fix, and whether to include thumbnail sync in models/Reference"
---

# Reference Catalog Maintenance

Use this skill to audit and update SD.Next model reference catalogs with minimal, safe, and deterministic edits.

## When To Use

- Adding or updating model entries in `data/reference*.json`
- Cleaning duplicates, stale entries, or inconsistent metadata
- Verifying category placement across `base/cloud/quant/distilled/nunchaku/community`
- Syncing catalog entries with thumbnail files in `models/Reference`

## Catalog Files In Scope

- `data/reference.json` (base)
- `data/reference-cloud.json`
- `data/reference-quant.json`
- `data/reference-distilled.json`
- `data/reference-nunchaku.json`
- `data/reference-community.json`

## Core Rules

- Do not move entries between categories unless explicitly requested or strongly evidenced.
- Keep changes targeted to only affected records.
- Preserve existing field names and conventions used by neighboring entries.
- Prefer deterministic normalization (stable key order, consistent value style).
- Do not overwrite real thumbnails with placeholders.
- For `size` backfill, use `cli/hf-info.py` as the primary source of truth.

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
- Report category conflicts; only auto-fix when rules are explicit.

4. Thumbnail alignment
- Check expected thumbnail presence under `models/Reference`.
- If missing and requested, create zero-byte placeholder only.
- Never replace existing non-empty image assets with placeholders.

5. Deterministic formatting
- Keep formatting style consistent with nearby file conventions.
- Avoid broad reformatting unrelated to edited records.

6. Fields checks
- Detect missing or extra fields compared to similar entries.
- Validate field value formats (e.g. size in GB, date format).
- Ensure that all fields are consistent and not null, empty or contain zero values.  

7. Size backfill checks (`size: 0`)
- Enumerate all entries with `"size": 0` across `data/reference*.json`.
- For each Hugging Face repo-style path (`owner/name`), run `cli/hf-info.py`.
- Parse `data.size` from tool output when present (format is MB string, e.g. `"23933.4MB"`).
- Convert MB to GB using deterministic rounding: `gb = round(mb / 1024, 2)`.
- Update only the `size` field for resolvable records; do not modify unrelated fields.
- If `cli/hf-info.py` returns `ok: false`, missing `data.size`, or non-repo paths, leave `size` unchanged and report as unresolved.
- Do not invent fallback sizes unless explicitly requested.

## Safe Edit Workflow

1. Identify target entries and category intent.
2. Audit only relevant catalog files first.
3. Run size backfill using `cli/hf-info.py`.
4. Propose minimal edits (or apply when asked).
5. Re-validate JSON and duplicate checks.
6. Summarize exact changed records and rationale.

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
- Duplicate/conflict report across catalogs
- Thumbnail sync result for `models/Reference`
- `size: 0` backfill report: total candidates, updated count, unresolved count, unresolved reasons
- Residual risks or follow-up items
