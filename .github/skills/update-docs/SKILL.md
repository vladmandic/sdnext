---
name: update-docs
description: "Update wiki markdown docs for syntax correctness, readability, link integrity, heading hierarchy normalization, and code block language tagging. Use when a user asks to clean up markdown formatting and improve clarity while preserving technical meaning."
argument-hint: "Provide one or more wiki/*.md paths and optional scope (syntax-only, readability, or full pass)"
---

# Update Wiki Docs

Read markdown files in the `wiki/` folder, fix markdown syntax issues, and improve readability without changing technical intent.

## When To Use

- The user asks to clean, polish, or normalize markdown docs in `wiki/`
- Headings/lists/code fences/tables render incorrectly
- Documentation is hard to scan due to long paragraphs or inconsistent structure
- A doc needs editorial cleanup before sharing or release

## Scope

Primary scope:

- Files under `wiki/**/*.md`

Baseline actions:

- Correct markdown syntax
- Correct general readability issues
- Run link integrity pass for obvious broken links and anchors
- Normalize heading hierarchy and section flow
- Add or correct code block language tags when known

Expanded actions (when user allows full pass):

- Improve list/table/code-block consistency
- Standardize terminology and naming across the document
- Flag stale or unverifiable claims for follow-up

## Style Rules

- Write in concise technical style that remains approachable to non-expert users
- Prefer short to medium sentences; avoid unnecessary long sentences
- Explain terms briefly when first used if they may be unfamiliar
- Avoid unexplained jargon and acronym-heavy phrasing
- Keep wording direct, specific, and neutral

## Safety Rules

- Preserve technical meaning and factual content
- Do not invent commands, API behavior, or version claims
- Keep edits narrow and reversible
- Preserve existing project-specific terminology unless clearly inconsistent
- If uncertainty is high, prefer adding a short clarification request over guessing

## Procedure

### 1. Confirm Target And Depth

Extract from user prompt:

- target markdown file(s) in `wiki/`
- desired depth: syntax-only, readability, or full pass
- constraints (tone, audience, preserve wording, max rewrite level)

If targets are missing, ask for paths before editing.

### 2. Read And Diagnose

For each target file:

- Scan for syntax/rendering issues
- Identify readability pain points (dense blocks, weak headings, mixed terminology)
- Note risky sections where edits may alter meaning

### 3. Normalize Heading Hierarchy

Apply heading structure rules before deep rewrites:

- keep one logical top-level heading per file where appropriate
- avoid skipped levels (`##` directly to `####`) unless source constraints require it
- ensure sibling sections use consistent levels
- rename headings only when it improves clarity without changing meaning

### 4. Apply Syntax Fixes First

Fix rendering/correctness issues first, such as:

- broken heading levels
- malformed lists
- unclosed/misfenced code blocks
- malformed links/images
- inconsistent table delimiter rows
- accidental HTML/markdown mixing that breaks rendering

### 5. Apply Readability Improvements

Make editorial improvements while preserving meaning:

- split long paragraphs
- convert prose enumerations into lists when clearer
- improve section titles for scanability
- remove repetition and tighten wording
- align terminology within the same document

Apply tone constraints during edits:

- concise technical phrasing
- approachable wording for normal users
- no unexplained technical babble

### 6. Run Link Integrity Pass

Check and fix obvious link issues:

- malformed inline/reference links
- anchors that no longer match heading text after edits
- obvious relative-path mistakes in wiki cross-links

If link targets cannot be verified from repo context, keep the original target and flag it in the report.

### 7. Add Code Block Language Tags

For fenced code blocks:

- add language tags when confidently inferable (`bash`, `python`, `json`, `yaml`, etc.)
- correct clearly wrong tags
- leave tag blank only when language cannot be inferred safely

### 8. Run Completion Checks

Validate each edited file against this checklist:

- markdown renders correctly
- heading hierarchy is logical
- code fences include language where known
- links/anchors are internally consistent and obviously valid
- no factual changes introduced
- tone is concise, technical, and approachable

### 9. Report Results

Return:

- files edited
- categories of changes made (syntax/readability/structure)
- any unresolved ambiguity or potential factual follow-ups

## Branching Guidance

- If the user asks minimal edits, still run heading normalization, link integrity checks, and code-block language tagging with minimal wording changes.
- If the user asks broad cleanup, run full pass including structure and terminology normalization.
- If a section appears technically outdated but cannot be verified from repo context, do not rewrite claims; flag it in the report.

## Pass Criteria

A successful pass means:

- requested wiki markdown files were edited
- syntax/rendering issues were corrected
- heading hierarchy was normalized
- link integrity pass was completed
- code block language tagging was applied where known
- readability clearly improved without changing intent
- output report summarizes edits and open follow-ups
