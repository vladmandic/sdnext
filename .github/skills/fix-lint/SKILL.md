---
name: fix-lint
description: "Run SD.Next lint workflow tools in order and fix issues as needed, while ignoring lint findings explicitly marked with TODO."
argument-hint: "Optionally focus on a subset of tools or files, otherwise run full workflow"
---

# Fix Lint Workflow

Run the project lint workflow in the required order, fix findings, and re-run affected tools until clean.

## When To Use

- The user asks to run lint and fix issues
- A PR requires lint-clean status before review
- Multiple files changed and style/static checks may have drifted
- You need a repeatable full-lint remediation pass

## Required Environment Step

Always start from repository root and activate virtual environment first:

- Linux: `. venv/bin/activate`
- Windows: `venv\\scripts\\activate`

If activation fails, report the blocker and stop before running Python-based tools.

## Required Tool Execution Order

Run tools in this exact sequence:

1. `pre-commit run --all-files`
2. `{PROJECT_ROOT}/node_modules/.bin/eslint . javascript/`
3. `cd extensions-builtin/sdnext-modernui && {PROJECT_ROOT}/node_modules/.bin/eslint . javascript/`
4. `ruff check`
5. `pylint *.py`
6. `pylint modules/`
7. `pylint pipelines/`
8. `pylint scripts/`
9. `pylint extensions-builtin/`

Note that `pylint` can run for considerable time, so run with no timeouts.

## Fix Policy

- Fix issues reported by each tool before moving on.
- Ignore lint issues explicitly marked with `TODO`.
- Do not suppress errors globally just to pass checks.
- Keep fixes minimal and targeted to reported findings.
- Preserve existing project conventions and avoid unrelated refactors.

## Procedure

### 1. Initialize Environment

- Confirm current directory is repository root.
- Activate venv using OS-appropriate command.

### 2. Execute And Repair Per Tool

For each tool in the required order:

- Run the command.
- Parse failures and group by file.
- Apply minimal code fixes.
- Re-run the same command until it passes or only `TODO`-marked findings remain.
- If a fix introduces new failures in earlier tools, re-run impacted earlier tools.

### 3. Cross-Check Regression

After all tools pass individually:

- Re-run the full ordered sequence once to ensure no cross-tool regressions.

### 4. Report

Return:

- Commands executed (in order)
- Files changed
- Which issues were fixed
- Any findings intentionally ignored due to `TODO` markers
- Any remaining blockers that could not be auto-fixed

## Pass Criteria

A successful run means:

- All listed tools executed in order
- No remaining fixable lint errors from those tools
- Remaining issues are only those explicitly marked with `TODO` (if any)
- Final verification pass completed
