---
name: todo
description: "Search the entire codebase for TODO markers and create a markdown document with proposed next steps for each."
argument-hint: "Optionally focus on specific folders or TODO categories, otherwise scan the full repo"
---

# Audit TODO Markers And Propose Next Steps

Search the repository for TODO markers, collect each actionable item, and produce a markdown report with recommended next steps.

## When To Use

- The user wants a backlog extracted from inline TODO comments
- You need a maintenance audit of unfinished work across the repo
- You want to triage technical debt before a release or PR
- You need a markdown planning document based on code comments

## Scope

Search the full repository for TODO markers, including at minimum:

- Python files
- JavaScript files
- TypeScript files
- Markdown and config files where TODOs may appear
- Built-in extensions and pipelines

Prefer whole-repo search unless the user explicitly narrows scope.

## Required Search Targets

Look for common TODO variants such as:

- `TODO`
- `TODO:`
- `# TODO`
- `// TODO`
- `/* TODO */`
- inline TODO notes in comments or docstrings

Ignore generated/vendor output when clearly not user-maintained.

## What To Capture

For each TODO found, collect:

- file path
- line number
- short excerpt of the TODO text
- likely subsystem or area (`api`, `pipeline`, `ui`, `model loading`, `lint`, etc.)
- proposed next step
- rough priority (`high`, `medium`, `low`) based on likely user impact or breakage risk

## Procedure

### 1. Search Entire Codebase

- Run a repo-wide search for TODO markers.
- Deduplicate repeated hits when the same TODO appears multiple times from generated mirrors or repeated excerpts.

### 2. Read Local Context

For each meaningful TODO:

- Read enough surrounding lines to understand intent.
- Determine whether it is:
  - bug fix
  - missing feature
  - refactor
  - cleanup
  - documentation
  - test gap
  - temporary workaround

### 3. Propose Next Steps

For each TODO, write a concrete next step, for example:

- implement missing branch
- remove obsolete compatibility path
- add test coverage
- document required behavior
- replace placeholder logic
- validate runtime behavior

Do not just restate the TODO. Convert it into an actionable recommendation.

### 4. Produce Markdown Document

Create a markdown report that groups TODOs in a readable way.

Preferred structure:

- Title
- Summary counts
- Grouped sections by subsystem or priority
- One entry per TODO with:
  - file and line
  - TODO excerpt
  - proposed next step
  - priority

### 5. Report Limitations

If a TODO is ambiguous, say so and give the most likely next step instead of inventing certainty.

## Output Expectations

When this skill is used, return:

- total TODO count found
- files containing TODOs
- markdown document content or path to generated markdown file
- grouped actionable next steps for each TODO

## Pass Criteria

A successful pass means:

- the repository was searched comprehensively
- TODO markers were collected with file locations
- each TODO has a proposed next step
- the final output is a markdown document suitable for planning or triage
