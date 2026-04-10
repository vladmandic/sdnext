---
name: github-issues
description: "Read SD.Next GitHub issues with [Issues] in the title and generate a markdown report with short summary, status, and suggested next steps per issue."
argument-hint: "Optionally specify state (open/closed/all), max issues, and whether to include labels/assignees"
---

# Summarize SD.Next [Issues] GitHub Issues

Fetch issues from the SD.Next GitHub repository that contain `[Issues]` in the title, then produce a concise markdown report with one entry per issue.

## When To Use

- The user asks for periodic issue triage summaries
- You need an actionable status report for `[Issues]` tracker items
- You want suggested next actions for each matching issue

## Repository

Default target repository:

- owner: `vladmandic`
- name: `sdnext`

## Required Output

Create markdown containing, for each matching issue:

- issue link and title
- short summary (1-3 sentences)
- status (open/closed and relevant labels)
- suggested next steps (1-3 concrete actions)

## Procedure

### 1. Search Matching Issues

Use GitHub search to find issues with `[Issues]` in title.

Preferred search query template:

- `is:issue in:title "[Issues]" repo:vladmandic/sdnext`

State filters (when requested):

- open only: add `is:open` (default)
- closed only: add `is:closed`

Use `github-pull-request_doSearch` for the search step.

### 2. Fetch Full Issue Details

For each matched issue (within requested limit):

- fetch details with `github-pull-request_issue_fetch`
- capture body, labels, assignees, state, updated time, and key discussion context

### 3. Build Per-Issue Summary

For each issue, produce:

1. Short summary:
- describe problem/request in plain language
- include current progress signal if present

2. Status:
- open/closed
- notable labels (for example: bug, enhancement, stale, blocked)
- optional assignee and last update signal

3. Suggested next steps:
- propose concrete, minimal actions
- tailor actions to issue state and content
- avoid generic filler

### 4. Produce Markdown Report

Return a markdown table or bullet report with one row/section per issue.

Recommended table columns:

- Issue
- Summary
- Status
- Suggested Next Steps

If there are many issues, keep summaries short and prioritize clarity.

## Reporting Rules

- Keep each issue summary concise and actionable.
- Do not invent facts not present in issue data.
- If issue body is sparse, state assumptions explicitly.
- If no matching issues are found, output a clear "no matches" report.

## Pass Criteria

A successful run must:

- search SD.Next issues with `[Issues]` in title
- include all matched issues in scope (or explicitly mention applied limit)
- provide summary, status, and suggested next steps for each issue
- return the final result as markdown
