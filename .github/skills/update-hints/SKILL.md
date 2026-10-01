---
name: update-hints
description: "Scan modules/ui*.py, scripts/*.py, and ui/locale/locale_en.json to create missing hints, identify duplicates, flag incorrect hints without auto-updating them, and generate a categorized audit report."
argument-hint: "Specify scope (files/sections) and mode: 'audit-only' (report without edits) or 'apply' (create missing hints and report)"
---

# Update UI Hints

Scan UI definitions and components in `modules/ui*.py`, `scripts/*.py`, and built-in extensions, and audit `ui/locale/locale_en.json` to identify and create missing UI hints, detect duplicate entries, check for long labels or broken cross-references, and identify incorrect or outdated hints. Hints must be concise, accurately describe the argument/control, use standard HTML typography, and include recommended values only when necessary for specific use cases.

## When To Use

- Adding new UI components, settings, or controls in `modules/ui*.py`, `scripts/*.py`, or extensions that lack hints.
- Auditing `ui/locale/locale_en.json` for empty hints (`"hint": ""`), missing labels, or duplicate entries.
- Verifying existing entries in `locale_en.json` against actual code labels to catch renamed, changed, or obsolete entries.
- Reviewing UI hints for factual accuracy against current backend/pipeline implementations.
- Identifying and reporting incorrect hints without making automated destructive changes.
- Checking for labels exceeding the 63-character limit or broken `<b><i>...</i></b>` cross-references.
- Checking for very long hints (e.g. `len(hint) > 500` characters) that may benefit from condensing.
- Preparing an audit report of hint status across the codebase.

## Guidance

- Always consult `.github/instructions/hints.instructions.md` and `wiki/Dev-Hints.md` for formatting and typography rules.
- Follow core runtime guidelines in `.github/instructions/core.instructions.md` when reviewing Python UI code.
- When writing helper scripts to extract or audit UI definitions, always place temporary scripts in `tmp/` (e.g. `tmp/scan_hints.py`).

## Primary Files

- `ui/locale/locale_en.json`: Primary source of truth for English hints and localizations.
- `modules/ui*.py`: Core UI definition and component modules (e.g. `modules/ui_definitions.py`, `modules/ui_txt2img.py`, `modules/ui_img2img.py`, `modules/ui_control_elements.py`, `modules/ui_caption.py`, `modules/ui_video.py`, `modules/ui_guidance.py`, `modules/ui_models_load.py`, etc.).
- `scripts/*.py`: Script UI definitions (e.g. `scripts/xyz_grid.py`, `scripts/postprocessing_*.py`, custom script parameters).
- `extensions-builtin/**/*.py`: Built-in extension UI elements.
- `test/validate-locale.py`: Validation script for checking duplicate labels, missing hints, long labels, and formatting.

## Priority Ordering

When identifying and drafting missing hints, prioritize the following sources:
- **High-Priority Items**: UI controls from `modules/ui_control.py` (and `modules/ui_control_*.py`), `modules/ui_video.py` (and sub-modules in `modules/video_models/` and `modules/minimax/`), `modules/ui_postprocessing.py`, and `modules/ui_caption.py`, including all nested UI controls instantiated via helper functions (e.g. `ui_sections.py`, `ui_guidance.py`, `ui_common.py`, `masking.py`, `create_ui_outputs()`, etc.).
- **Medium-Priority Items**: All settings defined in `modules/ui_definitions.py` (`options_templates` OptionInfo definitions).
- **Lower-Priority Items**: Script UI definitions in `scripts/*.py`, built-in extension elements in `extensions-builtin/`, model management tabs, and sub-level interface controls.

## Typography and Formatting Rules

Hint strings render as HTML. Use only the following tags:

| Tag | Purpose | Examples |
| --- | --- | --- |
| `<b>` | Specific values, defaults, dropdown choices, numerics | `<b>0.30</b>`, `<b>Karras</b>`, `<b>v_prediction</b>`, `<b>Euler</b>` |
| `<b><i>...</i></b>` | Cross-references to other UI controls by exact visible label | `<b><i>Denoising strength</i></b>`, `<b><i>Use init image</i></b>`, `<b><i>Images</i></b>` tab |
| `<i>` | Proper nouns: model families, datasets, technique names | `<i>SDXL</i>`, `<i>Flux</i>`, `<i>ControlNet</i>`, `<i>YOLO</i>` |
| `<code>` | Literals: paths, filename tokens, verbatim commands/tokens | `<code>models/yolo</code>`, `<code>-seg</code>`, `<code>[PROMPT]</code>` |

### Content Guidelines

- **Concise & Direct**: Keep hints short and focused (typically under 500 characters). Describe the exact argument or function of the control without filler words (avoid phrases like "This button allows you to...").
- **Recommended Values**: If there is a specific recommended value or range for a given use-case (e.g. recommended CFG scale for specific architectures or recommended step counts), add it briefly, but only if necessary.
- **Unified Tab Reference**: Always refer to the primary generation tab as `<b><i>Images</i></b>` (the ModernUI label). Never write "Control tab".
- **Structure**:
  - Use `<br>` for a single line break within a paragraph.
  - Use `<br><br>` for paragraph breaks.
  - Use `<br>- <b>key</b>: description` for short keyed bullet lists (e.g. dropdown options or mode descriptions).
  - Do not use `<ul>`, `<li>`, Markdown asterisks, or unicode bullet characters.
- **ASCII & PUA Glyphs**: Keep all prose text and descriptions ASCII (avoid unicode curly quotes, em-dashes, or special symbols like →). Non-ASCII Unicode Private Use Area (PUA) icon glyphs (e.g. Nerd Font icons such as `\uf06e`, `\uf0eb`, ``, ``, `⟲`, `🎲️`, `📐`, `※`, `` used for UI controls, badges, and labels) are valid and must not be flagged as incorrect.

## Core Rules

1. **Auto-Add Missing Hints**: Missing UI labels or empty hints (`"hint": ""`) should be drafted and added directly to `ui/locale/locale_en.json` following the typography guidelines (unless running in `audit-only` mode).
2. **Code vs. Locale Alignment & Stale Entry Check**: Verify all existing labels and IDs in `ui/locale/locale_en.json` against actual UI definitions in code (`modules/ui*.py`, `scripts/*.py`, `extensions-builtin/`). As labels evolve or get renamed over time, entries in `locale_en.json` may no longer match. Flag any unmatched or obsolete entries in the report for review (do not auto-delete).
3. **Duplicate Detection**: Identify duplicate labels within and across sections in `ui/locale/locale_en.json` and flag them in the audit report.
4. **Incorrect Hints Handling (No Auto-Update)**: If an existing hint is found to be incorrect (e.g. misleading description, wrong default, invalid choice options, broken cross-references, or mismatched control functionality), **DO NOT update it automatically**. Instead, record the issue in the report with the current text, the reason it is incorrect, and a recommended fix for manual review.
5. **Label Length Constraint**: Flag any UI label exceeding 63 characters (`len(label) > 63`) to prevent layout and localization breaks.
6. **Cross-Reference Integrity**: Verify that any `<b><i>Label</i></b>` cross-reference refers to an actual, verbatim visible label present in `locale_en.json`. Flag broken or obsolete cross-references.
7. **Long Hint Identification**: Flag hints exceeding 500 characters (`len(hint) > 500`) in the audit report to identify overly verbose or complex tooltips that may benefit from streamlining.
8. **Symbol and Tool Button Placement**: Buttons with icon/symbol labels (e.g. `⟲`, `🎲️`, `📐`, `※`, ``) or empty labels identified by `elem_id` belong in the `"_"` section of `locale_en.json`.
9. **Single-Line JSON Record Formatting**: Each entry record in `ui/locale/locale_en.json` MUST be formatted on a single line (e.g. `  {"id": "...", "label": "...", "localized": "", "hint": "..."}`). Do not break individual JSON entry objects across multiple lines.
10. **Final Report**: Always write the complete structured audit report to `tmp/HINTS.md` at the end of the run and output it in the response, containing:
   - Added hints
   - Updated hints
   - Duplicate hints
   - Incorrect hints (with proposed fixes)
   - Stale / Unmatched locale entries (in `locale_en.json` but not found in code)
   - Long hints (>500 characters)
   - Flagged items (long labels, broken cross-references)

## Procedure

### 1. Scan UI Components in `modules/ui*.py` and `scripts/*.py`

Extract all UI controls and settings definitions:
- **`modules/ui_definitions.py`**: Inspect `OptionInfo` entries, note setting keys, labels, component types (`gr.Slider`, `gr.Checkbox`, `gr.Dropdown`, `gr.Radio`, etc.), default values, and choice lists.
- **`modules/ui_*.py` & `scripts/*.py`**: Scan for Gradio components (`gr.Slider`, `gr.Checkbox`, `gr.Dropdown`, `gr.Radio`, `gr.Textbox`, `gr.Button`, `gr.Accordion`, `gr.Tab`, `ToolButton`, etc.) with `label="..."`, `value="..."`, or `elem_id="..."`.
- **Nested UI & Helper Scanning**: When scanning high-priority UI modules, extraction must account for nested builder calls and helper function invocations (such as helper functions in `modules/ui_sections.py`, `modules/ui_guidance.py`, `modules/ui_common.py`, `modules/masking.py`, and sub-module builders like `create_ui_outputs()`, `minimax_ui.create_ui()`, etc.) that instantiate Gradio components on behalf of the parent tab or interface.
- Understand the context and purpose of each control by checking how the parameter is used in execution or pipeline processing.
- For helper extraction scripts, write them to `tmp/` (e.g. `tmp/scan_ui_hints.py`).

### 2. Audit `ui/locale/locale_en.json`

Read `ui/locale/locale_en.json` and perform the audit checks:
1. **Missing Hints**:
   - Existing entries in `locale_en.json` where `"hint": ""` is empty.
   - Labels defined in `modules/ui*.py` or `scripts/*.py` that do not exist in `locale_en.json`.
2. **Code vs. Locale Alignment (Stale / Renamed Entries)**:
   - Verify all existing `locale_en.json` labels and IDs against active codebase definitions.
   - Flag labels in `locale_en.json` that cannot be matched to any code component (indicating the label in code was renamed, modified, or removed).
3. **Duplicate Hints**:
   - Run `python test/validate-locale.py` or inspect dictionary keys for labels appearing multiple times in `locale_en.json`.
   - Note if duplicates have conflicting or identical hints/IDs.
4. **Incorrect Hints**:
   - Compare existing hint text with actual underlying behavior in `modules/ui*.py`, `scripts/*.py`, and backend logic.
   - Check for:
     - Factual errors (e.g. describes a feature that behaves differently).
     - Outdated parameter names, wrong defaults, or non-existent choices.
     - Broken cross-references (`<b><i>Label</i></b>` where `Label` does not exist).
     - Obsolete tab references (e.g. referring to "Control tab" instead of `<b><i>Images</i></b>`).
     - Formatting/HTML errors (e.g. malformed tags, unclosed brackets).
5. **Long Labels**:
   - Check for any label whose length exceeds 63 characters.
6. **Long Hints**:
   - Check for hints whose length exceeds 500 characters (`len(hint) > 500`) to highlight verbose tooltips.
7. **Valid Existing Updates**:
   - Check if existing hints only need supplementary recommended values for specific models/use cases.

### 3. Draft & Apply Safe Updates

- **For Missing Hints** (when not in `audit-only` mode):
  - Draft concise, accurate descriptions.
  - Add optional recommended values if necessary for specific use cases.
  - Place icon/symbol buttons into section `"_"` with their `elem_id`.
  - Place standard text labels into their alphabetical section (`"0"`, `"a"`-`"z"`, or `"reference"`).
  - Schema (MUST be on a single line per record):
    ```json
    {"id": "optional_elem_id", "label": "Label Text", "localized": "", "hint": "Concise hint description..."}
    ```
- **JSON Formatting Rule**:
  - Each object in a section array must be formatted on a single line without newlines breaking the object properties.
  - Ensure standard JSON structure:
    ```json
    {
      "a": [
        {"id": "...", "label": "...", "localized": "", "hint": "..."},
        {"id": "...", "label": "...", "localized": "", "hint": "..."}
      ]
    }
    ```
- **For Incorrect Hints**:
  - **Do NOT modify or overwrite automatically in `locale_en.json`**.
  - Document the finding, rationale, and suggested replacement for the report.

### 4. Validation

- Run locale validation:
  ```bash
  python test/validate-locale.py
  ```
- Check JSON syntax:
  ```bash
  jq empty ui/locale/locale_en.json
  ```
- Lint JSON:
  ```bash
  pnpm eslint -- ui/locale/locale_en.json
  ```

### 5. Prepare Audit Report

At the end of the execution, always write the full structured markdown report to `tmp/HINTS.md` and output it in the response, categorized as follows:

```markdown
## UI Hints Audit Report

### Added Hints
- **`Label Name`** (`section`): `Hint text added`

### Updated Hints
- **`Label Name`** (`section`): `Updated hint text and rationale`

### Duplicate Hints
- **`Label Name`**: Found in sections `[sec1, sec2]` with notes on differences/IDs.

### Incorrect Hints (Flagged for Review - Not Auto-Updated)
- **`Label Name`** (`section`):
  - **Current Hint**: `Existing hint text`
  - **Issue**: Explanation of why the hint is incorrect or misleading
  - **Proposed Fix**: Suggested replacement hint text

### Stale / Unmatched Locale Entries (In locale_en.json but not found in code)
- **`Label Name`** (`section`): `Current hint` — not found in active code (may have been renamed or removed)

### Long Hints (>500 characters)
- **`Label Name`** (`section`, length: XXX): `Hint text summary`

### Flagged Items (Long Labels / Broken Cross-References)
- **Long Label**: `Label Name` (length: XX > 63)
- **Broken Cross-Reference**: `Label Name` references `<b><i>Target</i></b>` which does not exist in `locale_en.json`
```
