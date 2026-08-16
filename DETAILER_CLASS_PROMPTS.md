# Detailer Per-Class Prompts (`[CLASS=name]`)

> **Status:** unofficial patch, not an upstream PR. Tested against SD.Next build `2026-08-07` (commit `ea889af1c`). Lives on this fork's branch: `claude/sdnext-detailer-class-prompts-sunhax`.

## TL;DR

The Detailer (ADetailer equivalent) assigns multi-line prompts to detections **by position**, not by what was actually detected. With a multi-class YOLO model, detection order isn't guaranteed to stay stable between runs, so line 1 of your prompt doesn't reliably mean "the same body part" every time.

This patch adds a `[CLASS=name]` prefix you can put on any line of the Detailer prompt/negative fields, so a template is bound to a YOLO class **by name** instead of by line order:

```
[CLASS=face] detailed eyes, sharp iris, clean skin
[CLASS=hand] five fingers, correct anatomy
```

No UI changes, no new options — it's a parsing change to the existing prompt/negative text fields, so old single-line or plain multi-line prompts keep working exactly as before.

## The problem

SD.Next's Detailer runs a YOLO model, gets N detections back, and splits your prompt text on `\n` to build N template strings — `prompt_lines[index]` mapped straight onto `items[index]`. That's fine for single-class models (every detection gets the same treatment anyway), but it breaks down the moment a model detects several different things in one pass: a segmentation model reporting `face`, `hand`, `pussy` in a single call returns them in whatever order the network's output happened to sort them, and that order isn't guaranteed to be stable across seeds, resolutions, or model updates. There is no per-class parameter anywhere in the request schema — `detailer_prompt` is one flat string for the whole model chain.

## The fix

### Syntax

```
[CLASS=name] your prompt text for this class
[CLASS=name1,name2] shared text for either class
plain line with no tag → fallback pool
```

- `class_tag_re = re.compile(r'^\[class\s*=\s*([^\]]+)\]\s*(.*)$', re.IGNORECASE)`
- The `CLASS` keyword is case-insensitive (`[class=...]`, `[Class=...]`, `[CLASS=...]` are equivalent).
- Whitespace around `=` is tolerated — `[CLASS = face]`, `[CLASS= face]`, `[CLASS =face]` all parse the same. There must be **no** space between `[` and `class` itself.
- Class names are matched case-insensitively against the label YOLO reports (`model.names`), and comma-separated lists route multiple classes to the same text.
- Lines with no tag are pooled as **positional fallback** — applied, in order, to any detection whose class had no matching tag, cycling if there are more untagged detections than fallback lines. A prompt with zero `[CLASS=...]` tags behaves exactly like it did before this patch.
- Blank spacer lines are ignored when building the fallback pool (a blank line between a tagged and an untagged line doesn't consume a fallback slot).

### Don't rely on fallback order for multiple classes

The fallback pool is filled **positionally**, matching the untagged lines to untagged detections *in the order each is encountered* — it has no idea what a line's text is about. Writing two distinct untagged lines and expecting each to land on "the right" class is just re-introducing the exact positional-order problem this patch exists to fix, one level down:

```
[CLASS=pussy] pussy prompt text
face prompt text, no tag
nipple prompt text, no tag
```

If detections come back as `[pussy, face, nipple]` this run, the untagged lines happen to land correctly (`face` → face text, `nipple` → nipple text). If a later run returns `[pussy, nipple, face]` instead — a perfectly normal reordering — the same untagged lines land **swapped**: `nipple` gets the face text, `face` gets the nipple text. Silent, no warning, because both class names are still real detections; it's just wired by position.

**This is expected behavior, not a bug.** Any class you actually want to distinguish must get its own explicit `[CLASS=name]` tag. Reserve untagged lines for text you're fine applying to *any* leftover detection regardless of which class it is (e.g. a generic quality boost) — not for a second or third class-specific template.

### What an untagged/unmatched detection gets

If a detection's class has no matching tag **and** there's no fallback line at all, it gets an **empty string**, not the main generation prompt. The "fall back to the main prompt when the Detailer field is empty" rule only fires when the *entire* field is empty before parsing — once you've typed anything (even just one `[CLASS=...]` line), that whole-field check no longer applies. If you want untagged classes to inherit the main prompt, add an explicit fallback line containing the literal token `[PROMPT]` (already substituted before parsing):

```
[CLASS=face] deformed mouth, floating teeth
[PROMPT]
```

### Typo protection

A misspelled tag (`[CLASS=hnad]` instead of `hand`) previously failed silently — the detection just fell through to the fallback pool with no signal anything was wrong. This patch adds an aggregated warning: prompt/negative are parsed once, every model in the detailer chain reports which of the declared class names its detections actually matched, and only tags that matched **nothing across the entire chain** get flagged — once, at the end of the whole pass:

```
WARNING detailer  Detailer prompt: class tags did not match any detection across models=['face-yolo8n', 'ntd11_anime_nsfw_segm_v5']: unmatched=['gace']
```

This had to be aggregate rather than per-model: chaining two detailer models with disjoint classes (say, a face-only model and a separate NSFW segmentation model reporting `nipples`/`pussy`/`anus`/etc.) is a completely normal setup, and a naive per-model check would flag `[CLASS=pussy]` as unmatched on every pass through the face model, and `[CLASS=face]` as unmatched on every pass through the segmentation model — pure noise despite both tags being perfectly correct. The aggregate version only complains when a tag never matches *any* model in the chain, which is the actual signature of a typo.

## Implementation

Three files touched, all in `modules/detailer/`:

**`helper.py`** — two new functions:

```python
def parse_prompt_lines(text: str):
    """Split a detailer prompt into class-tagged templates and positional fallback lines."""
    class_map: dict[str, str] = {}
    fallback: list[str] = []
    for line in (text or '').split('\n'):
        line = line.strip()
        m = class_tag_re.match(line)
        if m:
            names = [n.strip().lower() for n in m.group(1).split(',') if n.strip()]
            for name in names:
                class_map[name] = m.group(2).strip()
        else:
            fallback.append(line)
    return class_map, fallback


def assign_prompts(text: str, items: list) -> list[str]:
    """Resolve a detailer prompt/negative string into one entry per detection."""
    class_map, fallback = parse_prompt_lines(text)
    if len(fallback) == 0:
        fallback = ['']
    resolved = []
    fallback_idx = 0
    for item in items:
        label = (getattr(item, 'label', None) or '').strip().lower()
        if label in class_map:
            resolved.append(class_map[label])
        else:
            resolved.append(fallback[fallback_idx % len(fallback)])
            fallback_idx += 1
    return resolved
```

`item.label` was already populated by the YOLO backend (`modules/detailer/yolo.py`) — it's the same class name printed in the `Load: type=Detailer ... classes=[...]` line at model load time, so no new detection-side plumbing was needed. This alone was the key finding that made the feature straightforward: the data was already there, just discarded before it reached the prompt-assignment step.

**`detailer.py`** — `restore()` now:
1. Resolves `detailer_prompt` / `detailer_negative` once, before the per-model loop (they're identical every iteration; previously recomputed redundantly per model).
2. Parses declared `[CLASS=...]` names once via `parse_prompt_lines`.
3. Inside the loop, after each model's `predict()` call, accumulates which declared class names got matched by that model's detections, and calls `assign_prompts()` to resolve `pc.prompt` / `pc.negative_prompt` per detection instead of the old `prompt_lines[i*len(items)+j]` positional index.
4. After the loop, logs one warning per field for any declared class name that matched zero detections across every model that ran.

**`__init__.py`** — exports `assign_prompts` and `parse_prompt_lines` alongside the existing `detailer_opt`/`DetailerResult`/`list_models`.

## Real-world validation

Tested with SDXL inpainting through a two-model detailer chain: a single-class face model (`face-yolo8n`, class `face`) followed by a multi-class NSFW segmentation model (`ntd11_anime_nsfw_segm_v5`, classes `nipples`/`pussy`/`anus`/`penis`/`cross-section`/`x-ray`/`testicles`). Debug log confirmed each detection received its own class-specific text (`label='face' ... prompt='...'`, `label='pussy' ... prompt='...'`) with zero false-positive typo warnings from the cross-model tag targeting, and one correctly-caught real typo (`[CLASS=gace]` against an actual detected `face`) before the fix, silenced immediately after correcting the tag.

## Known limitations

- This is a parsing convention layered on top of the existing flat `detailer_prompt`/`detailer_negative` strings — there's still no per-model or per-class field in the request schema. Anyone driving the API directly (not through the WebUI textbox) gets the same syntax for free, since it's resolved server-side regardless of how the string arrived.
- No validation against the model's *known* class list at parse time (i.e. no warning the moment you type a bad tag) — the warning only fires after a generation actually runs and the mismatch is confirmed empirically.
- **Incompatible with "Merge detailers".** `Detailer.merge()` (pre-existing, unrelated to this patch) collapses every detection from a model's pass into a single bounding box, and keeps only `items[0].label` — the first detection's class, decided by whatever order the model happened to return them in. If a single multi-class model detects e.g. `face` and `hand` in the same pass with merge enabled, they become one box with one label, and whichever `[CLASS=...]` tag matches that surviving label is the only one applied — the other class's tag is silently dropped, and which one survives can flip between runs. This is conceptually the inverse of what class-tagging is for: don't use "Merge detailers" together with per-class tags on a model that can report more than one class per pass.
- Not upstreamed. If there's community interest, the diff is small (~130 lines across 3 files) and could be proposed against `vladmandic/sdnext` directly.

## Files changed

- `modules/detailer/helper.py`
- `modules/detailer/detailer.py`
- `modules/detailer/__init__.py`
