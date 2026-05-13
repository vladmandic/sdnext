---
name: analyze-model
description: "Analyze an external model URL (typically Hugging Face) to determine implementation style and estimate SD.Next porting difficulty using the port-model workflow."
argument-hint: "Provide model URL and, if applicable, specify target scope: text2img, img2img, edit, video, or full integration"
---

# Analyze External Model For SD.Next Porting

Given an external model URL, inspect how the model is implemented and estimate how hard it is to port into SD.Next according to the port-model skill.

## When To Use

- User provides a Hugging Face model URL and asks if or how it can be ported
- User wants effort estimation before implementation work
- User wants to classify whether integration should reuse Diffusers, use custom Diffusers code, or require full custom implementation

## Accepted Inputs

- Hugging Face model URL (preferred)
- Hugging Face repo id
- Optional source links to custom code repos or PRs

Example:

- https://huggingface.co/jdopensource/JoyAI-Image-Edit

## Required Outputs

For each analyzed model, provide:

1. Implementation classification
2. Evidence for classification
3. SD.Next porting path recommendation
4. Difficulty rating and effort breakdown
5. Main risks and blockers
6. First implementation steps aligned with port-model skill

## Implementation Classification Buckets

Classify into one of these (or closest fit):

1. Integrated into upstream Diffusers library
2. Custom Diffusers implementation in model repo (custom pipeline/classes, not upstream)
3. Fully custom implementation (non-Diffusers inference stack)
4. Existing integration in ComfyUI (node or PR) but not in Diffusers
5. Other (describe clearly)

## Procedure

Process in this order:

1. Handle gated models first: if access returns HTTP 403 or requires gated approval, check `secrets.json` for `huggingface_token` and retry with auth. If access still fails, abort analysis and report model URL, access requirement, and required user action.
2. Inspect repository artifacts: collect model card details, key config files (for example `model_index.json`, `config.json`, scheduler/tokenizer files), Diffusers-style layout signals, and any custom module or remote code requirements.
3. Determine runtime stack: classify usage as standard Diffusers, custom Diffusers with `trust_remote_code`, fully custom inference framework, or node-based host integration (for example ComfyUI).
4. Cross-check SD.Next integration surface: identify needed touchpoints in `pipelines/model_name.py`, `modules/sd_detect.py`, `modules/sd_models.py`, `modules/modeldata.py`, optional custom packages under `pipelines/model/`, and reference/preview catalog updates.
5. Estimate difficulty with this scale: Low (mostly loader wiring), Medium (custom Diffusers or limited adaptation), High (full custom architecture or major behavior differences), Very High (no practical Diffusers path plus runtime mismatch). Break down by loader, API contract, scheduler/sampler, prompt encoding, checkpoint remap, and validation burden.
6. Identify concrete risks: scheduler incompatibility, unclear output domain, custom text encoder constraints, nonstandard checkpoint format, or external runtime dependencies not available in SD.Next.
7. Recommend a port-model path: upstream Diffusers reuse, custom Diffusers pipeline package, or raw checkpoint plus remap; include the smallest viable first implementation milestone.

## Reporting Format

Return sections in this order:

1. Classification
2. Evidence
3. Porting difficulty
4. Recommended SD.Next integration path
5. Risks and unknowns
6. Next actions

If critical information is missing, explicitly list unknowns and what to inspect next.

## Notes

- Prefer concrete evidence from repository files and model card over assumptions.
- If source suggests multiple possible paths, compare at least two and state why one is preferred.
- Keep recommendations aligned with the conventions defined in the port-model skill.
