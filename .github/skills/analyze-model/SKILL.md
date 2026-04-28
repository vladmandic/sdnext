---
name: analyze-model
description: "Analyze an external model URL (typically Hugging Face) to determine implementation style and estimate SD.Next porting difficulty using the port-model workflow."
argument-hint: "Provide model URL and optional target scope: text2img, img2img, edit, video, or full integration"
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

### 0. Handle Gated Models

If the model repository returns HTTP 403 (Forbidden) or requires acceptance of a gating agreement:

1. Check `secrets.json` in the workspace root for a `huggingface_token` field
2. If token exists, retry accessing the model using that token for authentication
3. If token does not exist, is invalid, or access still denied, **abort the analysis** and report:
   - Model name and URL
   - Access requirement (waiting list, gated, license agreement)
   - Instructions for user to authenticate or request access
   - Skip further analysis

### 1. Inspect Model Repository Artifacts

From the provided URL/repo, collect:

- model card details
- files such as model_index.json, config.json, scheduler config, tokenizer files
- presence of Diffusers-style folder layout
- references to custom Python modules or remote code requirements

### 2. Determine Runtime Stack

Identify whether model usage is:

- standard Diffusers pipeline call
- custom Diffusers pipeline class with trust_remote_code
- pure custom inference script or framework
- node-based integration in ComfyUI or another host

### 3. Cross-Check Integration Surface

Determine required SD.Next touchpoints if ported:

- loader file in pipelines/model_name.py
- detect and dispatch updates in modules/sd_detect.py and modules/sd_models.py
- model type mapping in modules/modeldata.py
- optional custom pipeline package under pipelines/model/
- reference catalog updates and preview asset requirements

### 4. Estimate Porting Difficulty

Use this scale:

- Low: mostly loader wiring to existing upstream Diffusers pipeline
- Medium: custom Diffusers classes or limited checkpoint/config adaptation
- High: full custom architecture, major prompt/sampler/output differences, or sparse docs
- Very High: no usable Diffusers path plus major runtime assumptions mismatch

Break down difficulty by:

- loader complexity
- pipeline/API contract complexity
- scheduler/sampler compatibility
- prompt encoding complexity
- checkpoint conversion/remapping complexity
- validation and testing burden

### 5. Identify Risks

Call out concrete risks:

- missing or incompatible scheduler config
- unclear output domain (latent vs pixel)
- custom text encoder or processor constraints
- nonstandard checkpoint format
- dependency on external runtime features unavailable in SD.Next

### 6. Recommend Porting Path

Map recommendation to port-model workflow:

- Upstream Diffusers reuse path
- Custom Diffusers pipeline package path
- Raw checkpoint plus remap path

Provide a concise first-step plan with smallest viable integration milestone.

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
