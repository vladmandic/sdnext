---
name: check-models
description: "Audit SD.Next model integrations end-to-end: loaders, detect/routing, reference catalogs, and pipeline API contracts."
argument-hint: "Optionally focus on a model family, repo id, or a subset: loader, detect-routing, references, pipeline-contracts"
---

# Check Model Integrations End-To-End

Run a consolidated model-integration audit that combines loader checks, detect/routing checks, reference-catalog checks, and pipeline contract checks.

## When To Use

- A new model family was added and needs a completeness audit
- Existing model support appears inconsistent across detection, loading, and UI references
- A custom pipeline was ported and needs contract validation
- You want a pre-PR integration quality gate for model-related changes

## Combined Scope

This skill combines four audit surfaces:

1. Loader consistency (`check-loaders` equivalent)
2. Detect/routing parity (`check-detect-routing` equivalent)
3. Reference-catalog integrity (`check-reference-catalog` equivalent)
4. Pipeline API contract conformance (`check-pipeline-contracts` equivalent)

## Primary Files

- `pipelines/model_*.py`
- `modules/sd_detect.py`
- `modules/sd_models.py`
- `modules/modeldata.py`
- `data/reference.json`
- `data/reference-cloud.json`
- `data/reference-quant.json`
- `data/reference-distilled.json`
- `data/reference-nunchaku.json`
- `data/reference-community.json`
- `models/Reference/`

Pipeline files as needed:

- `pipelines/<model>/pipeline.py`
- `pipelines/<model>/model.py`

## Audit A: Loader Consistency

For each target model loader in `pipelines/model_*.py`, verify:

- Correct `sd_models.path_to_repo(checkpoint_info)` and `sd_models.hf_auth_check(...)` usage
- Load args built with `model_quant.get_dit_args(...)` where applicable
- No duplicated kwargs (for example duplicate `torch_dtype`)
- Correct component loading path (`generic.load_transformer`, `generic.load_text_encoder`, tokenizer/processor)
- Proper post-load hooks (`sd_hijack_te`, `sd_hijack_vae`) where required
- Correct `pipe.task_args` defaults where needed
- Cleanup and `devices.torch_gc(...)` present

Flag stale patterns, missing hooks, or conflicting load behavior.

## Audit B: Detect/Routing Parity

Verify model family alignment across:

- `modules/sd_detect.py` detection heuristics
- `modules/sd_models.py` load dispatch branch
- `modules/modeldata.py` reverse classification from loaded pipeline class

Checks:

- Family is detectable by name/repo conventions
- Dispatch routes to the intended loader
- Loaded pipeline class is classified back to the same model family
- Branch ordering does not cause broad matches to shadow specific families

## Audit C: Reference Catalog Integrity

Verify references for model families intended to appear in model references.

Checks:

- Correct category file placement by type:
  - base -> `data/reference.json`
  - cloud -> `data/reference-cloud.json`
  - quant -> `data/reference-quant.json`
  - distilled -> `data/reference-distilled.json`
  - nunchaku -> `data/reference-nunchaku.json`
  - community -> `data/reference-community.json`
- Required fields present per entry (`path`, `preview`, `desc` when expected)
- Duplicate repo/path collisions across reference files are intentional or flagged
- Preview filename convention is consistent
- Referenced preview file exists in `models/Reference/` (or explicitly placeholder if intentional)
- JSON validity for touched reference files

## Audit D: Pipeline API Contracts

For custom pipelines (`pipelines/<model>/pipeline.py`), verify:

- Inherits from `diffusers.DiffusionPipeline`
- Registers modules correctly
- `from_pretrained` wiring is coherent with actual artifact layout
- `encode_prompt` semantics are consistent with tokenizer/text encoder setup
- `__call__` supports expected public args for its task and does not expose unsupported generic args
- Batch and negative prompt behavior are coherent
- Output conversion aligns with model output domain (latent vs pixel space)
- `output_type` and `return_dict` behavior are consistent

## Runtime Validation (Preferred)

When feasible:

- Import-level smoke tests for loaders and pipeline modules
- Lightweight loader construction checks without full heavy generation where possible
- One minimal generation/sampling pass for changed model families

If runtime checks are not feasible, report limitations clearly.

## Reporting Format

Return findings by severity:

1. Blocking integration failures
2. Contract mismatches (load/detect/reference/pipeline)
3. Consistency and quality issues
4. Optional improvements

For each finding include:

- model family
- layer (`loader`, `detect-routing`, `reference`, `pipeline-contract`)
- file location
- mismatch summary
- minimal fix

Also include summary counts:

- loaders checked
- model families checked for detect/routing parity
- reference files checked
- pipeline contracts checked
- runtime checks executed vs skipped

## Pass Criteria

A full pass requires all of the following in audited scope:

- loader path is coherent and non-conflicting
- detect/routing/modeldata parity holds
- reference entries are valid, categorized correctly, and have preview files
- custom pipeline contracts are consistent with actual model behavior

If any area is intentionally out of scope, mark as partial pass with explicit exclusions.
