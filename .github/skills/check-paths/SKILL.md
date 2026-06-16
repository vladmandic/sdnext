---
name: check-paths
description: "Audit SD.Next model-loading code for cache_dir routing on from_pretrained/from_single_file calls and verify diffusers vs HF cache path selection."
argument-hint: "Optionally focus on a subset of loaders, model families, or call patterns"
---

# Check Model Loading Cache Paths

Audit model-loading code and verify every `from_pretrained(...)` and `from_single_file(...)` call explicitly sets `cache_dir` to the correct cache root.

## When To Use

- A loader was added or changed and its cache path needs validation
- A model family loads from the wrong local cache or redownloads unexpectedly
- Auxiliary components are loaded before the full pipeline and need to land in the HF cache
- You want a pre-PR sanity pass for model download/cache routing

## Guidance

- Consult `.github/instructions/core.instructions.md` before proceeding.
- Use existing cache-path conventions rather than introducing new ones.

## Path Policy

1. Full image model pipeline loads:
- Use `shared.opts.diffusers_dir`
- Applies to top-level pipeline `from_pretrained(...)` / `from_single_file(...)` calls that load the full pipeline or full checkpoint into diffusers layout.

2. Auxiliary model loads:
- Use `shared.opts.hfcache_dir`
- Applies to supporting downloads such as VAE, scheduler, processor, tokenizer, image encoder, text encoder, transformer, and other components loaded independently.

3. Component-first loading before pipeline assembly:
- Use `shared.opts.hfcache_dir`
- Applies when a transformer, text encoder, processor, or similar component is loaded before the complete pipeline is instantiated.

## Primary Files

- `modules/sd_models.py`
- `modules/model*.py`
- `modules/ui_models_load.py`
- `modules/models_hf.py`
- `modules/merging/**/*.py`
- `pipelines/**/*.py`

## Audit Goals

For each `from_pretrained(...)` or `from_single_file(...)` call, verify:

1. `cache_dir` is present explicitly.
2. The selected cache root matches the path policy above.
3. Helper wrappers preserve the correct cache choice when they forward args.
4. No load path mixes full pipeline cache and auxiliary cache in a way that would scatter artifacts.

Flag issues such as:

- missing `cache_dir`
- `cache_dir` set to the wrong option
- component loaders using `shared.opts.diffusers_dir` when they should use `shared.opts.hfcache_dir`
- full pipeline loads using `shared.opts.hfcache_dir`
- wrapper functions dropping or overriding the intended cache path

## Procedure

1. Enumerate every `from_pretrained` and `from_single_file` call in the target scope.
2. Classify each call as full pipeline, auxiliary component, or ambiguous wrapper.
3. Confirm the explicit `cache_dir` argument and compare it to the path policy.
4. For wrappers, trace the forwarded path to the final loader call.
5. Report concrete mismatches with minimal fixes.

## Reporting Format

Return findings ordered by severity:

1. Missing `cache_dir`
2. Wrong cache root for the load type
3. Wrapper or forwarding bugs
4. Consistency warnings

For each finding include:

- file location
- loader call
- expected cache root
- actual cache root or missing argument
- minimal fix

Also include summary counts:

- total `from_pretrained` calls checked
- total `from_single_file` calls checked
- full pipeline loads checked
- auxiliary/component loads checked
- ambiguous wrapper loads checked

## Pass Criteria

A full pass requires all audited loader calls to specify `cache_dir` and use the correct cache root for the load category.
