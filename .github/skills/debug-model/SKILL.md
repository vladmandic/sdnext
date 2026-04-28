---
name: debug-model
description: "Debug a broken SD.Next or Diffusers model integration. Use when a newly added or ported model fails to load, misdetects, crashes during prompt encoding or sampling, or produces incorrect outputs."
argument-hint: "Describe the failing model, where it fails, the error message, and whether the model is upstream Diffusers, custom pipeline, or raw checkpoint based"
---

# Debug SD.Next And Diffusers Model Port

Read the error, identify which integration layer is failing, isolate the smallest reproducible failure point, fix the root cause, and validate the fix without expanding scope.

## When To Use

- A newly added SD.Next model type does not autodetect correctly
- The loader fails to instantiate a pipeline or component
- A custom pipeline imports but fails during `from_pretrained`
- Prompt encoding fails because of tokenizer, processor, or text encoder mismatch
- Sampling fails due to tensor shape, dtype, device, or scheduler issues
- The model loads but outputs corrupted images, wrong output type, or obviously incorrect results

## Debugging Order

Always debug from the outside in.

1. Detection and routing
2. Loader arguments and component selection
3. Checkpoint path and artifact layout
4. Weight loading and key mapping
5. Prompt encoding
6. Sampling forward path
7. Output postprocessing and SD.Next task integration

Do not start by rewriting the architecture if the failure is likely in detection, loader wiring, or output handling.

## Files To Check First

- `.github/copilot-instructions.md`
- `.github/instructions/core.instructions.md`
- `modules/sd_detect.py`
- `modules/sd_models.py`
- `modules/modeldata.py`
- `pipelines/model_<name>.py`
- `pipelines/<model>/model.py`
- `pipelines/<model>/pipeline.py`
- `pipelines/generic.py`

If the port is based on a standalone script, compare the failing path against the original reference implementation and identify the first semantic divergence.

## Failure Classification

### 1. Model Not Detected Or Misclassified

Check:

- Filename and repo-name heuristics in `modules/sd_detect.py`
- Loader dispatch branch in `modules/sd_models.py`
- Reverse pipeline classification in `modules/modeldata.py`

Typical symptoms:

- Wrong loader called
- Pipeline classified as a broader family such as `chroma` instead of a custom `zetachroma`
- Task switching behaves incorrectly because the loaded pipeline type is wrong

### 2. Loader Fails Before Pipeline Construction

Check:

- `sd_models.path_to_repo(checkpoint_info)` output
- `generic.load_transformer(...)` and `generic.load_text_encoder(...)` arguments
- Duplicate kwargs such as `torch_dtype`
- Wrong class chosen for text encoder, tokenizer, or processor
- Whether the source is really a Diffusers repo or only a raw checkpoint

Typical symptoms:

- Missing subfolder errors
- `from_pretrained` argument mismatch
- Component class mismatch

### 3. Raw Checkpoint Load Fails

Check:

- Checkpoint path resolution for local file, local directory, and Hub repo
- State dict load method
- Key remapping logic
- Config inference from tensor shapes
- Missing versus unexpected keys after `load_state_dict`

Typical symptoms:

- Key mismatch explosion
- Wrong inferred head counts, dimensions, or decoder settings
- Silent shape corruption caused by a bad remap

### 4. Prompt Encoding Fails

Check:

- Tokenizer or processor choice
- `trust_remote_code` requirements
- Chat template or custom prompt formatting
- Hidden state index selection
- Padding and batch alignment between positive and negative prompts

Typical symptoms:

- Tokenizer attribute errors
- Hidden state shape mismatch
- CFG failures when negative prompts do not match prompt batch length

### 5. Sampling Or Forward Pass Fails

Check:

- Input tensor shape and channel count
- Device and dtype alignment across all components
- Scheduler timesteps and expected timestep convention
- Classifier-free guidance concatenation and split logic
- Pixel-space versus latent-space assumptions

Typical symptoms:

- Shape mismatch in attention or decoder blocks
- Device mismatch between text encoder output and model tensors
- Images exploding to NaNs because timestep semantics are inverted

### 6. Output Is Wrong But No Exception Is Raised

Check:

- Whether the model predicts `x0`, noise, or velocity
- Whether the Euler or other sampler update matches the model objective
- Final scaling and clamp path
- `output_type` handling and `pipe.task_args`
- Whether a VAE is being applied incorrectly to direct pixel-space output

Typical symptoms:

- Black, gray, washed-out, or heavily clipped images
- Output with correct size but obviously broken semantics
- Correct tensors but wrong SD.Next display behavior because output type is mismatched

## Minimal Debug Procedure

### 1. Reproduce Narrowly

Capture the smallest failing operation.

- Pure import failure
- Loader-only failure
- `from_pretrained` failure
- Prompt encode failure
- Single forward pass failure
- First sampler step failure

Prefer narrow Python checks before attempting a full generation run.

### 2. Compare Against Working Pattern

Find the closest working in-repo analogue and compare:

- Loader structure
- Registered module names
- Pipeline class name and module registration
- Prompt encoding path
- Output conversion path

### 3. Fix The Root Cause

Examples:

- Add the missing `modeldata` branch instead of patching downstream task handling
- Fix checkpoint remapping rather than forcing `strict=False` and ignoring real mismatches
- Correct the output path for pixel-space models instead of routing through a VAE
- Make config inference fail explicitly when ambiguous instead of guessing silently

### 4. Validate In Layers

After each meaningful fix, validate the narrowest relevant layer first.

- `compileall` or syntax check
- `ruff` on touched files
- Import smoke test
- Loader-only smoke test
- Full run only when the lower layers are stable

## Common Root Causes

- `modules/modeldata.py` not updated after adding a new custom pipeline family
- `modules/sd_detect.py` branch order causes overbroad detection to win first
- Loader passes duplicated keyword args like `torch_dtype`
- Shared text encoder assumptions do not match the actual model variant
- `from_pretrained` assumes `transformer/` or `text_encoder/` subfolders that do not exist
- Key remapping merges QKV in the wrong order
- CFG path concatenates embeddings or latents incorrectly
- Direct pixel-space models are postprocessed like latent-space diffusion outputs
- Negative prompts are not padded or repeated to match prompt batch shape
- Pipeline class naming collides with broader family checks in `modeldata`

## Validation Checklist

When closing the task, report which of these were completed:

1. Exact failing layer identified
2. Root cause fixed
3. Syntax check passed
4. Focused lint passed
5. Import or loader smoke test passed
6. Real generation tested, or explicitly not tested

## Example Request Shapes

- "The new model port fails in from_pretrained"
- "SD.Next detects my custom pipeline as the wrong model type"
- "The loader works but generation returns black images"
- "This standalone-script port loads weights but crashes in attention"
