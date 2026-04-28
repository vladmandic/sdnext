---
name: port-model
description: "Port or add a model to SD.Next using existing Diffusers and custom pipeline patterns. Use when implementing a new model loader, custom pipeline, checkpoint conversion path, or SD.Next model-type integration."
argument-hint: "Describe the source model, target task, checkpoint format, and whether the model already has a Diffusers pipeline"
---

# Port Model To SD.Next And Diffusers

Read the task, identify the model architecture and artifact layout, choose the narrowest integration path that matches existing SD.Next patterns, implement the loader and pipeline wiring, and validate the result.

## When To Use

- The user wants to add a new model family to SD.Next
- A standalone inference script needs to become a Diffusers-style pipeline
- A raw checkpoint or safetensors repo needs an SD.Next loader
- A model already exists in Diffusers but is not yet wired into SD.Next
- A custom architecture needs a repo-local `pipelines/<model>` package and loader

## Core Rule

Prefer the smallest correct integration path.

- If upstream Diffusers already supports the model cleanly, reuse the upstream pipeline and only add SD.Next wiring.
- If the model requires custom architecture or sampler behavior, add a repo-local pipeline package under `pipelines/<model>`.
- If the model ships as a raw checkpoint instead of a Diffusers repo, load or remap weights explicitly instead of pretending it is a standard Diffusers layout.

## Handle Gated Models

**Before collecting inputs:** If the model repository is gated (HTTP 403, access agreement required, or waiting list):

1. Check `secrets.json` in the workspace root for a `huggingface_token` field
2. If token exists, use it to authenticate and attempt access
3. If token missing, invalid, or access still denied, **abort the port work** immediately and report:
   - Model name and repo link
   - Specific access barrier (gated, closed, waiting list, license agreement)
   - Required steps for user to gain access or provide token
   - Do **not** proceed with implementation

## Inputs To Collect First

Before editing anything, determine these facts:

- Model task: text-to-image, img2img, inpaint, editing, video, multimodal, or other
- Artifact format: Diffusers repo, local folder, single-file safetensors, ckpt, GGUF, or custom layout
- Core components: transformer or UNet, VAE or VQ model, text encoder, tokenizer or processor, scheduler, image encoder, adapters
- Whether output is latent-space or direct pixel-space
- Whether the model family has one fixed architecture or multiple variants
- Whether prompt encoding follows a normal tokenizer path or a custom chat/template path
- Whether there is an existing in-repo analogue to copy structurally

If any of these remain unclear after reading the repo and source artifacts, ask concise clarifying questions before implementing.

## Mandatory Category Question

Before implementing model-reference updates, explicitly ask the user which category the model belongs to:

- `base`
- `cloud`
- `quant`
- `distilled`
- `nunchaku`
- `community`

Do not guess this category. Use the user answer to decide which reference JSON file(s) to update.

## Repo Files To Check

Start by reading the task description, then inspect the closest matching implementations.

- `.github/copilot-instructions.md`
- `.github/instructions/core.instructions.md`
- `pipelines/generic.py`
- `pipelines/model_*.py` files similar to the target model
- `modules/sd_models.py`
- `modules/sd_detect.py`
- `modules/modeldata.py`

Useful examples by pattern:

- Existing upstream Diffusers loader: `pipelines/model_chroma.py`, `pipelines/model_z_image.py`
- Custom in-repo pipeline package: `pipelines/f_lite/`
- Shared Qwen loader pattern: `pipelines/model_z_image.py`, `pipelines/model_flux2_klein.py`

## Integration Decision Tree

### 1. Upstream Diffusers Support Exists

Use this path when the model already has a usable Diffusers pipeline and component classes.

Implement:

- `pipelines/model_<name>.py`
- `modules/sd_models.py` dispatch branch
- `modules/sd_detect.py` filename autodetect branch if appropriate
- `modules/modeldata.py` model type detection branch

Reuse:

- `generic.load_transformer(...)`
- `generic.load_text_encoder(...)`
- Existing processor or tokenizer loading patterns
- `sd_hijack_te.init_hijack(pipe)` and `sd_hijack_vae.init_hijack(pipe)` where relevant

### 2. Custom Pipeline Needed

Use this path when the model architecture, sampler, or prompt encoding is not available upstream.

Implement:

- `pipelines/<model>/__init__.py`
- `pipelines/<model>/model.py`
- `pipelines/<model>/pipeline.py`
- `pipelines/model_<name>.py`
- SD.Next registration in `modules/sd_models.py`, `modules/sd_detect.py`, and `modules/modeldata.py`

Model module responsibilities:

- Architecture classes
- Config handling and defaults
- Weight remapping or checkpoint conversion helpers
- Raw checkpoint loading helpers if needed

Pipeline module responsibilities:

- `DiffusionPipeline` subclass
- `__init__`
- `from_pretrained`
- `encode_prompt` or equivalent prompt preparation
- `__call__`
- Output dataclass
- Optional callback handling and output conversion

### 3. Raw Checkpoint Or Single-File Weights

Use this path when the model source is not a normal Diffusers repository.

Requirements:

- Resolve checkpoint path from local file, local directory, or Hub repo
- Load state dict directly
- Remap keys when training format differs from inference format
- Infer config from tensor shapes only if the model family truly varies and no config file exists
- Raise explicit errors for ambiguous or incomplete layouts

Do not fake a `from_pretrained` implementation that silently assumes missing subfolders exist.

## Required SD.Next Touchpoints

Most new model families need all of these:

- `pipelines/model_<name>.py`
  Purpose: SD.Next loader entry point
- `modules/sd_models.py`
  Purpose: route detected model type to the correct loader
- `modules/sd_detect.py`
  Purpose: detect model family from filename or repo name
- `modules/modeldata.py`
  Purpose: classify loaded pipeline instance back into SD.Next model type

Add only what the model actually needs.

Reference catalog touchpoints are also required for model ports intended to appear in SD.Next model references.

- `data/reference.json` for `base`
- `data/reference-cloud.json` for `cloud`
- `data/reference-quant.json` for `quant`
- `data/reference-distilled.json` for `distilled`
- `data/reference-nunchaku.json` for `nunchaku`
- `data/reference-community.json` for `community`

If the model belongs to multiple categories, update each corresponding `data/reference*.json` file.

Possible extra integration points:

- `diffusers.pipelines.auto_pipeline.AUTO_*_PIPELINES_MAPPING` when task switching matters
- `pipe.task_args` when SD.Next needs default runtime kwargs such as `output_type`
- VAE hijack, TE hijack, or task-specific processors

## Loader Conventions

In `pipelines/model_<name>.py`:

- Use `sd_models.path_to_repo(checkpoint_info)`
- Call `sd_models.hf_auth_check(checkpoint_info)`
- Use `model_quant.get_dit_args(...)` for load args
- Respect `devices.dtype`
- Log meaningful loader details
- Reuse `generic.load_transformer(...)` and `generic.load_text_encoder(...)` when possible
- Set `pipe.task_args = {'output_type': 'np'}` when the pipeline should default to numpy output for SD.Next
- Clean up temporary references and call `devices.torch_gc(...)`

Do not hardcode assumptions about CUDA-only execution, local paths, or one-off environment state.

## Pipeline Conventions

When building a custom pipeline:

- Inherit from `diffusers.DiffusionPipeline`
- Register modules with `DiffusionPipeline.register_modules(...)`
- Keep prompt encoding and output conversion inside the pipeline
- Support `prompt`, `negative_prompt`, `generator`, `output_type`, and `return_dict` when the task is text-to-image-like
- Expose only parameters that are part of the model’s actual sampling surface
- Preserve direct pixel-space output if the model does not use a VAE
- Use a VAE only if the model genuinely outputs latents

Do not add generic Stable Diffusion arguments that the model does not support.

## Validation Checklist

After implementation, validate in this order:

1. Syntax or bytecode compilation for new files
2. Focused linting on touched files
3. Import-level smoke test for new modules when safe
4. Loader-path validation without doing a full generation if the model is too large
5. One real load and generation pass if feasible

Always report what was validated and what was not.

## Reference Asset Update Requirements

When the user asks to add or port a model for references, also perform these steps:

1. Update the correct `data/reference*.json` file(s) based on the user-confirmed category.
2. Create a placeholder zero-byte thumbnail file in `models/Reference` for the new model.

Notes:

- In this repo, the folder is `models/Reference` (capital `R`).
- Use a deterministic filename that matches the model entry naming convention used in the target reference JSON.
- If a real thumbnail already exists, do not overwrite it with a zero-byte file.

## CHANGELOG Update Requirement

When porting a new model or model family to SD.Next, also update `CHANGELOG.md`:

1. Locate the **Models** section under the most recent date heading (e.g., `## Update for 2026-04-10`)
2. Add a new bullet entry at the top of the Models list with:
   - Model name as a markdown link to the HF repo or official source
   - Short one-line description (1-2 sentences max)
   - Key details in italics on separate lines:
     - Unique architecture features, parameter counts, or efficiency claims
     - Variant information (e.g., *Normal*, *Edit*, *Lite* for distilled versions)
     - Notable text encoder or technology notes

Example format:
```
  - [BRIA FIBO](https://huggingface.co/briaai/FIBO) 8B parameter text-to-image model using Flow Matching  
    includes *Normal*, *Edit*, and *Lite* (distilled) variants  
    features lightweight SmolLM3-3B text encoder with efficient inference  
```

## Common Failure Modes

- Model type is added to `sd_models.py` but not `sd_detect.py`
- Loader exists but `modules/modeldata.py` still classifies the pipeline incorrectly
- A custom pipeline is named in a way that collides with a broader existing branch such as `Chroma`
- `torch_dtype` or other loader args are passed twice
- The code assumes a Diffusers repo layout when the source is really a single checkpoint file
- Negative prompt handling does not match prompt batch size
- Output is postprocessed as latents even though the model is pixel-space
- Custom config inference uses fragile defaults without clear failure paths

## Output Expectations

When using this skill, the final implementation should usually include:

- A clear integration path choice
- The minimal set of new files required
- SD.Next routing updates
- Targeted validation results
- Any remaining runtime risks called out explicitly

## Example Request Shapes

- "Port this standalone inference script into an SD.Next Diffusers pipeline"
- "Add support for this Hugging Face model repo to SD.Next"
- "Wire this upstream Diffusers pipeline into SD.Next autodetect and loading"
- "Convert this single-file checkpoint model into a custom Diffusers pipeline for SD.Next"
