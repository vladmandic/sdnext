---
description: "Use when changing model detection, loading, component downloads, cache routing, quantization, or offload behavior."
name: "Model Loading Guidelines"
applyTo: "modules/sd_checkpoint.py, modules/sd_models.py, modules/sd_detect.py, modules/modelloader.py, modules/models_hf.py, modules/model*.py, modules/modular_load.py, modules/ui_models.py, modules/ui_models_load.py, modules/sd_models_utils.py, modules/sd_vae.py, modules/sd_unet.py, modules/sd_offload*.py, modules/api/endpoints.py, pipelines/**/*.py"
---
# Model Loading Guidelines

1. Use `modules/shared.py` as the source of truth for model and runtime options. Preserve environment-driven behavior and backend/device compatibility; do not assume CUDA or a single loading mode.
2. Classify each Hugging Face/Diffusers load as a full pipeline/checkpoint or an auxiliary component. Full pipeline loads use `shared.opts.diffusers_dir`; independently loaded components use `shared.opts.hfcache_dir`. Preserve the distinction when forwarding arguments through wrappers.
3. Preserve caller load configuration through the complete path, including dtype, revision, offline/auth settings, quantization arguments, and component overrides. Verify that options reach the final constructor instead of being silently dropped.
4. Keep detection and loading behavior aligned: supported model layouts, identifiers, and component names must route to a loader that can actually consume them. Reuse established loader and pipeline patterns instead of adding parallel dispatch paths.
5. Respect existing device placement and offload hooks. Avoid unconditional `.to(...)` calls or device assumptions that conflict with Accelerate, quantized modules, or configured offload.
6. When adding or auditing a model integration, use the relevant `port-model`, `check-models`, and `check-paths` skills for their focused workflows.

## Trace The Load Path

For a model-loading change, follow one concrete selection from its entrypoint through the object that becomes active. Do not stop at the first `from_pretrained(...)` or `from_single_file(...)` call.

1. **Selection and resolution:** Find the UI, API, startup, or reload caller. Follow `sd_checkpoint.select_checkpoint(...)` and `sd_models.reload_model_weights(...)` to the resolved `CheckpointInfo`; distinguish the base model, refiner, and other load operation.
2. **Detection:** Follow `sd_detect.detect_pipeline(...)` and verify both the selected pipeline class and detected model type against the actual checkpoint, folder layout, metadata, or configured override.
3. **Dispatch:** Follow `sd_models.load_diffuser(...)` through the selected branch: forced/custom loader, SDNQ, single-file, or folder load. Check `handled` and fallback behavior so a loader failure cannot be mistaken for a successful load or silently skip the intended path.
4. **Components and arguments:** Trace separately loaded VAE, UNet, text encoder, transformer, tokenizer, and other components to their final constructors. Preserve the full-pipeline versus auxiliary cache distinction and verify load options survive wrapper forwarding.
5. **Finalization:** Follow overrides/defaults, text-encoder and embedding setup, modular conversion, quantization, offload, compile, and model-loaded callbacks through to the model stored in runtime state. Confirm failure paths do not leave a partially initialized model looking active.
6. **Evidence:** Use the existing `Load`/`Autodetect` logs and load timers to confirm each expected stage ran. Add new tracking only when existing evidence cannot distinguish the failure or branch under investigation.
