---
description: "Use when changing LoRA discovery, metadata detection, prompt parsing, loading strategy, activation, weight application, or UI integration."
name: "LoRA Integration Guidelines"
applyTo: "modules/lora/**/*.py, modules/extra_networks.py, modules/processing_prompt.py, modules/processing_diffusers.py, modules/prompt_parser.py, modules/ui_extra_networks_lora.py, pipelines/**/*_lora.py"
---
# LoRA Integration Guidelines

1. Trace the full lifecycle for every LoRA change: discovery and `NetworkOnDisk` metadata in `modules/lora/networks.py`, prompt-tag parsing in `modules/lora/extra_networks_lora.py`, method selection/loading in `modules/lora/lora_load.py`, and activation/restoration in `modules/lora/networks.py`. Follow the call sites in `modules/processing_diffusers.py`: activate before prompt encoding so text-encoder LoRAs affect embeddings, and deactivate through the established cleanup path after sampling.
2. Keep path, display name, alias, hash, and detected base-model version distinct. Changes to discovery or detection must preserve folder aliases, cache refresh, and lookup behavior used by prompt tags and the extra-network UI.
3. Preserve prompt semantics when parsing names, strengths, stepwise values, block weights, and component selectors. Keep parsed lists aligned by index and verify zero-strength or missing-component handling.
4. Route loading through `lora_overrides.get_method(...)` and the existing native, Diffusers, or Nunchaku path. A new model family or format must be supported by the selected loader and its component mapping; do not assume one generic weight format works for every architecture.
5. Keep loading and activation separate. Preserve cached templates versus per-use network instances, and make sure changed or removed networks are deactivated before new weights are applied.
6. Treat activation and deactivation as a matched lifecycle. Preserve clean-base weight restoration, adapter disable/unload behavior, fused-weight rules, quantized-layer constraints, and offload-hook handling on both apply and removal paths.
7. Validate the complete affected path on a compatible loaded model: discover the file, resolve its reference and method, load it, activate the intended components and strengths, then remove or change it and confirm weights/adapters and offload state are restored.
