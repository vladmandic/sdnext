"""Anima custom-transformer loader.

Called from :func:`pipelines.model_anima.load_anima` when the user has selected
a transformer file via the UNET dropdown (``shared.opts.sd_unet``). Reads the
safetensors directly, strips the BFL-style prefix, splits off the bundled
``llm_adapter.*`` keys, and routes the two halves into the diffusers
``CosmosTransformer3DModel`` and the remote ``AnimaLLMAdapter`` respectively.

The transformer half is run through diffusers'
``convert_cosmos_transformer_checkpoint_to_diffusers`` (Cosmos 2.0 branch),
whose rename table covers Anima's native key fragments exactly, so the
converted state dict drops cleanly into ``CosmosTransformer3DModel`` with no
ad-hoc renames needed here. The adapter half matches the base repo's
``llm_adapter/diffusion_pytorch_model.safetensors`` exactly, so it loads
as-is.

Supported input formats (safetensors only; GGUF and .pth are rejected early):

- Bare BFL keys: ``blocks.0.self_attn.q_proj.weight`` (e.g. ``rdbtAnima_v027``)
- ``model.diffusion_model.`` prefix (e.g. ``animaika_v35``)
- ``diffusion_model.`` prefix (ComfyUI-style export)
- ``net.`` prefix (NVIDIA/Cosmos native export, e.g. ``animayume_v04``)

Quantization: SDNQ (pre/post/auto) and ``layerwise_quantization`` are honored.
SDNQ pre-mode is applied post-load here because this path bypasses
``from_pretrained``, where ``quantization_config`` normally takes effect.
TensorRT (``NVIDIAModelOptConfig``) is not supported and is skipped with a
warning. GGUF would require a separate converter and is not supported.
"""

import os
import time
import diffusers
import huggingface_hub as hf
from modules import shared, devices, sd_models, model_quant, errors
from modules.logger import log


KNOWN_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.")
ADAPTER_PREFIX = "llm_adapter."
COSMOS_1_MARKER = "net.blocks.block1.blocks.0.block.attn.to_q.0.weight"

# Buffer keys that CosmosTransformer3DModel creates at __init__ time and do
# not appear in trainer state dicts. Acceptable in the "missing" set.
ACCEPTABLE_MISSING = ("rope.", "pos_embedder.", "learnable_pos_embed.")


def load_custom_transformer(repo_id, local_file, diffusers_load_config, adapter_cls):
    """Load a custom Anima transformer (and optional bundled adapter) from a safetensors file.

    Returns ``(transformer, llm_adapter_or_none)``. If the file does not bundle
    an adapter, the second element is ``None`` and the caller should fall back
    to the base repo's adapter via ``AnimaLLMAdapter.from_pretrained``.
    Raises on any hard failure (prefix mix, shape mismatch, missing configs).
    """
    t0 = time.time()

    if not local_file.lower().endswith('.safetensors'):
        raise ValueError(f'Load model: type=Anima custom transformer requires .safetensors, got "{local_file}"')

    # from_config + load_state_dict does not consume load_args (device_map,
    # torch_dtype, etc.); dtype is applied via explicit .to() below. Only
    # quant_type is read from this call.
    _, quant_args = model_quant.get_dit_args(
        diffusers_load_config, module='Model', device_map=True, allow_quant=True,
    )
    quant_type = model_quant.get_quant_type(quant_args)

    transformer_cfg = fetch_component_config(repo_id, 'transformer/config.json')
    adapter_cfg = fetch_component_config(repo_id, 'llm_adapter/config.json')

    state_dict = sd_models.read_state_dict(local_file, what='transformer')
    state_dict = strip_prefix(state_dict)
    transformer_sd, adapter_sd = partition_adapter(state_dict)
    del state_dict

    if COSMOS_1_MARKER in transformer_sd:
        raise ValueError(f'Load model: type=Anima custom transformer has unsupported Cosmos 1.0 structure (file="{local_file}")')

    log.info(f'Load model: type=Anima custom="{os.path.basename(local_file)}" transformer_keys={len(transformer_sd)} adapter_keys={len(adapter_sd)}')

    transformer = build_transformer(transformer_sd, transformer_cfg, quant_args, quant_type)
    del transformer_sd
    devices.torch_gc()

    if adapter_sd:
        llm_adapter = build_adapter(adapter_sd, adapter_cfg, adapter_cls)
    else:
        log.info('Load model: type=Anima custom transformer has no bundled adapter, caller will load from base repo')
        llm_adapter = None

    sd_models.allow_post_quant = False  # transformer already quantized above
    devices.torch_gc()
    log.debug(f'Load model: type=Anima custom transformer time={time.time()-t0:.2f}')
    return transformer, llm_adapter


def fetch_component_config(repo_id, relative_path):
    """Download and parse a component config.json from the base repo."""
    try:
        local = hf.hf_hub_download(repo_id, filename=relative_path, cache_dir=shared.opts.diffusers_dir)
    except Exception as e:
        raise RuntimeError(f'Load model: type=Anima failed to download {relative_path} from repo="{repo_id}": {e}') from e
    return shared.readfile(local, as_type='dict')


def strip_prefix(state_dict):
    """Detect and uniformly strip the BFL-style prefix from all keys.

    Supported prefixes (longest first, so ``model.diffusion_model.`` beats ``diffusion_model.``):
    ``model.diffusion_model.``, ``diffusion_model.``, or no prefix. Raises
    ValueError if some keys match the dominant prefix and others do not,
    since mixed prefixes indicate a malformed file.
    """
    counts = {p: sum(1 for k in state_dict if k.startswith(p)) for p in KNOWN_PREFIXES}
    total = len(state_dict)
    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        log.debug('Load model: type=Anima custom transformer prefix=bare')
        return state_dict
    if counts[dominant] != total:
        raise ValueError(
            f'Load model: type=Anima custom transformer has mixed prefixes '
            f'(total={total} {dominant}={counts[dominant]})'
        )
    log.debug(f'Load model: type=Anima custom transformer prefix="{dominant}"')
    offset = len(dominant)
    return {k[offset:]: v for k, v in state_dict.items()}


def partition_adapter(state_dict):
    """Split into (transformer_sd, adapter_sd) by the ``llm_adapter.`` prefix."""
    transformer_sd = {}
    adapter_sd = {}
    for key, value in state_dict.items():
        if key.startswith(ADAPTER_PREFIX):
            adapter_sd[key[len(ADAPTER_PREFIX):]] = value
        else:
            transformer_sd[key] = value
    return transformer_sd, adapter_sd


def build_transformer(transformer_sd, transformer_cfg, quant_args, quant_type):
    """Convert, instantiate, load, dtype-cast, quantize, and (if offloading) move to CPU."""
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
    try:
        converted = convert_cosmos_transformer_checkpoint_to_diffusers(transformer_sd)
        transformer = diffusers.CosmosTransformer3DModel.from_config(transformer_cfg)
        missing, unexpected = transformer.load_state_dict(converted, strict=False)
        validate_state_dict_load('transformer', missing, unexpected)
        del converted
        devices.torch_gc()
        transformer = transformer.to(dtype=devices.dtype)
    except Exception as e:
        log.error(f'Load model: type=Anima transformer load failed: {e}')
        errors.display(e, 'Load')
        raise

    apply_quant(transformer, quant_type)

    if shared.opts.diffusers_offload_mode != 'none':
        sd_models.move_model(transformer, devices.cpu)

    if not hasattr(transformer, 'quantization_config'):
        if hasattr(transformer, 'config') and hasattr(transformer.config, 'quantization_config'):
            transformer.quantization_config = transformer.config.quantization_config
        elif (quant_type is not None) and (quant_args.get('quantization_config', None) is not None):
            transformer.quantization_config = quant_args.get('quantization_config', None)
    return transformer


def build_adapter(adapter_sd, adapter_cfg, adapter_cls):
    """Instantiate AnimaLLMAdapter from the base repo config and load bundled weights."""
    try:
        adapter = adapter_cls.from_config(adapter_cfg)
        missing, unexpected = adapter.load_state_dict(adapter_sd, strict=False)
        validate_state_dict_load('adapter', missing, unexpected)
        adapter = adapter.to(dtype=devices.dtype)
    except Exception as e:
        log.error(f'Load model: type=Anima adapter load failed: {e}')
        errors.display(e, 'Load')
        raise
    if shared.opts.diffusers_offload_mode != 'none':
        sd_models.move_model(adapter, devices.cpu)
    return adapter


def validate_state_dict_load(component, missing, unexpected):
    """Raise ValueError if load_state_dict produced unexpected keys or non-buffer missing keys."""
    if unexpected:
        sample = ', '.join(unexpected[:5])
        raise ValueError(f'Load model: type=Anima {component} has {len(unexpected)} unexpected keys (sample: {sample})')
    hard_missing = [k for k in missing if not any(k.startswith(p) for p in ACCEPTABLE_MISSING)]
    if hard_missing:
        sample = ', '.join(hard_missing[:5])
        raise ValueError(f'Load model: type=Anima {component} missing {len(hard_missing)} required keys (sample: {sample})')
    if missing:
        log.debug(f'Load model: type=Anima {component} ignored {len(missing)} buffer-only missing keys')


def apply_quant(transformer, quant_type):
    """Apply SDNQ / layerwise quantization to the bare transformer.

    SDNQ 'pre' and 'auto' would normally route through ``quantization_config``
    at ``from_pretrained`` time; since we bypass that boundary, we call the
    per-module quant path directly. SDNQ 'post' and ``layerwise_quantization``
    go through ``do_post_load_quant`` as usual.
    """
    if quant_type == 'NVIDIAModelOptConfig':
        log.warning('Load model: type=Anima quant=TRT not supported on custom transformer path, skipping')
    elif quant_type == 'SDNQConfig':
        if shared.opts.sdnq_quantize_mode == 'pre':
            log.info('Load model: type=Anima quant=SDNQ pre-mode applied post-load on custom transformer path')
        model_quant.sdnq_quantize_model(transformer, op='transformer')
    # allow=False avoids double-applying SDNQ in auto mode (applied directly
    # above); post mode fires regardless of allow, and layerwise always fires.
    model_quant.do_post_load_quant(transformer, allow=False)
