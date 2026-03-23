"""Flux2/Klein-specific LoRA loading.

Handles:
- Bare BFL-format keys in state dicts (adds diffusion_model. prefix for converter)
- LoKR adapters via native module loading (bypasses diffusers PEFT system)

Installed via apply_patch() during pipeline loading.
"""

import os
import time
from modules import shared, sd_models
from modules.logger import log
from modules.lora import network, network_lokr, lora_convert
from modules.lora import lora_common as l


BARE_FLUX_PREFIXES = ("single_blocks.", "double_blocks.", "img_in.", "txt_in.",
                      "final_layer.", "time_in.", "single_stream_modulation.",
                      "double_stream_modulation_")

# BFL -> diffusers module path mapping for Flux2/Klein
F2_SINGLE_MAP = {
    'linear1': 'attn.to_qkv_mlp_proj',
    'linear2': 'attn.to_out',
}
F2_DOUBLE_MAP = {
    'img_attn.proj': 'attn.to_out.0',
    'txt_attn.proj': 'attn.to_add_out',
    'img_mlp.0': 'ff.linear_in',
    'img_mlp.2': 'ff.linear_out',
    'txt_mlp.0': 'ff_context.linear_in',
    'txt_mlp.2': 'ff_context.linear_out',
}
F2_QKV_MAP = {
    'img_attn.qkv': ('attn', ['to_q', 'to_k', 'to_v']),
    'txt_attn.qkv': ('attn', ['add_q_proj', 'add_k_proj', 'add_v_proj']),
}


def apply_lora_alphas(state_dict):
    """Bake kohya-format .alpha scaling into lora_down weights and remove alpha keys.

    Diffusers' Flux2 converter only handles lora_A/lora_B (or lora_down/lora_up) keys.
    Kohya-format LoRAs store per-layer alpha values as separate .alpha keys that the
    converter doesn't consume, causing a ValueError on leftover keys. This matches the
    approach used by _convert_kohya_flux_lora_to_diffusers for Flux 1.
    """
    alpha_keys = [k for k in state_dict if k.endswith('.alpha')]
    if not alpha_keys:
        return state_dict
    for alpha_key in alpha_keys:
        base = alpha_key[:-len('.alpha')]
        down_key = f'{base}.lora_down.weight'
        if down_key not in state_dict:
            continue
        down_weight = state_dict[down_key]
        rank = down_weight.shape[0]
        alpha = state_dict.pop(alpha_key).item()
        scale = alpha / rank
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        state_dict[down_key] = down_weight * scale_down
        up_key = f'{base}.lora_up.weight'
        if up_key in state_dict:
            state_dict[up_key] = state_dict[up_key] * scale_up
    remaining = [k for k in state_dict if k.endswith('.alpha')]
    if remaining:
        log.debug(f'Network load: type=LoRA stripped {len(remaining)} orphaned alpha keys')
        for k in remaining:
            del state_dict[k]
    return state_dict


def preprocess_f2_keys(state_dict):
    """Add 'diffusion_model.' prefix to bare BFL-format keys so
    Flux2LoraLoaderMixin's format detection routes them to the converter."""
    if any(k.startswith("diffusion_model.") or k.startswith("base_model.model.") for k in state_dict):
        return state_dict
    if any(k.startswith(p) for k in state_dict for p in BARE_FLUX_PREFIXES):
        log.debug('Network load: type=LoRA adding diffusion_model prefix for bare BFL-format keys')
        state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}
    return state_dict


def try_load_lokr(name, network_on_disk, lora_scale):
    """Try loading a Flux2/Klein LoRA as LoKR native modules.

    Returns a Network with native modules if the state dict contains LoKR keys,
    or None to fall through to the generic diffusers path.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not any('.lokr_w1' in k for k in state_dict):
        return None
    net = load_lokr_native(name, network_on_disk, state_dict)
    if len(net.modules) == 0:
        log.error(f'Network load: type=LoKR name="{name}" no modules matched')
        return None
    log.debug(f'Network load: type=LoKR name="{name}" native modules={len(net.modules)} scale={lora_scale}')
    l.timer.activate += time.time() - t0
    return net


def load_lokr_native(name, network_on_disk, state_dict):
    """Load Flux2 LoKR as native modules applied at inference time.

    Stores only the compact LoKR factors (w1, w2) and computes kron(w1, w2)
    on-the-fly during weight application. For fused QKV modules in double
    blocks, NetworkModuleLokrChunk computes the full Kronecker product and
    returns only its designated Q/K/V chunk, then frees the temporary.
    """
    prefix = "diffusion_model."
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    for key in list(state_dict.keys()):
        if not key.endswith('.lokr_w1'):
            continue
        if not key.startswith(prefix):
            continue

        base = key[len(prefix):].rsplit('.lokr_w1', 1)[0]
        lokr_weights = {}
        for suffix in ['lokr_w1', 'lokr_w2', 'lokr_w1_a', 'lokr_w1_b', 'lokr_w2_a', 'lokr_w2_b', 'lokr_t2', 'alpha']:
            full_key = f'{prefix}{base}.{suffix}'
            if full_key in state_dict:
                lokr_weights[suffix] = state_dict[full_key]

        parts = base.split('.')
        block_type, block_idx, module_suffix = parts[0], parts[1], '.'.join(parts[2:])

        targets = []  # (module_path, chunk_index, num_chunks)
        if block_type == 'single_blocks' and module_suffix in F2_SINGLE_MAP:
            path = f'single_transformer_blocks.{block_idx}.{F2_SINGLE_MAP[module_suffix]}'
            targets.append((path, None, None))
        elif block_type == 'double_blocks':
            if module_suffix in F2_DOUBLE_MAP:
                path = f'transformer_blocks.{block_idx}.{F2_DOUBLE_MAP[module_suffix]}'
                targets.append((path, None, None))
            elif module_suffix in F2_QKV_MAP:
                attn_prefix, proj_keys = F2_QKV_MAP[module_suffix]
                for i, proj_key in enumerate(proj_keys):
                    path = f'transformer_blocks.{block_idx}.{attn_prefix}.{proj_key}'
                    targets.append((path, i, len(proj_keys)))

        for module_path, chunk_index, num_chunks in targets:
            network_key = "lora_transformer_" + module_path.replace(".", "_")
            sd_module = sd_model.network_layer_mapping.get(network_key)
            if sd_module is None:
                log.warning(f'Network load: type=LoKR module not found in mapping: {network_key}')
                continue
            weights = network.NetworkWeights(
                network_key=network_key,
                sd_key=network_key,
                w=dict(lokr_weights),
                sd_module=sd_module,
            )
            if chunk_index is not None:
                net.modules[network_key] = network_lokr.NetworkModuleLokrChunk(net, weights, chunk_index, num_chunks)
            else:
                net.modules[network_key] = network_lokr.NetworkModuleLokr(net, weights)

    return net


patched = False


def apply_patch():
    """Patch Flux2LoraLoaderMixin.lora_state_dict to handle bare BFL-format keys.

    When a LoRA file has bare BFL keys (no diffusion_model. prefix), the original
    lora_state_dict won't detect them as AI toolkit format. This patch checks for
    bare keys after the original returns and adds the prefix + re-runs conversion.
    """
    global patched # pylint: disable=global-statement
    if patched:
        return
    patched = True

    from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin
    original_lora_state_dict = Flux2LoraLoaderMixin.lora_state_dict.__func__

    @classmethod # pylint: disable=no-self-argument
    def patched_lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = preprocess_f2_keys(pretrained_model_name_or_path_or_dict)
            pretrained_model_name_or_path_or_dict = apply_lora_alphas(pretrained_model_name_or_path_or_dict)
        elif isinstance(pretrained_model_name_or_path_or_dict, (str, os.PathLike)):
            path = str(pretrained_model_name_or_path_or_dict)
            if path.endswith('.safetensors'):
                try:
                    from safetensors import safe_open
                    with safe_open(path, framework="pt") as f:
                        keys = list(f.keys())
                    needs_load = (
                        any(k.endswith('.alpha') for k in keys)
                        or (not any(k.startswith("diffusion_model.") or k.startswith("base_model.model.") for k in keys)
                            and any(k.startswith(p) for k in keys for p in BARE_FLUX_PREFIXES))
                    )
                    if needs_load:
                        from safetensors.torch import load_file
                        sd = load_file(path)
                        sd = preprocess_f2_keys(sd)
                        pretrained_model_name_or_path_or_dict = apply_lora_alphas(sd)
                except Exception:
                    pass
        return original_lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs)

    Flux2LoraLoaderMixin.lora_state_dict = patched_lora_state_dict
