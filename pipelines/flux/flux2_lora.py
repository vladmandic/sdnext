"""Flux2/Klein-specific LoRA loading.

Handles:
- Kohya-format LoRA via native module loading (lora_unet_ prefix keys)
- LoKR adapters via native module loading (bypasses diffusers PEFT system)
- Bare BFL-format keys in state dicts (adds diffusion_model. prefix for converter)

Installed via apply_patch() during pipeline loading.
"""

import os
import time
import torch
from modules import shared, sd_models
from modules.logger import log
from modules.lora import network, network_lokr, network_lora, lora_convert
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


# Kohya underscore suffix -> BFL dot suffix (last underscore becomes dot)
# Used to convert kohya key fragments to look up F2_DOUBLE_MAP / F2_QKV_MAP
KOHYA_SUFFIX_MAP = {
    'img_attn_proj': 'img_attn.proj',
    'txt_attn_proj': 'txt_attn.proj',
    'img_attn_qkv': 'img_attn.qkv',
    'txt_attn_qkv': 'txt_attn.qkv',
    'img_mlp_0': 'img_mlp.0',
    'img_mlp_2': 'img_mlp.2',
    'txt_mlp_0': 'txt_mlp.0',
    'txt_mlp_2': 'txt_mlp.2',
}


def try_load_lora(name, network_on_disk, lora_scale):
    """Try loading a Flux2/Klein LoRA as native modules.

    Handles three key formats:
    - Kohya: lora_unet_double_blocks_0_img_attn_proj.lora_down.weight
    - AI toolkit (BFL): diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight
    - Diffusers PEFT: transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight

    Returns a Network with native modules, or None to fall through to the diffusers path.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    has_lora = any('.lora_down.' in k or '.lora_up.' in k or '.lora_A.' in k or '.lora_B.' in k for k in state_dict)
    if not has_lora:
        return None
    is_f2_keys = any(
        k.startswith(('lora_unet_single_blocks_', 'lora_unet_double_blocks_',
                       'diffusion_model.single_blocks.', 'diffusion_model.double_blocks.',
                       'transformer.single_transformer_blocks.', 'transformer.transformer_blocks.'))
        for k in state_dict
    )
    if not is_f2_keys:
        return None
    net = load_lora_native(name, network_on_disk, state_dict)
    if len(net.modules) == 0:
        return None
    log.debug(f'Network load: type=LoRA name="{name}" native modules={len(net.modules)} scale={lora_scale}')
    l.timer.activate += time.time() - t0
    return net


def _group_lora_keys(state_dict):
    """Group LoRA state dict keys into (targets, weights_dict) pairs.

    Normalizes all three formats into a common structure. Weight keys are
    normalized to lora_down.weight / lora_up.weight regardless of input naming.
    Returns list of (targets, weights_dict) where targets come from BFL->diffusers mapping.
    """
    # Detect format from first relevant key
    sample = next((k for k in state_dict if '.lora_' in k), None)
    if sample is None:
        return []

    if sample.startswith('lora_unet_'):
        return _group_kohya(state_dict)
    elif sample.startswith('diffusion_model.'):
        return _group_bfl(state_dict)
    elif sample.startswith('transformer.'):
        return _group_peft(state_dict)

    # Bare BFL keys (no prefix)
    if any(k.startswith(p) for k in state_dict for p in BARE_FLUX_PREFIXES):
        return _group_bfl(state_dict, prefix='')

    return []


def _normalize_weight_key(suffix):
    """lora_A.weight -> lora_down.weight, lora_B.weight -> lora_up.weight"""
    return suffix.replace('lora_A.', 'lora_down.').replace('lora_B.', 'lora_up.')


def _group_kohya(state_dict):
    """Group kohya-format keys (lora_unet_ prefix, underscored module names)."""
    groups = {}
    for key, weight in state_dict.items():
        if not key.startswith('lora_unet_'):
            continue
        base, _, suffix = key.partition('.')
        if not suffix:
            continue
        if base not in groups:
            groups[base] = {}
        groups[base][_normalize_weight_key(suffix)] = weight

    results = []
    for base, weights_dict in groups.items():
        if 'lora_down.weight' not in weights_dict:
            continue
        stripped = base[len('lora_unet_'):]
        targets = _kohya_key_to_targets(stripped)
        if targets:
            results.append((targets, weights_dict))
    return results


def _group_bfl(state_dict, prefix='diffusion_model.'):
    """Group BFL/AI-toolkit-format keys (dot-separated module names)."""
    groups = {}
    for key, weight in state_dict.items():
        if prefix and not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        # Split at lora boundary: double_blocks.0.img_attn.proj.lora_A.weight
        for marker in ('.lora_A.', '.lora_B.', '.lora_down.', '.lora_up.', '.alpha', '.dora_scale'):
            pos = stripped.find(marker)
            if pos != -1:
                base = stripped[:pos]
                suffix = stripped[pos + 1:] if stripped[pos + 1:] else marker[1:]  # handle bare .dora_scale / .alpha
                break
        else:
            continue
        if base not in groups:
            groups[base] = {}
        groups[base][_normalize_weight_key(suffix)] = weight

    results = []
    for base, weights_dict in groups.items():
        if 'lora_down.weight' not in weights_dict:
            continue
        targets = _bfl_key_to_targets(base)
        if targets:
            results.append((targets, weights_dict))
    return results


def _group_peft(state_dict):
    """Group diffusers PEFT-format keys (transformer. prefix, diffusers module names)."""
    groups = {}
    for key, weight in state_dict.items():
        if not key.startswith('transformer.'):
            continue
        stripped = key[len('transformer.'):]
        for marker in ('.lora_A.', '.lora_B.', '.lora_down.', '.lora_up.', '.alpha'):
            pos = stripped.find(marker)
            if pos != -1:
                module_path = stripped[:pos]
                suffix = stripped[pos + 1:]
                break
        else:
            continue
        if module_path not in groups:
            groups[module_path] = {}
        groups[module_path][_normalize_weight_key(suffix)] = weight

    results = []
    for module_path, weights_dict in groups.items():
        if 'lora_down.weight' not in weights_dict:
            continue
        # Already in diffusers path format — direct target, no mapping needed
        results.append(([(module_path, None, None)], weights_dict))
    return results


def load_lora_native(name, network_on_disk, state_dict):
    """Load Flux2/Klein LoRA as native modules from any supported key format."""
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    for targets, weights_dict in _group_lora_keys(state_dict):
        for module_path, chunk_index, num_chunks in targets:
            network_key = "lora_transformer_" + module_path.replace(".", "_")
            sd_module = sd_model.network_layer_mapping.get(network_key)
            if sd_module is None:
                continue

            w = {}
            if chunk_index is not None:
                up = weights_dict['lora_up.weight']
                chunks = torch.chunk(up, num_chunks, dim=0)
                w['lora_up.weight'] = chunks[chunk_index].contiguous()
                w['lora_down.weight'] = weights_dict['lora_down.weight']
            else:
                w['lora_up.weight'] = weights_dict['lora_up.weight']
                w['lora_down.weight'] = weights_dict['lora_down.weight']

            # Validate dimensions match the target module
            if hasattr(sd_module, 'weight'):
                if hasattr(sd_module, 'sdnq_dequantizer'):
                    mod_shape = sd_module.sdnq_dequantizer.original_shape
                else:
                    mod_shape = sd_module.weight.shape
                if w['lora_down.weight'].shape[1] != mod_shape[1] or w['lora_up.weight'].shape[0] != mod_shape[0]:
                    log.warning(f'Network load: type=LoRA shape mismatch: {network_key} lora={w["lora_down.weight"].shape[1]}x{w["lora_up.weight"].shape[0]} module={mod_shape[1]}x{mod_shape[0]}')
                    continue

            if 'alpha' in weights_dict:
                w['alpha'] = weights_dict['alpha']
            if 'dora_scale' in weights_dict:
                w['dora_scale'] = weights_dict['dora_scale']

            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_lora.NetworkModuleLora(net, nw)

    return net


def _kohya_key_to_targets(stripped):
    """Map a stripped kohya key to (diffusers_module_path, chunk_index, num_chunks) targets.

    Input examples: 'double_blocks_0_img_attn_proj', 'single_blocks_5_linear1'
    """
    targets = []

    if stripped.startswith('single_blocks_'):
        rest = stripped[len('single_blocks_'):]
        idx, _, suffix = rest.partition('_')
        if suffix in F2_SINGLE_MAP:
            targets.append((f'single_transformer_blocks.{idx}.{F2_SINGLE_MAP[suffix]}', None, None))

    elif stripped.startswith('double_blocks_'):
        rest = stripped[len('double_blocks_'):]
        idx, _, kohya_suffix = rest.partition('_')
        bfl_suffix = KOHYA_SUFFIX_MAP.get(kohya_suffix)
        if bfl_suffix is None:
            return targets
        if bfl_suffix in F2_DOUBLE_MAP:
            targets.append((f'transformer_blocks.{idx}.{F2_DOUBLE_MAP[bfl_suffix]}', None, None))
        elif bfl_suffix in F2_QKV_MAP:
            attn_prefix, proj_keys = F2_QKV_MAP[bfl_suffix]
            for i, proj_key in enumerate(proj_keys):
                targets.append((f'transformer_blocks.{idx}.{attn_prefix}.{proj_key}', i, len(proj_keys)))

    return targets


def _bfl_key_to_targets(base):
    """Map a BFL dot-separated key to (diffusers_module_path, chunk_index, num_chunks) targets.

    Input examples: 'double_blocks.0.img_attn.proj', 'single_blocks.5.linear1'
    Same mapping as LoKR uses.
    """
    targets = []
    parts = base.split('.')
    if len(parts) < 3:
        return targets

    block_type, block_idx, module_suffix = parts[0], parts[1], '.'.join(parts[2:])

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

    return targets


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
