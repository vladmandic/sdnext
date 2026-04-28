"""Anima native LoRA loader.

Handles three trainer formats observed in community Anima LoRAs:
- Kohya: lora_unet_blocks_0_self_attn_q_proj.lora_down.weight (+ .alpha)
- BFL/AI-toolkit: diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight
- Hybrid: BFL key shape with .alpha scalars, optionally with a Qwen3 text
  encoder branch under text_encoders.qwen3_06b.transformer.model.layers...

Routes paths to one of three live components (transformer, llm_adapter,
text_encoder) via lora_convert.assign_network_names_to_compvis_modules and
the resulting network_layer_mapping.

Path renames mirror the Cosmos 2.0 table from
diffusers.loaders.single_file_utils.convert_cosmos_transformer_checkpoint_to_diffusers,
transposed into flat (underscore) form so the rewritten path matches
network_layer_mapping keys without further conversion.
"""

import os
import time
from collections import OrderedDict
from modules import shared, sd_models
from modules.logger import log
from modules.lora import network, network_lora, lora_convert
from modules.lora import lora_common as l


KOHYA_PREFIX = 'lora_unet_'
BFL_ADAPTER_PREFIX = 'diffusion_model.llm_adapter.'
BFL_TRANSFORMER_PREFIX = 'diffusion_model.'
BFL_TE_PREFIX = 'text_encoders.qwen3_06b.transformer.model.'

LORA_MARKERS = ('.lora_down.', '.lora_up.', '.lora_A.', '.lora_B.')

# Cosmos 2.0 path rename, applied to underscore-flattened paths. Order
# mirrors diffusers TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0: longer
# substrings precede substrings nested inside them, so str.replace does
# not consume a fragment that a later rule still needs to match.
COSMOS_2_FLAT_RENAME = OrderedDict([
    ('t_embedder_1', 'time_embed_t_embedder'),
    ('t_embedding_norm', 'time_embed_norm'),
    ('blocks', 'transformer_blocks'),
    ('adaln_modulation_self_attn_1', 'norm1_linear_1'),
    ('adaln_modulation_self_attn_2', 'norm1_linear_2'),
    ('adaln_modulation_cross_attn_1', 'norm2_linear_1'),
    ('adaln_modulation_cross_attn_2', 'norm2_linear_2'),
    ('adaln_modulation_mlp_1', 'norm3_linear_1'),
    ('adaln_modulation_mlp_2', 'norm3_linear_2'),
    ('self_attn', 'attn1'),
    ('cross_attn', 'attn2'),
    ('q_proj', 'to_q'),
    ('k_proj', 'to_k'),
    ('v_proj', 'to_v'),
    ('output_proj', 'to_out_0'),
    ('q_norm', 'norm_q'),
    ('k_norm', 'norm_k'),
    ('mlp_layer1', 'ff_net_0_proj'),
    ('mlp_layer2', 'ff_net_2'),
    ('x_embedder_proj_1', 'patch_embed_proj'),
    ('final_layer_adaln_modulation_1', 'norm_out_linear_1'),
    ('final_layer_adaln_modulation_2', 'norm_out_linear_2'),
    ('final_layer_linear', 'proj_out'),
])


def try_load_lora(name, network_on_disk, lora_scale):
    """Try loading an Anima LoRA as native modules.

    Returns a Network with native modules, or None if the file is not a
    recognized Anima format. Recognition is gated on Anima-specific path
    fragments so a non-Anima file routed here under a mounted Anima model
    is rejected rather than force-loaded.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    has_lora = any(any(m in k for m in LORA_MARKERS) for k in state_dict)
    if not has_lora:
        return None
    is_anima = any(
        k.startswith(KOHYA_PREFIX + 'blocks_')
        or k.startswith(BFL_TRANSFORMER_PREFIX + 'blocks.')
        or k.startswith(BFL_ADAPTER_PREFIX)
        or k.startswith(BFL_TE_PREFIX)
        for k in state_dict
    )
    if not is_anima:
        return None
    state_dict = apply_lora_alphas(state_dict)
    sd_model = getattr(shared.sd_model, 'pipe', shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    matched = 0
    unmatched = 0
    unmatched_samples = []
    for base, weights in group_keys(state_dict):
        network_key = resolve_network_key(base)
        if network_key is None:
            unmatched += 1
            if len(unmatched_samples) < 5:
                unmatched_samples.append(base)
            continue
        sd_module = sd_model.network_layer_mapping.get(network_key)
        if sd_module is None:
            unmatched += 1
            if len(unmatched_samples) < 5:
                unmatched_samples.append(f'{base} (key={network_key})')
            continue
        if not shapes_match(sd_module, weights):
            log.warning(f'Network load: type=LoRA name="{name}" shape mismatch key={network_key}')
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=weights, sd_module=sd_module)
        net.modules[network_key] = network_lora.NetworkModuleLora(net, nw)
        matched += 1
    if matched == 0:
        return None
    log.debug(f'Network load: type=LoRA name="{name}" native modules={matched} unmatched={unmatched} scale={lora_scale}')
    if unmatched > 0 and l.debug:
        log.debug(f'Network load: type=LoRA name="{name}" unmatched_samples={unmatched_samples}')
    l.timer.activate += time.time() - t0
    return net


def apply_lora_alphas(state_dict):
    """Bake .alpha scalars into lora_down / lora_up.

    Native module loading consumes lora_down/lora_up directly; there is no
    alpha kwarg path. Mirrors flux2_lora.apply_lora_alphas, with kohya and
    BFL down/up names both supported (the hybrid format uses BFL keys with
    kohya-style .alpha).
    """
    alpha_keys = [k for k in state_dict if k.endswith('.alpha')]
    if not alpha_keys:
        return state_dict
    for alpha_key in alpha_keys:
        base = alpha_key[:-len('.alpha')]
        down_key = next((c for c in (f'{base}.lora_down.weight', f'{base}.lora_A.weight') if c in state_dict), None)
        if down_key is None:
            continue
        rank = state_dict[down_key].shape[0]
        alpha = state_dict.pop(alpha_key).item()
        scale = alpha / rank
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        state_dict[down_key] = state_dict[down_key] * scale_down
        up_key = next((c for c in (f'{base}.lora_up.weight', f'{base}.lora_B.weight') if c in state_dict), None)
        if up_key is not None:
            state_dict[up_key] = state_dict[up_key] * scale_up
    remaining = [k for k in state_dict if k.endswith('.alpha')]
    if remaining:
        log.debug(f'Network load: type=LoRA stripped {len(remaining)} orphaned alpha keys')
        for k in remaining:
            del state_dict[k]
    return state_dict


def normalize_weight_key(suffix):
    """Rewrite PEFT-style suffixes to kohya/native form (lora_A to lora_down, lora_B to lora_up)."""
    return suffix.replace('lora_A.', 'lora_down.').replace('lora_B.', 'lora_up.')


def group_keys(state_dict):
    """Group state-dict keys by base path and yield (base, weights) pairs.

    Bases keep their source key shape (kohya flat or BFL dotted) for
    component routing in resolve_network_key.
    """
    groups = {}
    for key, weight in state_dict.items():
        base = None
        suffix = None
        if key.startswith(KOHYA_PREFIX):
            base, _, suffix = key.partition('.')
        else:
            for marker in LORA_MARKERS:
                pos = key.find(marker)
                if pos != -1:
                    base = key[:pos]
                    suffix = key[pos + 1:]
                    break
        if base is None or suffix is None:
            continue
        groups.setdefault(base, {})[normalize_weight_key(suffix)] = weight
    for base, weights in groups.items():
        if 'lora_down.weight' in weights and 'lora_up.weight' in weights:
            yield base, weights


def resolve_network_key(base):
    """Map a grouped base path to the network_layer_mapping lookup key.

    Adapter and TE prefixes must be checked before the bare BFL transformer
    prefix because both are extensions of diffusion_model.
    """
    if base.startswith(KOHYA_PREFIX):
        flat = base[len(KOHYA_PREFIX):]
        return 'lora_transformer_' + cosmos_rename_flat(flat)
    if base.startswith(BFL_ADAPTER_PREFIX):
        rest = base[len(BFL_ADAPTER_PREFIX):]
        return 'lora_llm_adapter_' + rest.replace('.', '_')
    if base.startswith(BFL_TE_PREFIX):
        rest = base[len(BFL_TE_PREFIX):]
        return 'lora_te_' + rest.replace('.', '_')
    if base.startswith(BFL_TRANSFORMER_PREFIX):
        rest = base[len(BFL_TRANSFORMER_PREFIX):]
        flat = rest.replace('.', '_')
        return 'lora_transformer_' + cosmos_rename_flat(flat)
    return None


def cosmos_rename_flat(flat):
    """Apply the Cosmos 2.0 path rename to a flattened path."""
    for k, v in COSMOS_2_FLAT_RENAME.items():
        flat = flat.replace(k, v)
    return flat


def shapes_match(sd_module, weights):
    """Confirm LoRA rank dimensions line up with the live module weight."""
    if not hasattr(sd_module, 'weight'):
        return True
    if hasattr(sd_module, 'sdnq_dequantizer'):
        mod_shape = sd_module.sdnq_dequantizer.original_shape
    else:
        mod_shape = sd_module.weight.shape
    if len(mod_shape) < 2:
        return False
    return (
        weights['lora_down.weight'].shape[1] == mod_shape[1]
        and weights['lora_up.weight'].shape[0] == mod_shape[0]
    )
