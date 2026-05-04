"""Chroma native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``chroma`` in ``allow_native``). Reads the
safetensors directly and writes into sdnext's existing
``network_layer_mapping``, returning a ``Network`` populated with
``NetworkModule*`` entries that ``network_activate`` will apply. If the
setting is on, the diffusers PEFT path handles the file instead.

Entry points, one per family:

- LoRA (+ DoRA) via :func:`try_load_lora`
- LoKR via :func:`try_load_lokr`
- LoHA via :func:`try_load_loha` (fused groups skipped, no chunk variant)
- OFT  via :func:`try_load_oft`  (fused groups skipped, no chunk variant)

Recognized key prefixes: ``diffusion_model.``, ``transformer.``, ``lora_unet_``.
Diffusers-PEFT ``lora_A``/``lora_B`` are normalized to ``lora_down``/``lora_up``.

Chroma LoRAs are trained against the Flux block layout regardless of which
key style they save in:

- ``double_blocks.{i}.{img,txt}_attn.{proj,qkv}``
- ``double_blocks.{i}.{img,txt}_mlp.{0,2}``
- ``single_blocks.{i}.{linear1,linear2}``

The diffusers ``ChromaTransformer2DModel`` exposes split-attention modules at:

- ``transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}``
- ``transformer_blocks.{i}.{ff,ff_context}.net.{0.proj,2}``
- ``single_transformer_blocks.{i}.attn.{to_q,to_k,to_v}``
- ``single_transformer_blocks.{i}.{proj_mlp,proj_out}``

This loader path-rewrites Flux-layout keys to diffusers names and expands
fused QKV at load time (LoRA) or apply time (LoKR via
:class:`NetworkModuleLokrSliceChunk`). For LoHA/OFT the fused groups are
skipped with a warning.

Chroma's modulation generator is the central ``distilled_guidance_layer``
approximator (replacing Flux's per-block ``norm1.linear``). The pruned
AdaLN classes have no ``.linear`` submodule, so any ``_mod_lin`` /
``_modulation_lin`` keys naturally land in ``unmapped`` and are reported.
LoRAs that target the approximator itself work without special casing,
since ``distilled_guidance_layer.<...>`` is a real module path that
``assign_network_names_to_compvis_modules`` registers.
"""

import os
import time
import torch
from modules import shared, sd_models
from modules.logger import log
from modules.lora import network, network_lora, network_lokr, network_hada, network_oft, lora_convert
from modules.lora import lora_common as l


KNOWN_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_")

LORA_SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight",    ".lora_B.weight",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
LOKR_SUFFIXES = (
    ".lokr_w1", ".lokr_w2",
    ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2_a", ".lokr_w2_b",
    ".lokr_t2",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
LOHA_SUFFIXES = (
    ".hada_w1_a", ".hada_w1_b",
    ".hada_w2_a", ".hada_w2_b",
    ".hada_t1",   ".hada_t2",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
OFT_SUFFIXES = (
    ".oft_blocks", ".oft_diag",
    ".alpha", ".dora_scale", ".bias", ".scale",
)

LORA_MARKERS = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight")
LOKR_MARKERS = (".lokr_w1", ".lokr_w2")
LOHA_MARKERS = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
OFT_MARKERS = (".oft_blocks", ".oft_diag")

SUFFIX_NORMALIZE = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}

# Default block counts for Chroma1-HD; overridden at runtime from the live
# transformer's config when available.
DEFAULT_NUM_DOUBLE_LAYERS = 19
DEFAULT_NUM_SINGLE_LAYERS = 38

# Fused QKV split dims. Single blocks fuse Q/K/V plus proj_mlp into linear1;
# the last chunk is unequal (12288 vs 3072 for Q/K/V).
QKV_DIMS = [3072, 3072, 3072]
LINEAR1_DIMS = [3072, 3072, 3072, 12288]

# Static (non-fused) renames from underscore-flat Flux-layout paths to
# underscore-flat diffusers paths. Built per-call by format()-ing the layer index.
DOUBLE_RENAME_TEMPLATES = {
    'double_blocks_{i}_img_attn_proj':  'transformer_blocks_{i}_attn_to_out_0',
    'double_blocks_{i}_img_mlp_0':      'transformer_blocks_{i}_ff_net_0_proj',
    'double_blocks_{i}_img_mlp_2':      'transformer_blocks_{i}_ff_net_2',
    'double_blocks_{i}_txt_attn_proj':  'transformer_blocks_{i}_attn_to_add_out',
    'double_blocks_{i}_txt_mlp_0':      'transformer_blocks_{i}_ff_context_net_0_proj',
    'double_blocks_{i}_txt_mlp_2':      'transformer_blocks_{i}_ff_context_net_2',
}
SINGLE_RENAME_TEMPLATES = {
    'single_blocks_{i}_linear2': 'single_transformer_blocks_{i}_proj_out',
}

# Fused-target qkv mappings. Double-block qkv fans out to img-side (to_*) and
# context-side (add_*_proj). Single-block linear1 fans out to single attn and
# proj_mlp.
DOUBLE_IMG_QKV_TARGETS = ('attn_to_q', 'attn_to_k', 'attn_to_v')
DOUBLE_TXT_QKV_TARGETS = ('attn_add_q_proj', 'attn_add_k_proj', 'attn_add_v_proj')
SINGLE_LINEAR1_TARGETS = ('attn_to_q', 'attn_to_k', 'attn_to_v', 'proj_mlp')


def build_static_rename(num_double, num_single):
    """Return {flux_flat_name: diffusers_flat_name} for non-fused paths."""
    out = {}
    for i in range(num_double):
        for src, dst in DOUBLE_RENAME_TEMPLATES.items():
            out[src.format(i=i)] = dst.format(i=i)
    for i in range(num_single):
        for src, dst in SINGLE_RENAME_TEMPLATES.items():
            out[src.format(i=i)] = dst.format(i=i)
    return out


def get_block_counts():
    """Read num_layers / num_single_layers from the live transformer, with fallback."""
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    transformer = getattr(sd_model, 'transformer', None)
    config = getattr(transformer, 'config', None)
    num_double = getattr(config, 'num_layers', DEFAULT_NUM_DOUBLE_LAYERS) if config is not None else DEFAULT_NUM_DOUBLE_LAYERS
    num_single = getattr(config, 'num_single_layers', DEFAULT_NUM_SINGLE_LAYERS) if config is not None else DEFAULT_NUM_SINGLE_LAYERS
    return num_double, num_single


class NetworkModuleLokrSliceChunk(network_lokr.NetworkModuleLokr):
    """LoKR module that returns one row-range of the Kronecker product.

    Used when a LoKR adapter targets a fused weight with unequal chunk sizes
    (e.g., Chroma single ``linear1`` = Q/K/V/proj_mlp at dims [3072, 3072, 3072, 12288]).
    The shared ``NetworkModuleLokrChunk`` only supports equal-sized chunks via
    ``torch.chunk``; this variant slices an explicit row range.
    """
    def __init__(self, net, weights, start_row, end_row):
        super().__init__(net, weights)
        self.start_row = start_row
        self.end_row = end_row

    def calc_updown(self, target):
        if self.w1 is not None:
            w1 = self.w1.to(target.device, dtype=target.dtype)
        else:
            w1a = self.w1a.to(target.device, dtype=target.dtype)
            w1b = self.w1b.to(target.device, dtype=target.dtype)
            w1 = w1a @ w1b
        if self.w2 is not None:
            w2 = self.w2.to(target.device, dtype=target.dtype)
        else:
            from modules.lora import lyco_helpers
            if self.t2 is None:
                w2a = self.w2a.to(target.device, dtype=target.dtype)
                w2b = self.w2b.to(target.device, dtype=target.dtype)
                w2 = w2a @ w2b
            else:
                t2 = self.t2.to(target.device, dtype=target.dtype)
                w2a = self.w2a.to(target.device, dtype=target.dtype)
                w2b = self.w2b.to(target.device, dtype=target.dtype)
                w2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)
        full_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        updown = network_lokr.make_kron(full_shape, w1, w2)
        updown = updown[self.start_row:self.end_row]
        output_shape = list(updown.shape)
        return self.finalize_updown(updown, target, output_shape)


def try_load_lora(name, network_on_disk, lora_scale):
    """Try loading a Chroma LoRA (plus DoRA) as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    static_rename = build_static_rename(*get_block_counts())

    groups = group_by_suffixes(state_dict, LORA_SUFFIXES)
    groups = expand_chroma_fused_lora(groups)
    groups = apply_static_rename(groups, static_rename)

    unmapped = 0
    shape_mismatch = 0
    for network_key, w in groups.items():
        if 'lora_down.weight' not in w or 'lora_up.weight' not in w:
            continue
        sd_module = mapping.get(network_key)
        if sd_module is None:
            unmapped += 1
            continue
        if not shapes_match(sd_module, w['lora_down.weight'], w['lora_up.weight']):
            log.warning(f'Network load: type=LoRA name="{name}" key={network_key} shape mismatch')
            shape_mismatch += 1
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
        net.modules[network_key] = network_lora.NetworkModuleLora(net, nw)

    return finalize_network(net, name, 'LoRA', lora_scale, t0, unmapped=unmapped, mismatch=shape_mismatch)


def try_load_lokr(name, network_on_disk, lora_scale):
    """Try loading a Chroma LoKR as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOKR_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    static_rename = build_static_rename(*get_block_counts())

    groups = group_by_suffixes(state_dict, LOKR_SUFFIXES)
    groups, slice_info = expand_chroma_fused_lokr(groups)
    groups = apply_static_rename(groups, static_rename)
    slice_info = {static_rename.get(k, k): v for k, v in slice_info.items()}

    unmapped = 0
    for network_key, w in groups.items():
        has_1 = "lokr_w1" in w or ("lokr_w1_a" in w and "lokr_w1_b" in w)
        has_2 = "lokr_w2" in w or ("lokr_w2_a" in w and "lokr_w2_b" in w)
        if not (has_1 and has_2):
            continue
        sd_module = mapping.get(network_key)
        if sd_module is None:
            unmapped += 1
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
        rng = slice_info.get(network_key)
        if rng is not None:
            start, end = rng
            net.modules[network_key] = NetworkModuleLokrSliceChunk(net, nw, start, end)
        else:
            net.modules[network_key] = network_lokr.NetworkModuleLokr(net, nw)

    return finalize_network(net, name, 'LoKR', lora_scale, t0, unmapped=unmapped)


def try_load_loha(name, network_on_disk, lora_scale):
    """Try loading a Chroma LoHA as native modules. Fused qkv/linear1 groups are skipped."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOHA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    static_rename = build_static_rename(*get_block_counts())

    groups = group_by_suffixes(state_dict, LOHA_SUFFIXES)
    groups, skipped = drop_chroma_fused_groups(groups, family='LoHA', name=name)
    groups = apply_static_rename(groups, static_rename)

    unmapped = 0
    for network_key, w in groups.items():
        if not all(k in w for k in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")):
            continue
        sd_module = mapping.get(network_key)
        if sd_module is None:
            unmapped += 1
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
        net.modules[network_key] = network_hada.NetworkModuleHada(net, nw)

    return finalize_network(net, name, 'LoHA', lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_oft(name, network_on_disk, lora_scale):
    """Try loading a Chroma OFT adapter as native modules. Fused qkv/linear1 groups are skipped."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, OFT_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    static_rename = build_static_rename(*get_block_counts())

    groups = group_by_suffixes(state_dict, OFT_SUFFIXES)
    groups, skipped = drop_chroma_fused_groups(groups, family='OFT', name=name)
    groups = apply_static_rename(groups, static_rename)

    unmapped = 0
    for network_key, w in groups.items():
        if not ("oft_blocks" in w or "oft_diag" in w):
            continue
        sd_module = mapping.get(network_key)
        if sd_module is None:
            unmapped += 1
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
        net.modules[network_key] = network_oft.NetworkModuleOFT(net, nw)

    return finalize_network(net, name, 'OFT', lora_scale, t0, unmapped=unmapped, skipped=skipped)


def has_marker(state_dict, markers):
    return any(any(m in k for m in markers) for k in state_dict)


def resolve_mapping():
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    return getattr(shared.sd_model, 'network_layer_mapping', {}) or {}


def new_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    return net


def finalize_network(net, name, family, lora_scale, t0, unmapped=0, mismatch=0, skipped=0):
    if len(net.modules) == 0:
        if unmapped or mismatch or skipped:
            log.debug(
                f'Network load: type={family} name="{name}" native no-match'
                f' unmapped={unmapped} mismatch={mismatch} skipped={skipped}'
            )
        return None
    log.debug(
        f'Network load: type={family} name="{name}" native modules={len(net.modules)}'
        f' unmapped={unmapped} mismatch={mismatch} skipped={skipped} scale={lora_scale}'
    )
    l.timer.activate += time.time() - t0
    return net


def shapes_match(sd_module, down_w: torch.Tensor, up_w: torch.Tensor) -> bool:
    if not hasattr(sd_module, 'weight'):
        return False
    if hasattr(sd_module, 'sdnq_dequantizer'):
        mod_shape = sd_module.sdnq_dequantizer.original_shape
    else:
        mod_shape = sd_module.weight.shape
    if len(mod_shape) < 2 or len(down_w.shape) < 2 or len(up_w.shape) < 2:
        return False
    return down_w.shape[1] == mod_shape[1] and up_w.shape[0] == mod_shape[0]


def group_by_suffixes(state_dict, suffixes):
    """Group state_dict entries by target module.

    Returns ``{flat_key: {suffix: tensor, ...}}`` where the flat key follows
    the Flux pre-rename layout (``double_blocks_{i}_img_attn_qkv`` etc.).
    """
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        parsed = parse_key(key, suffixes)
        if parsed is None:
            continue
        flat_key, suffix = parsed
        slot = groups.get(flat_key)
        if slot is None:
            slot = {}
            groups[flat_key] = slot
        slot[suffix] = value
    return groups


def parse_key(key, suffixes):
    """Strip prefix and suffix, return (flat_key, normalized_suffix) or None."""
    stripped = key
    for p in KNOWN_PREFIXES:
        if key.startswith(p):
            stripped = key[len(p):]
            break

    matched_suffix = None
    split_at = -1
    for marker in suffixes:
        if stripped.endswith(marker):
            split_at = len(stripped) - len(marker)
            matched_suffix = marker.lstrip('.')
            break
    if split_at < 0:
        return None

    base = stripped[:split_at]
    if not base:
        return None

    suffix = SUFFIX_NORMALIZE.get(matched_suffix, matched_suffix)
    flat_key = base.replace('.', '_')
    return flat_key, suffix


def expand_chroma_fused_lora(groups):
    """Split fused QKV / linear1 LoRA groups into their per-projection components.

    Chroma LoRAs are trained against Flux's fused-attention layout, while the
    diffusers ``ChromaTransformer2DModel`` exposes split modules. The
    fused-attention LoRA convention shares ``down`` across the splits and
    concatenates ``up`` along dim 0; the inverse here splits ``up`` by the
    per-target dim list while copying ``down`` to each child.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, w in groups.items():
        if key.endswith('_img_attn_qkv') or key.endswith('_txt_attn_qkv'):
            stem = key[:-len('img_attn_qkv')] if key.endswith('_img_attn_qkv') else key[:-len('txt_attn_qkv')]
            targets = DOUBLE_IMG_QKV_TARGETS if key.endswith('_img_attn_qkv') else DOUBLE_TXT_QKV_TARGETS
            block_prefix = stem.replace('double_blocks_', 'transformer_blocks_')
            split_groups = split_fused_lora_group(w, QKV_DIMS, [block_prefix + t for t in targets])
            if split_groups is None:
                out[key] = w
                continue
            out.update(split_groups)
        elif key.endswith('_linear1') and 'single_blocks' in key:
            stem = key[:-len('linear1')]
            block_prefix = stem.replace('single_blocks_', 'single_transformer_blocks_')
            split_groups = split_fused_lora_group(w, LINEAR1_DIMS, [block_prefix + t for t in SINGLE_LINEAR1_TARGETS])
            if split_groups is None:
                out[key] = w
                continue
            out.update(split_groups)
        else:
            out[key] = w
    return out


def split_fused_lora_group(w, dims, target_keys):
    """Split a fused LoRA (down, up) into per-target groups by row dim.

    Returns ``{target_key: {suffix: tensor, ...}}`` or ``None`` if the input is
    malformed (missing tensors, up-weight rows don't sum to dims).
    """
    down = w.get('lora_down.weight')
    up = w.get('lora_up.weight')
    if down is None or up is None:
        return None
    if up.shape[0] != sum(dims):
        return None
    alpha = w.get('alpha')
    dora = w.get('dora_scale')
    bias = w.get('bias')
    scale = w.get('scale')
    out: dict[str, dict[str, torch.Tensor]] = {}
    start = 0
    for tk, d in zip(target_keys, dims):
        chunk_up = up[start:start + d].contiguous()
        start += d
        slot = {'lora_down.weight': down, 'lora_up.weight': chunk_up}
        if alpha is not None:
            slot['alpha'] = alpha
        if dora is not None:
            slot['dora_scale'] = dora
        if bias is not None:
            slot['bias'] = bias
        if scale is not None:
            slot['scale'] = scale
        out[tk] = slot
    return out


def expand_chroma_fused_lokr(groups):
    """Mark fused QKV / linear1 LoKR groups as slice-chunked.

    LoKR factorizations don't compose with row-splitting at load time without
    materializing the full Kronecker product. Instead, each target gets a
    shallow copy of the same tensor dict, plus an entry in ``slice_info`` that
    drives :class:`NetworkModuleLokrSliceChunk` to slice rows lazily on each
    forward pass.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    slice_info: dict[str, tuple[int, int]] = {}
    for key, w in groups.items():
        if key.endswith('_img_attn_qkv') or key.endswith('_txt_attn_qkv'):
            stem = key[:-len('img_attn_qkv')] if key.endswith('_img_attn_qkv') else key[:-len('txt_attn_qkv')]
            targets = DOUBLE_IMG_QKV_TARGETS if key.endswith('_img_attn_qkv') else DOUBLE_TXT_QKV_TARGETS
            block_prefix = stem.replace('double_blocks_', 'transformer_blocks_')
            assign_lokr_slices(out, slice_info, w, QKV_DIMS, [block_prefix + t for t in targets])
        elif key.endswith('_linear1') and 'single_blocks' in key:
            stem = key[:-len('linear1')]
            block_prefix = stem.replace('single_blocks_', 'single_transformer_blocks_')
            assign_lokr_slices(out, slice_info, w, LINEAR1_DIMS, [block_prefix + t for t in SINGLE_LINEAR1_TARGETS])
        else:
            out[key] = w
    return out, slice_info


def assign_lokr_slices(out, slice_info, w, dims, target_keys):
    start = 0
    for tk, d in zip(target_keys, dims):
        out[tk] = dict(w)
        slice_info[tk] = (start, start + d)
        start += d


def drop_chroma_fused_groups(groups, family, name):
    """Remove fused QKV / linear1 groups (no chunk variant for LoHA/OFT)."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    skipped = 0
    for key, w in groups.items():
        is_fused_qkv = key.endswith('_img_attn_qkv') or key.endswith('_txt_attn_qkv')
        is_fused_linear1 = key.endswith('_linear1') and 'single_blocks' in key
        if is_fused_qkv or is_fused_linear1:
            log.warning(f'Network load: type={family} name="{name}" key={key} fused group skipped (unsupported)')
            skipped += 1
            continue
        out[key] = w
    return out, skipped


def apply_static_rename(groups, static_rename):
    """Rewrite Flux-layout flat keys to diffusers flat keys, then prepend ``lora_transformer_``.

    Keys without an entry in ``static_rename`` are passed through unchanged
    (they may already be diffusers paths from PEFT-style files, or they may
    target the ``distilled_guidance_layer`` approximator). The final
    ``lora_transformer_`` prefix is added uniformly to match the format
    ``assign_network_names_to_compvis_modules`` registers.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, w in groups.items():
        renamed = static_rename.get(key, key)
        out['lora_transformer_' + renamed] = w
    return out
