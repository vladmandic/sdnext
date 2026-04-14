"""Z-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``zimage`` in ``allow_native``). Reads the
safetensors directly and writes into sdnext's existing
``network_layer_mapping``, returning a ``Network`` populated with
``NetworkModule*`` entries that ``network_activate`` will apply. If the
setting is on, the diffusers PEFT path handles the file instead.

Entry points, one per family:

- LoRA (+ DoRA) via :func:`try_load_lora`
- LoKR via :func:`try_load_lokr`
- LoHA via :func:`try_load_loha` (not yet validated against a real Z-Image adapter)
- OFT  via :func:`try_load_oft` (not yet validated against a real Z-Image adapter)

Recognized key prefixes for every family: ``diffusion_model.``,
``transformer.``, ``lora_unet_``, or bare. Diffusers-PEFT ``lora_A``/``lora_B``
are normalized to ``lora_down``/``lora_up``.

Pre-refactor Z-Image attention layouts (fused ``attention.qkv``, bare
``attention.out`` / ``attention.wo``) are rewritten to the current diffusers
``to_q``/``to_k``/``to_v`` and ``to_out.0``. For LoRA the fused qkv up-weight
is chunked along dim 0 at load time. For LoKR the split is deferred to apply
time via :class:`NetworkModuleLokrChunk`, which materializes ``kron(w1, w2)``
once per forward pass and returns the designated slice.

Fused ``attention.qkv`` for LoHA and OFT is skipped with a warning: no
``NetworkModuleHadaChunk`` exists, and an OFT block structure is tied to the
target module's ``out_features``, so a Q/K/V split is not a drop-in.
"""

import os
import time
import torch
from modules import shared, sd_models
from modules.logger import log
from modules.lora import network, network_lora, network_lokr, network_hada, network_oft, lora_convert
from modules.lora import lora_common as l


KNOWN_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_")

# Every family also picks up the universal optional keys
# (alpha, scale, bias, dora_scale) via base NetworkModule.__init__.
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

# Presence of any of these substrings anywhere in a key marks a file as belonging to that family.
LORA_MARKERS = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight")
LOKR_MARKERS = (".lokr_w1", ".lokr_w2")
LOHA_MARKERS = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
OFT_MARKERS = (".oft_blocks", ".oft_diag")

SUFFIX_NORMALIZE = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}

ATTENTION_OUT_ALIASES = ("attention_out", "attention_out_0", "attention_wo")
ATTENTION_OUT_TARGET = "attention_to_out_0"
ATTENTION_QKV_SUFFIX = "attention_qkv"
ATTENTION_QKV_TARGETS = ("attention_to_q", "attention_to_k", "attention_to_v")


def try_load_lora(name, network_on_disk, lora_scale):
    """Try loading a Z-Image LoRA (plus DoRA) as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, LORA_SUFFIXES)
    groups = expand_legacy_attention_lora(groups)

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
    """Try loading a Z-Image LoKR as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOKR_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, LOKR_SUFFIXES)
    groups, chunk_info = expand_legacy_attention_lokr(groups)

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
        chunked = chunk_info.get(network_key)
        if chunked is not None:
            idx, num = chunked
            net.modules[network_key] = network_lokr.NetworkModuleLokrChunk(net, nw, idx, num)
        else:
            net.modules[network_key] = network_lokr.NetworkModuleLokr(net, nw)

    return finalize_network(net, name, 'LoKR', lora_scale, t0, unmapped=unmapped)


def try_load_loha(name, network_on_disk, lora_scale):
    """Try loading a Z-Image LoHA as native modules. Fused attention.qkv groups are skipped."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOHA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, LOHA_SUFFIXES)
    groups = rename_attention_out(groups)

    unmapped = 0
    skipped_qkv = 0
    for network_key, w in groups.items():
        if network_key.endswith("_" + ATTENTION_QKV_SUFFIX):
            log.warning(f'Network load: type=LoHA name="{name}" key={network_key} fused qkv skipped (unsupported)')
            skipped_qkv += 1
            continue
        if not all(k in w for k in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")):
            continue
        sd_module = mapping.get(network_key)
        if sd_module is None:
            unmapped += 1
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
        net.modules[network_key] = network_hada.NetworkModuleHada(net, nw)

    return finalize_network(net, name, 'LoHA', lora_scale, t0, unmapped=unmapped, skipped=skipped_qkv)


def try_load_oft(name, network_on_disk, lora_scale):
    """Try loading a Z-Image OFT adapter as native modules. Fused attention.qkv groups are skipped."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, OFT_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, OFT_SUFFIXES)
    groups = rename_attention_out(groups)

    unmapped = 0
    skipped_qkv = 0
    for network_key, w in groups.items():
        if network_key.endswith("_" + ATTENTION_QKV_SUFFIX):
            log.warning(f'Network load: type=OFT name="{name}" key={network_key} fused qkv skipped (unsupported)')
            skipped_qkv += 1
            continue
        if not ("oft_blocks" in w or "oft_diag" in w):
            continue
        sd_module = mapping.get(network_key)
        if sd_module is None:
            unmapped += 1
            continue
        nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
        net.modules[network_key] = network_oft.NetworkModuleOFT(net, nw)

    return finalize_network(net, name, 'OFT', lora_scale, t0, unmapped=unmapped, skipped=skipped_qkv)


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

    Returns ``{network_key: {suffix: tensor, ...}}`` where ``network_key`` follows
    the sdnext convention ``lora_transformer_<path_with_underscores>``. Only keys
    whose suffix appears in ``suffixes`` are kept; ``lora_A``/``lora_B`` are
    normalized to ``lora_down``/``lora_up``.
    """
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        parsed = parse_key(key, suffixes)
        if parsed is None:
            continue
        network_key, suffix = parsed
        slot = groups.get(network_key)
        if slot is None:
            slot = {}
            groups[network_key] = slot
        slot[suffix] = value
    return groups


def parse_key(key, suffixes):
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
    network_key = 'lora_transformer_' + base.replace('.', '_')
    return network_key, suffix


def rename_attention_out(groups):
    """Rename legacy ``attention.out`` / ``attention.wo`` keys to ``attention.to_out.0``."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, w in groups.items():
        new_key = None
        for alias in ATTENTION_OUT_ALIASES:
            if key.endswith("_" + alias):
                new_key = key[: -len(alias)] + ATTENTION_OUT_TARGET
                break
        out[new_key or key] = w
    return out


def expand_legacy_attention_lora(groups):
    """Rename attention.out, then split fused attention.qkv into three per-projection groups.

    The down-weight is shared across Q/K/V and the up-weight is chunked along
    dim 0 (the concatenated output dim in the fused layout).
    """
    groups = rename_attention_out(groups)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, w in groups.items():
        if not key.endswith("_" + ATTENTION_QKV_SUFFIX):
            out[key] = w
            continue
        stem = key[: -len(ATTENTION_QKV_SUFFIX)]
        down = w.get("lora_down.weight")
        up = w.get("lora_up.weight")
        if down is None or up is None or up.shape[0] % 3 != 0:
            out[key] = w
            continue
        chunks = torch.chunk(up, 3, dim=0)
        alpha = w.get("alpha")
        dora = w.get("dora_scale")
        for target, chunk in zip(ATTENTION_QKV_TARGETS, chunks):
            split_key = stem + target
            split = {
                "lora_down.weight": down,
                "lora_up.weight": chunk.contiguous(),
            }
            if alpha is not None:
                split["alpha"] = alpha
            if dora is not None:
                split["dora_scale"] = dora
            out[split_key] = split
    return out


def expand_legacy_attention_lokr(groups):
    """Rename attention.out, then split fused attention.qkv for LoKR.

    For LoKR the split is deferred to apply time via ``NetworkModuleLokrChunk``
    (returned via a parallel ``chunk_info`` dict keyed by the split network key).
    The three resulting groups share the same tensor dict by shallow copy; the
    chunk module computes ``kron(w1, w2)`` once and slices the designated row
    range.

    Returns ``(groups, chunk_info)`` where ``chunk_info[network_key] == (index, num)``.
    """
    groups = rename_attention_out(groups)
    out: dict[str, dict[str, torch.Tensor]] = {}
    chunk_info: dict[str, tuple[int, int]] = {}
    for key, w in groups.items():
        if not key.endswith("_" + ATTENTION_QKV_SUFFIX):
            out[key] = w
            continue
        stem = key[: -len(ATTENTION_QKV_SUFFIX)]
        for i, target in enumerate(ATTENTION_QKV_TARGETS):
            split_key = stem + target
            out[split_key] = dict(w)
            chunk_info[split_key] = (i, 3)
    return out, chunk_info
