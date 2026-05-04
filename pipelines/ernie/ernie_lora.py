"""ERNIE-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``ernieimage`` in ``allow_native``). Reads the
safetensors directly and writes into sdnext's existing
``network_layer_mapping``, returning a ``Network`` populated with
``NetworkModule*`` entries that ``network_activate`` will apply. If the
setting is on, the diffusers PEFT path handles the file instead.

Entry points, one per family:

- LoRA (+ DoRA) via :func:`try_load_lora`
- LoKR via :func:`try_load_lokr`
- LoHA via :func:`try_load_loha`
- OFT  via :func:`try_load_oft`

Recognized key prefixes for every family: ``diffusion_model.``,
``transformer.``, ``lora_unet_``, or bare. Diffusers-PEFT ``lora_A``/``lora_B``
are normalized to ``lora_down``/``lora_up``.

The ERNIE-Image transformer has separate ``self_attention.to_q``/``to_k``/
``to_v``/``to_out.0`` linear modules (no fused QKV layout), so no
chunk/split machinery is needed and all four families are supported uniformly.
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

LORA_MARKERS = (".lora_down.weight", ".lora_up.weight", ".lora_A.weight", ".lora_B.weight")
LOKR_MARKERS = (".lokr_w1", ".lokr_w2")
LOHA_MARKERS = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
OFT_MARKERS = (".oft_blocks", ".oft_diag")

SUFFIX_NORMALIZE = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}


def try_load_lora(name, network_on_disk, lora_scale):
    """Try loading an ERNIE-Image LoRA (plus DoRA) as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, LORA_SUFFIXES)

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
    """Try loading an ERNIE-Image LoKR as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOKR_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, LOKR_SUFFIXES)

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
        net.modules[network_key] = network_lokr.NetworkModuleLokr(net, nw)

    return finalize_network(net, name, 'LoKR', lora_scale, t0, unmapped=unmapped)


def try_load_loha(name, network_on_disk, lora_scale):
    """Try loading an ERNIE-Image LoHA as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, LOHA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, LOHA_SUFFIXES)

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

    return finalize_network(net, name, 'LoHA', lora_scale, t0, unmapped=unmapped)


def try_load_oft(name, network_on_disk, lora_scale):
    """Try loading an ERNIE-Image OFT adapter as native modules."""
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if not has_marker(state_dict, OFT_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)

    groups = group_by_suffixes(state_dict, OFT_SUFFIXES)

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

    return finalize_network(net, name, 'OFT', lora_scale, t0, unmapped=unmapped)


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


def finalize_network(net, name, family, lora_scale, t0, unmapped=0, mismatch=0):
    if len(net.modules) == 0:
        if unmapped or mismatch:
            log.debug(
                f'Network load: type={family} name="{name}" native no-match'
                f' unmapped={unmapped} mismatch={mismatch}'
            )
        return None
    log.debug(
        f'Network load: type={family} name="{name}" native modules={len(net.modules)}'
        f' unmapped={unmapped} mismatch={mismatch} scale={lora_scale}'
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
