"""Z-Image native LoRA loader.

Bypasses the diffusers PEFT pipeline entirely. Reads the safetensors directly,
maps each target module into sdnext's existing ``network_layer_mapping``, and
returns a ``Network`` populated with ``NetworkModuleLora`` entries that
``network_activate`` will apply.

Recognized LoRA key formats:

- AI Toolkit diffusers:   ``diffusion_model.layers.0.attention.to_q.lora_A.weight``
- AI Toolkit down/up:     ``diffusion_model.layers.0.attention.to_q.lora_down.weight``
- Kohya (lora_unet):      ``lora_unet_layers_0_attention_to_q.lora_down.weight``
- Bare transformer:       ``transformer.layers.0.attention.to_q.lora_down.weight``
- Bare (no prefix):       ``layers.0.attention.to_q.lora_A.weight``

Alpha and dora_scale keys are preserved and passed to ``NetworkModuleLora``
which applies the alpha/rank scale at weight-apply time.
"""

import os
import time
import torch
from modules import shared, sd_models
from modules.logger import log
from modules.lora import network, network_lora, lora_convert
from modules.lora import lora_common as l


KNOWN_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_")

WEIGHT_MARKERS = (
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight",    ".lora_B.weight",
)
META_MARKERS = (".alpha", ".dora_scale")

SUFFIX_NORMALIZE = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}

# Pre-refactor Z-Image module names map to current diffusers Attention submodule
# names. LoRAs trained against the original Lumina/Z-Image attention layout use
# fused `qkv` and bare `out` / `wo`; diffusers now exposes `to_q`/`to_k`/`to_v`
# and `to_out.0`. The fused qkv up-weight is chunked into three along dim 0.
ATTENTION_OUT_ALIASES = ("attention_out", "attention_out_0", "attention_wo")
ATTENTION_OUT_TARGET = "attention_to_out_0"
ATTENTION_QKV_SUFFIX = "attention_qkv"
ATTENTION_QKV_TARGETS = ("attention_to_q", "attention_to_k", "attention_to_v")


def try_load_lora(name, network_on_disk, lora_scale):
    """Try loading a Z-Image LoRA as native modules.

    Returns a ``Network`` if at least one module matched, otherwise ``None``.
    """
    t0 = time.time()
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')

    if not any(any(m in k for m in WEIGHT_MARKERS) for k in state_dict):
        return None

    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    mapping = getattr(shared.sd_model, 'network_layer_mapping', {}) or {}

    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    groups = group_state_dict(state_dict)
    groups = expand_legacy_attention(groups)
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

    if len(net.modules) == 0:
        if unmapped or shape_mismatch:
            log.debug(f'Network load: type=LoRA name="{name}" native no-match unmapped={unmapped} mismatch={shape_mismatch}')
        return None

    log.debug(
        f'Network load: type=LoRA name="{name}" native modules={len(net.modules)}'
        f' unmapped={unmapped} mismatch={shape_mismatch} scale={lora_scale}'
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


def group_state_dict(state_dict):
    """Group state_dict entries by target module.

    Returns ``{network_key: {suffix: tensor, ...}}`` with suffixes normalized to
    ``lora_down.weight`` / ``lora_up.weight`` (plus optional ``alpha`` / ``dora_scale``).
    ``network_key`` matches the sdnext ``network_layer_name`` convention
    (``lora_transformer_<path_with_underscores>``).
    """
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        parsed = parse_key(key)
        if parsed is None:
            continue
        network_key, suffix = parsed
        slot = groups.get(network_key)
        if slot is None:
            slot = {}
            groups[network_key] = slot
        slot[suffix] = value
    return groups


def expand_legacy_attention(groups):
    """Rewrite pre-refactor Z-Image attention module names.

    - ``..._attention_qkv`` groups are split into three separate
      ``..._attention_to_q/k/v`` groups. The down weight is shared and the up
      weight is chunked along dim 0 (which carries the concatenated q/k/v
      outputs in the fused layout).
    - ``..._attention_out``, ``..._attention_out_0`` and ``..._attention_wo``
      are renamed to ``..._attention_to_out_0``.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, w in groups.items():
        new_key = None
        for alias in ATTENTION_OUT_ALIASES:
            if key.endswith("_" + alias):
                new_key = key[: -len(alias)] + ATTENTION_OUT_TARGET
                break
        if new_key is not None:
            out[new_key] = w
            continue

        if key.endswith("_" + ATTENTION_QKV_SUFFIX):
            stem = key[: -len(ATTENTION_QKV_SUFFIX)]
            down = w.get("lora_down.weight")
            up = w.get("lora_up.weight")
            alpha = w.get("alpha")
            dora = w.get("dora_scale")
            if down is None or up is None or up.shape[0] % 3 != 0:
                out[key] = w
                continue
            chunks = torch.chunk(up, 3, dim=0)
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
            continue

        out[key] = w
    return out


def parse_key(key: str):
    """Parse one state_dict key into ``(network_key, suffix)`` or ``None``.

    The suffix is one of: ``lora_down.weight``, ``lora_up.weight``, ``alpha``,
    ``dora_scale``. Diffusers ``lora_A/lora_B`` aliases are normalized to
    ``lora_down/lora_up`` here.
    """
    stripped = key
    for p in KNOWN_PREFIXES:
        if key.startswith(p):
            stripped = key[len(p):]
            break

    split_at = -1
    raw_suffix = None
    for marker in WEIGHT_MARKERS + META_MARKERS:
        if stripped.endswith(marker):
            split_at = len(stripped) - len(marker)
            raw_suffix = marker.lstrip('.')
            break
    if split_at < 0:
        return None

    base = stripped[:split_at]
    if not base:
        return None

    suffix = SUFFIX_NORMALIZE.get(raw_suffix, raw_suffix)
    network_key = 'lora_transformer_' + base.replace('.', '_')
    return network_key, suffix
