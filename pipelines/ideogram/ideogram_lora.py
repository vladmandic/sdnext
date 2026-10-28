"""Ideogram 4 native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``ideogram4`` in ``allow_native``).

LoRAs save the checkpoint's fused attention (``layers.N.attention.qkv`` and
``attention.o``) under ``diffusion_model.`` or ``conditional_transformer.``;
the diffusers transformer splits it into ``to_q``/``to_k``/``to_v`` and
``to_out.0``. The qkv up-weight is taken in equal q/k/v thirds, ``o`` is
renamed, and ``feed_forward.w1-3``, ``adaln_modulation`` and the embedders
bind verbatim. Every binding targets the conditional ``transformer``.
"""

import re

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec
from pipelines.ideogram import QKV_TARGETS


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT + ("conditional_transformer.",)

FUSED_QKV_RE = re.compile(r"^(layers([._])\d+)\2attention\2qkv$")
FUSED_OUT_RE = re.compile(r"^(layers([._])\d+)\2attention\2o$")


def resolve_targets(prefix_used, base):
    """Return ``[(module_path, ChunkSpec | None), ...]`` for a parsed group key.

    For the passthrough prefixes (``transformer.`` and the bare sentinel) an
    empty result defers to the verbatim fallback in
    :func:`native_adapter.resolve_group_targets`.
    """
    match = FUSED_QKV_RE.match(base)
    if match is not None:
        stem, sep = match.groups()
        return [(f"{stem}{sep}attention{sep}{name}", ChunkSpec(idx=idx, total=len(QKV_TARGETS))) for idx, name in enumerate(QKV_TARGETS)]
    match = FUSED_OUT_RE.match(base)
    if match is not None:
        stem, sep = match.groups()
        return [(f"{stem}{sep}attention{sep}to_out{sep}0", None)]
    if prefix_used in ("diffusion_model.", "conditional_transformer.", "lora_unet_"):
        return [(base, None)]
    return []


BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    arch_name="ideogram4",
)


def try_load_lora(name, network_on_disk, lora_scale):
    return native_adapter.try_load_lora(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_lokr(name, network_on_disk, lora_scale):
    return native_adapter.try_load_lokr(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_loha(name, network_on_disk, lora_scale):
    return native_adapter.try_load_loha(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_oft(name, network_on_disk, lora_scale):
    return native_adapter.try_load_oft(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_ia3(name, network_on_disk, lora_scale):
    return native_adapter.try_load_ia3(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_glora(name, network_on_disk, lora_scale):
    return native_adapter.try_load_glora(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_norm(name, network_on_disk, lora_scale):
    return native_adapter.try_load_norm(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load_full(name, network_on_disk, lora_scale):
    return native_adapter.try_load_full(name, network_on_disk, lora_scale, **BIND_KWARGS)


def try_load(name, network_on_disk, lora_scale):
    """Run every Ideogram 4 family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
