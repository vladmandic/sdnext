"""MOSS-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``mossimage`` in ``allow_native``).

The legacy transformer uses ``layers.N.proj.*`` and ``text_embedder.*`` names;
ComfyUI-format fused ``layers.N.proj.qkv`` weights are split into q/k/v chunks.
The upstream-layout transformer uses ``context_refiner.*`` and
``transformer_blocks.*`` names. Its ``context_refiner.N.attn.to_qkv`` remains a
single fused module, while the main transformer blocks use separate q/k/v
projections. Other paths bind verbatim for dotted and kohya prefixes.
"""

import re

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

QKV_SPLIT = ("q", "k", "v")
LEGACY_LAYER_RE = re.compile(r"^layers([._])(\d+)\1proj\1(qkv|q|k|v|o|ffn_up_gate|ffn_down)$")


def _upstream_layer_path(separator, layer, projection):
    stem = f"transformer_blocks{separator}{layer}{separator}"
    if projection in QKV_SPLIT:
        return f"{stem}attn{separator}to_{projection}"
    if projection == "o":
        return f"{stem}attn{separator}to_out"
    if projection == "ffn_up_gate":
        return f"{stem}ff_in"
    if projection == "ffn_down":
        return f"{stem}ff_out"
    raise ValueError(f"Unsupported MOSS projection: {projection}")


def _uses_upstream_layout():
    from modules import shared
    transformer = getattr(shared.sd_model, "transformer", None)
    return hasattr(transformer, "context_refiner") and hasattr(transformer, "transformer_blocks")


def resolve_targets(prefix_used, base):
    """Return ``[(module_path, ChunkSpec | None), ...]`` for a parsed group key.

    For the passthrough prefixes (``transformer.`` and the bare sentinel) an
    empty result defers to the verbatim fallback in
    :func:`native_adapter.resolve_group_targets`.
    """
    sep = "_" if prefix_used == "lora_unet_" else "."
    match = LEGACY_LAYER_RE.match(base)
    if match is not None:
        source_sep, layer, projection = match.groups()
        upstream = _uses_upstream_layout()
        if projection == "qkv":
            targets = []
            for idx, name in enumerate(QKV_SPLIT):
                chunk = ChunkSpec(idx=idx, total=len(QKV_SPLIT))
                legacy_path = f"layers{source_sep}{layer}{source_sep}proj{source_sep}{name}"
                path = _upstream_layer_path(source_sep, layer, name) if upstream else legacy_path
                targets.append((path, chunk))
            return targets
        path = _upstream_layer_path(source_sep, layer, projection) if upstream else base
        return [(path, None)]
    if base.endswith(f"{sep}attn{sep}to_qkv"):
        return [(base, None)]
    if prefix_used in ("diffusion_model.", "lora_unet_"):
        return [(base, None)]
    return []


BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    arch_name="mossimage",
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
    """Run every MOSS-Image family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
