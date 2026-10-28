"""MOSS-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``mossimage`` in ``allow_native``).

The legacy transformer uses ``layers.N.proj.*`` and ``text_embedder.*`` names;
ComfyUI-format fused ``layers.N.proj.qkv`` weights are split into q/k/v chunks.
The upstream-layout transformer uses ``context_refiner.*`` and
``transformer_blocks.*`` names. Its ``context_refiner.N.attn.to_qkv`` remains a
single fused module, while the main transformer blocks use separate q/k/v
projections. On that layout every legacy name is renamed to its upstream module
(``UPSTREAM_NAMES``). Other paths bind verbatim for dotted and kohya prefixes.
"""

import re

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

QKV_SPLIT = ("q", "k", "v")
LEGACY_QKV_RE = re.compile(r"^layers([._])(\d+)\1proj\1qkv$")
UPSTREAM_NAMES = tuple((re.compile(pattern), name) for pattern, name in ( # legacy module, underscored, to its upstream-layout module
    (r"^layers_(\d+)_proj_(q|k|v)$", r"transformer_blocks.\1.attn.to_\2"),
    (r"^layers_(\d+)_proj_o$", r"transformer_blocks.\1.attn.to_out"),
    (r"^layers_(\d+)_proj_ffn_up_gate$", r"transformer_blocks.\1.ff_in"),
    (r"^layers_(\d+)_proj_ffn_down$", r"transformer_blocks.\1.ff_out"),
    (r"^layers_(\d+)_(q|k)_norm$", r"transformer_blocks.\1.attn.norm_\2"),
    (r"^layers_(\d+)_norm(1|2)$", r"transformer_blocks.\1.norm\2"),
    (r"^text_embedder_blocks_(\d+)_qkv$", r"context_refiner.\1.attn.to_qkv"),
    (r"^text_embedder_blocks_(\d+)_proj$", r"context_refiner.\1.attn.to_out"),
    (r"^text_embedder_blocks_(\d+)_ffn_(gate|up|down)$", r"context_refiner.\1.ff_\2"),
    (r"^text_embedder_blocks_(\d+)_(q|k)_norm$", r"context_refiner.\1.attn.norm_\2"),
    (r"^text_embedder_blocks_(\d+)_norm(1|2)$", r"context_refiner.\1.norm\2"),
    (r"^text_embedder_proj$", "context_embedder"),
    (r"^video_embedder$", "x_embedder"),
    (r"^time_embedding_0$", "time_embedder.linear_1"),
    (r"^time_embedding_2$", "time_embedder.linear_2"),
    (r"^shared_block_modulation$", "block_modulation"),
    (r"^final_norm_video$", "norm_out"),
    (r"^final_linear_video$", "proj_out"),
    (r"^video_head_modulation$", "norm_out_modulation"),
))


def upstream_name(base, sep):
    """Upstream-layout module for a legacy module path, joined with ``sep``; ``None`` for a path with no legacy name."""
    flat = base.replace(".", "_")
    for pattern, name in UPSTREAM_NAMES:
        if pattern.match(flat):
            path = pattern.sub(name, flat)
            return path if sep == "." else path.replace(".", sep)
    return None


def _uses_upstream_layout():
    from modules import shared
    transformer = getattr(getattr(shared.sd_model, "pipe", shared.sd_model), "transformer", None)
    return hasattr(transformer, "context_refiner") and hasattr(transformer, "transformer_blocks")


def resolve_targets(prefix_used, base):
    """Return ``[(module_path, ChunkSpec | None), ...]`` for a parsed group key.

    For the passthrough prefixes (``transformer.`` and the bare sentinel) an
    empty result defers to the verbatim fallback in
    :func:`native_adapter.resolve_group_targets`.
    """
    sep = "_" if prefix_used == "lora_unet_" else "."
    upstream = _uses_upstream_layout()
    match = LEGACY_QKV_RE.match(base)
    if match is not None:
        source_sep, layer = match.groups()
        targets = []
        for idx, name in enumerate(QKV_SPLIT):
            legacy_path = f"layers{source_sep}{layer}{source_sep}proj{source_sep}{name}"
            path = upstream_name(legacy_path, source_sep) if upstream else legacy_path
            targets.append((path, ChunkSpec(idx=idx, total=len(QKV_SPLIT))))
        return targets
    renamed = upstream_name(base, sep) if upstream else None
    if renamed is not None:
        return [(renamed, None)]
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
