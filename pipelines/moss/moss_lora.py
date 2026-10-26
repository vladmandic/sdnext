"""MOSS-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``mossimage`` in ``allow_native``).

The transformer keeps the official checkpoint names:
``layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}``, ``text_embedder.*``,
``time_embedding.*``, ``shared_block_modulation``, ``video_embedder``,
``final_linear_video`` and ``video_head_modulation``. Trainers built on the
ComfyUI model (ai-toolkit, diffusion-pipe) save block attention fused as
``layers.N.proj.qkv`` in q, k, v order; :func:`resolve_targets` splits it into
equal chunks on the three projections. Every other path binds verbatim, dotted
(``diffusion_model.``, ``transformer.``, bare) or kohya flat-underscore
(``lora_unet_``). The text embedder's ``blocks.N.qkv`` is fused in the model
as well and binds as is.
"""

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

QKV_SPLIT = ("q", "k", "v")


def resolve_targets(prefix_used, base):
    """Return ``[(module_path, ChunkSpec | None), ...]`` for a parsed group key.

    For the passthrough prefixes (``transformer.`` and the bare sentinel) an
    empty result defers to the verbatim fallback in
    :func:`native_adapter.resolve_group_targets`.
    """
    sep = "_" if prefix_used == "lora_unet_" else "."
    if base.endswith(f"{sep}proj{sep}qkv"):
        stem = base[:-len("qkv")]
        return [(stem + name, ChunkSpec(idx=idx, total=len(QKV_SPLIT))) for idx, name in enumerate(QKV_SPLIT)]
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
