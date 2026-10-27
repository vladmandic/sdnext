"""Boogu-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``boogu`` in ``allow_native``).

The transformer keeps the official checkpoint names, which ComfyUI and its
trainers (ai-toolkit) use as well: ``double_stream_layers.N.*`` and
``single_stream_layers.N.*``, with the double-stream instruction attention's
projections on its processor module (``img_instruct_attn.processor.img_to_q``,
``...instruct_out``). No layer is fused, so every prefix binds verbatim,
dotted (``diffusion_model.``, ``transformer.``, bare) or kohya flat-underscore
(``lora_unet_``).
"""

from modules.lora import native_adapter


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT


def resolve_targets(prefix_used, base):
    """Return ``[(module_path, None)]``: every parsed group binds as named."""
    return [(base, None)]


BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    arch_name="boogu",
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
    """Run every Boogu-Image family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
