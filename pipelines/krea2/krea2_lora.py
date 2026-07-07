"""Krea 2 native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``krea2`` in ``allow_native``).

The transformer module tree mirrors the checkpoint (``blocks.N.attn.{wq,wk,wv,wo,gate}``,
``blocks.N.mlp.{gate,up,down}``, ``txtfusion.*``, ``first``, ``last`` ...), so dotted keys
bind verbatim with no name rewrite and no fused-QKV split. Kohya flat-underscore keys are
reconstructed back to dotted paths, protecting the two compound module names
(``layerwise_blocks``, ``refiner_blocks``).
"""

from modules.lora import native_adapter


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

# Top-level module names that a bare diffusers-format LoRA key can start with.
BARE_DIFFUSERS_PREFIXES = ("blocks.", "txtfusion.", "first.", "last.", "tmlp.", "tproj.", "txtmlp.")


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, None), ...]`` for a parsed group key.

    K2's diffusers module names equal the checkpoint names, so dotted keys map verbatim.
    Universal passthrough prefixes are handled upstream by
    :func:`native_adapter.resolve_group_targets`.
    """
    if prefix_used in (None, "diffusion_model.", "transformer."):
        return [(base, None)]
    if prefix_used in ("lora_unet_", "lora_transformer_"):
        return _underscore_to_dotted(base)
    return []


def _underscore_to_dotted(base):
    """Rebuild a dotted path from a kohya flat-underscore base, keeping compound names intact."""
    protected = base.replace("layerwise_blocks", "layerwise@blocks").replace("refiner_blocks", "refiner@blocks")
    return [(protected.replace("_", ".").replace("@", "_"), None)]


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    arch_name="krea2",
)


def try_load_lora(name, network_on_disk, lora_scale):
    return native_adapter.try_load_lora(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_lokr(name, network_on_disk, lora_scale):
    return native_adapter.try_load_lokr(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_loha(name, network_on_disk, lora_scale):
    return native_adapter.try_load_loha(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_oft(name, network_on_disk, lora_scale):
    return native_adapter.try_load_oft(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load(name, network_on_disk, lora_scale):
    """Run every Krea 2 family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(try_load_lora, try_load_lokr, try_load_loha, try_load_oft),
    )
