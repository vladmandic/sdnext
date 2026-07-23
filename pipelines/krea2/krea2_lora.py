"""Krea 2 native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``krea2`` in ``allow_native``).

The transformer module tree mirrors the checkpoint (``blocks.N.attn.{wq,wk,wv,wo,gate}``,
``blocks.N.mlp.{gate,up,down}``, ``txtfusion.*``, ``first``, ``last`` ...), so
checkpoint-style dotted keys bind verbatim with no fused-QKV split. Upstream diffusers
(``Krea2Transformer2DModel``) names the same modules differently
(``transformer_blocks.N.attn.to_q``, ``text_fusion.*``, ``img_in``, ``txt_in``,
``time_embed``, ``time_mod_proj``, ``final_layer``); LoRAs trained against it (e.g. the
official ``krea/Krea-2-LoRA-*`` releases) are rewritten to checkpoint names via the
``DIFFUSERS_*_MAP`` tables. Kohya flat-underscore keys are reconstructed back to dotted
checkpoint paths, protecting the two compound module names (``layerwise_blocks``,
``refiner_blocks``).
"""

from modules.lora import native_adapter


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

# Top-level module names that a bare LoRA key can start with: the transformer's own
# checkpoint-style names plus the upstream-diffusers names (as saved by
# ``Krea2Transformer2DModel.save_lora_adapter()``).
BARE_DIFFUSERS_PREFIXES = (
    "blocks.", "txtfusion.", "first.", "last.", "tmlp.", "tproj.", "txtmlp.",
    "transformer_blocks.", "text_fusion.", "img_in.", "txt_in.", "time_embed.", "time_mod_proj.", "final_layer.",
)

# Upstream-diffusers attention/ff leaves -> checkpoint leaves (block-level modules).
DIFFUSERS_LEAF_MAP = {
    ".attn.to_q": ".attn.wq",
    ".attn.to_k": ".attn.wk",
    ".attn.to_v": ".attn.wv",
    ".attn.to_gate": ".attn.gate",
    ".attn.to_out.0": ".attn.wo",
}

# Upstream-diffusers non-block modules -> checkpoint paths (exact match on the full base).
# Sequential containers on the checkpoint side are addressed by index
# (``tmlp``/``txtmlp``/``tproj`` interleave activations and norms with the Linears).
DIFFUSERS_EXTRA_MAP = {
    "img_in": "first",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "final_layer.linear": "last.linear",
}


# === Re-exports for test/back-compat ===
# The offline test suite addresses the family suffix/marker tables and the
# parse helpers through this module surface rather than importing native_adapter.

LORA_SUFFIXES = native_adapter.LORA_SUFFIXES
LOKR_SUFFIXES = native_adapter.LOKR_SUFFIXES
LOHA_SUFFIXES = native_adapter.LOHA_SUFFIXES
OFT_SUFFIXES = native_adapter.OFT_SUFFIXES
IA3_SUFFIXES = native_adapter.IA3_SUFFIXES
GLORA_SUFFIXES = native_adapter.GLORA_SUFFIXES
NORM_SUFFIXES = native_adapter.NORM_SUFFIXES
FULL_SUFFIXES = native_adapter.FULL_SUFFIXES

LORA_MARKERS = native_adapter.LORA_MARKERS
LOKR_MARKERS = native_adapter.LOKR_MARKERS
LOHA_MARKERS = native_adapter.LOHA_MARKERS
OFT_MARKERS = native_adapter.OFT_MARKERS
IA3_MARKERS = native_adapter.IA3_MARKERS
GLORA_MARKERS = native_adapter.GLORA_MARKERS
NORM_MARKERS = native_adapter.NORM_MARKERS
FULL_MARKERS = native_adapter.FULL_MARKERS

SUFFIX_NORMALIZE = native_adapter.SUFFIX_NORMALIZE
BARE_DIFFUSERS_PREFIX_USED = native_adapter.BARE_DIFFUSERS_PREFIX_USED
has_marker = native_adapter.has_marker


def parse_key(key, suffixes):
    """Krea2-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """Krea2-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def resolve_targets(prefix_used, base):
    """Return ``[(checkpoint_path, None), ...]`` for a parsed group key.

    Upstream-diffusers paths are rewritten to checkpoint names; checkpoint-style
    paths bind verbatim. For the passthrough prefixes (``transformer.`` and the
    bare-diffusers sentinel) returning ``[]`` defers to the verbatim fallback in
    :func:`native_adapter.resolve_group_targets`.
    """
    if prefix_used in (None, "diffusion_model."):
        return _diffusers_to_checkpoint(base) or [(base, None)]
    if prefix_used in ("transformer.", native_adapter.BARE_DIFFUSERS_PREFIX_USED):
        return _diffusers_to_checkpoint(base)
    if prefix_used == "lora_unet_":
        return _underscore_to_dotted(base)
    return []


def _diffusers_to_checkpoint(base):
    """Rewrite an upstream-diffusers module path to the checkpoint-style path.

    Returns ``[]`` for paths not in upstream-diffusers form so callers (and the
    shared verbatim fallback) handle checkpoint-style keys.
    """
    extra = DIFFUSERS_EXTRA_MAP.get(base)
    if extra is not None:
        return [(extra, None)]
    if base.startswith("transformer_blocks."):
        path = "blocks." + base[len("transformer_blocks."):]
    elif base.startswith("text_fusion."):
        path = "txtfusion." + base[len("text_fusion."):]
    else:
        return []
    for leaf, renamed in DIFFUSERS_LEAF_MAP.items():
        if path.endswith(leaf):
            path = path[:-len(leaf)] + renamed
            break
    path = path.replace(".ff.", ".mlp.")
    return [(path, None)]


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


def try_load_ia3(name, network_on_disk, lora_scale):
    return native_adapter.try_load_ia3(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_glora(name, network_on_disk, lora_scale):
    return native_adapter.try_load_glora(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_norm(name, network_on_disk, lora_scale):
    return native_adapter.try_load_norm(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_full(name, network_on_disk, lora_scale):
    return native_adapter.try_load_full(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load(name, network_on_disk, lora_scale):
    """Run every Krea 2 family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
