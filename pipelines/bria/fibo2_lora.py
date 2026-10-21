"""Bria FIBO 2.0 native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``fibo2`` in ``allow_native``).

Entry points, one per family: :func:`try_load_lora` (plus DoRA),
:func:`try_load_lokr`, :func:`try_load_loha`, :func:`try_load_oft`,
:func:`try_load_ia3`, :func:`try_load_glora`, :func:`try_load_norm`,
:func:`try_load_full`.

Recognized key prefixes: ``diffusion_model.``, ``transformer.`` (what the
release trainer in ``fine_tuning/train_lora.py`` saves), ``lora_unet_``, or
bare diffusers paths.

The diffusers module tree passes through verbatim:
``layers.N.attention.{to_q,to_k,to_v,to_out.0}``,
``layers.N.feed_forward.{w1,w2,w3}``, ``cross_attention`` on the injection
blocks, ``noise_refiner.N``, ``perceiver.layers.N.{cross_attention,
self_attention,ffn}``, ``x_embedder``, ``t_embedder.mlp.N`` and
``final_layer.*``. :func:`resolve_targets` rewrites the two other layouts an
adapter can be trained on:

- the original checkpoint names that diffusers' convert script renames
  (``cross_q`` / ``self_out`` / ``all_x_embedder.2-1`` and so on);
- the ComfyUI single-file names: fused ``attention.qkv`` is chunked into
  Q/K/V, fused ``to_kv`` into K/V, ``attention.out`` and ``to_out`` without
  the ModuleList index become ``to_out.0``, ``q_norm`` / ``k_norm`` become
  ``norm_q`` / ``norm_k``, and the fused ``w13`` projection, each half
  zero-padded to a multiple of 256 rows, is sliced to ``w1`` and ``w3``.
"""

from modules.logger import log
from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

# briaai/Fibo-2.0-dev transformer width, used when no model is loaded
DIM_DEFAULT = 3584

# Module names of the original checkpoint (left of diffusers' TRANSFORMER_KEYS_RENAME_DICT) and of the
# ComfyUI single file, each against the diffusers module path it stands for. Matched as path suffixes.
FIBO2_ALIASES = {
    ".cross_q": ".cross_attention.to_q",
    ".cross_k": ".cross_attention.to_k",
    ".cross_v": ".cross_attention.to_v",
    ".cross_out": ".cross_attention.to_out.0",
    ".cross_norm_q": ".cross_attention.norm_q",
    ".cross_norm_k": ".cross_attention.norm_k",
    ".self_q": ".self_attention.to_q",
    ".self_k": ".self_attention.to_k",
    ".self_v": ".self_attention.to_v",
    ".self_out": ".self_attention.to_out.0",
    ".self_norm_q": ".self_attention.norm_q",
    ".self_norm_k": ".self_attention.norm_k",
    ".attention.q_norm": ".attention.norm_q",
    ".attention.k_norm": ".attention.norm_k",
    ".attention.out": ".attention.to_out.0",
    ".attention.wo": ".attention.to_out.0",
    ".attention.to_out": ".attention.to_out.0",
    ".cross_attention.to_out": ".cross_attention.to_out.0",
    ".self_attention.to_out": ".self_attention.to_out.0",
}

# Non-block targets of the original checkpoint, keyed by patch size as in Z-Image
FIBO2_EXTRA_MAP = {
    "all_x_embedder.2-1": "x_embedder",
    "all_final_layer.2-1.linear": "final_layer.linear",
    "all_final_layer.2-1.adaLN_modulation.1": "final_layer.adaLN_modulation.1",
}

FUSED_QKV = ".attention.qkv"
FUSED_KV = (".cross_attention.to_kv", ".self_attention.to_kv")
FUSED_W13 = (".feed_forward.w13", ".ffn.w13")

# Prefixes whose bases are flat-underscore module paths
UNDERSCORE_PREFIXES = ("lora_unet_", "lora_transformer_", "lycoris_")


def ffn_rows():
    """``(hidden, padded)`` rows of one half of a fused ComfyUI ``w13``: the FFN width and its 256-row padding."""
    try:
        from modules import shared
        pipe = getattr(shared.sd_model, "pipe", shared.sd_model)
        dim = int(pipe.transformer.config.dim)
    except AttributeError:
        log.debug(f'LoRA: arch=fibo2 transformer width unavailable, using dim={DIM_DEFAULT}')
        dim = DIM_DEFAULT
    hidden = (dim * 8 // 3) // 64 * 64
    return hidden, -(-hidden // 256) * 256


# === Re-exports for test/back-compat ===

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
    """FIBO 2-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(key, suffixes, prefixes=KNOWN_PREFIXES)


def group_by_suffixes(state_dict, suffixes):
    """FIBO 2-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(state_dict, suffixes, prefixes=KNOWN_PREFIXES)


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group key."""
    sep = "_" if prefix_used in UNDERSCORE_PREFIXES else "."
    return rewrite(base, sep)


def rewrite(base, sep):
    """Targets for one base path; ``sep`` is ``.`` for dotted keys and ``_`` for flat-underscore keys."""
    def s(path):
        return path.replace(".", sep)

    for alias, target in FIBO2_EXTRA_MAP.items():
        if base == s(alias):
            return [(s(target), None)]
    if base.endswith(s(FUSED_QKV)):
        stem = base[:-len(s(FUSED_QKV))]
        return [(stem + s(f".attention.to_{name}"), ChunkSpec(idx=idx, total=3)) for idx, name in enumerate("qkv")]
    for fused in FUSED_KV:
        if base.endswith(s(fused)):
            stem = base[:-len(s(".to_kv"))]
            return [(stem + s(".to_k"), ChunkSpec(idx=0, total=2)), (stem + s(".to_v"), ChunkSpec(idx=1, total=2))]
    for fused in FUSED_W13:
        if base.endswith(s(fused)):
            stem = base[:-len(s(".w13"))]
            hidden, padded = ffn_rows() # the padded rows reach the output only through w2's padded columns, zero in the base model, so dropping them is exact
            return [(stem + s(".w1"), ChunkSpec(start=0, end=hidden)), (stem + s(".w3"), ChunkSpec(start=padded, end=padded + hidden))]
    for alias, target in FIBO2_ALIASES.items():
        if base.endswith(s(alias)):
            return [(base[:-len(s(alias))] + s(target), None)]
    return [(base, None)]


# === Native loaders (thin wrappers over native_adapter generics) ===


BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    arch_name="fibo2",
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
    """Run every FIBO 2 family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
