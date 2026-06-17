"""Ideogram 4 native adapter loader. Bound when ``ideogram4`` is in
``lora_overrides.allow_native`` and ``get_method`` returns ``'native'``.

Single-stream DiT. diffusers ``Ideogram4Transformer2DModel`` exposes ``layers.{i}``
with split attention (``attention.{to_q,to_k,to_v,to_out.0}``), SwiGLU
``feed_forward.{w1,w2,w3}`` and a per-block ``adaln_modulation`` Linear. Released
LoRAs save attention fused (``attention.qkv`` equal-width + ``attention.o``);
:func:`resolve_targets` thirds the qkv and renames ``o`` to ``to_out.0``, the rest
passes through. Prefixes: ``diffusion_model.``, ``transformer.``, ``lora_unet_``,
``lora_transformer_``, bare ``layers.``. Families: LoRA (+DoRA), LoKR, LoHA, OFT.

A bare reference fans out across both towers by default (conditional at full
strength, ``unconditional_transformer`` at ``model_ideogram4_uncond_lora_scale``;
see :func:`default_tower_expansion`). ``cond`` restricts to conditional only;
``uncond`` / ``both`` / ``module=`` select explicitly. Selection, stamping and
fan-out live in the shared apply path and parse layer, not the loaders.
"""

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

# Bare paths route through resolve_targets (prefix_used=None) rather than the
# generic passthrough, so a bare fused ``layers.0.attention.qkv`` still splits.
BARE_BFL_PREFIXES = ("layers.",)


# === Re-exports for test/back-compat ===

LORA_SUFFIXES = native_adapter.LORA_SUFFIXES
LOKR_SUFFIXES = native_adapter.LOKR_SUFFIXES
LOHA_SUFFIXES = native_adapter.LOHA_SUFFIXES
OFT_SUFFIXES = native_adapter.OFT_SUFFIXES

LORA_MARKERS = native_adapter.LORA_MARKERS
LOKR_MARKERS = native_adapter.LOKR_MARKERS
LOHA_MARKERS = native_adapter.LOHA_MARKERS
OFT_MARKERS = native_adapter.OFT_MARKERS

SUFFIX_NORMALIZE = native_adapter.SUFFIX_NORMALIZE
BARE_DIFFUSERS_PREFIX_USED = native_adapter.BARE_DIFFUSERS_PREFIX_USED
has_marker = native_adapter.has_marker


def parse_key(key, suffixes):
    """Ideogram-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_prefixes=BARE_BFL_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """Ideogram-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_prefixes=BARE_BFL_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group key.

    Rewrites the fused attention layout to diffusers split paths: ``attention.qkv``
    -> three equal Q/K/V chunks (``ChunkSpec(idx, total=3)``); ``attention.o`` ->
    ``attention.to_out.0``. Everything else (``feed_forward``, ``adaln_modulation``,
    embedders) is already diffusers-form, returned verbatim. Universal passthrough
    prefixes are handled upstream by :func:`native_adapter.resolve_group_targets`.
    """
    if prefix_used == "lora_unet_":
        return _underscore_to_diffusers(base)
    if prefix_used in (None, "diffusion_model."):
        return _dotted_to_diffusers(base)
    return []


def _dotted_to_diffusers(base):
    """For BFL dotted keys like ``layers.0.attention.qkv`` / ``layers.0.attention.o``."""
    if base.endswith(".attention.qkv"):
        stem = base[:-len(".attention.qkv")]
        return _split_qkv(stem, ".")
    if base.endswith(".attention.o"):
        stem = base[:-len(".attention.o")]
        return [(f"{stem}.attention.to_out.0", None)]
    return [(base, None)]


def _underscore_to_diffusers(base):
    """For kohya flat-underscore keys like ``layers_0_attention_qkv`` / ``layers_0_attention_o``."""
    if base.endswith("_attention_qkv"):
        stem = base[:-len("_attention_qkv")]
        return _split_qkv(stem, "_")
    if base.endswith("_attention_o"):
        stem = base[:-len("_attention_o")]
        return [(f"{stem}_attention_to_out_0", None)]
    return [(base, None)]


def _split_qkv(stem, sep):
    """Three equal-chunk ChunkSpec entries for the fused ``attention.qkv``.

    Q, K and V share ``out_features`` (``num_attention_heads * attention_head_dim``),
    so the fused up-weight slices into thirds along dim 0.
    """
    return [
        (f"{stem}{sep}attention{sep}to_q", ChunkSpec(idx=0, total=3)),
        (f"{stem}{sep}attention{sep}to_k", ChunkSpec(idx=1, total=3)),
        (f"{stem}{sep}attention{sep}to_v", ChunkSpec(idx=2, total=3)),
    ]


# === Native loaders (thin wrappers over native_adapter generics) ===


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_prefixes=BARE_BFL_PREFIXES,
    arch_name="ideogram4",
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
    """Run every Ideogram family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(try_load_lora, try_load_lokr, try_load_loha, try_load_oft),
    )


# === Default tower fan-out (asymmetric CFG) ===


def default_tower_expansion(sd_model):
    """Extra ``(component, scale)`` applications a bare reference adds beyond the
    conditional default.

    Ideogram runs two asymmetric-CFG transformers; conditional-only is washed out,
    equal strength on both over-cooks. Returns the unconditional tower at
    ``model_ideogram4_uncond_lora_scale`` when it is a distinct live module, else
    ``[]`` (guidance off -> uncond reuses the conditional transformer, nothing to add).
    """
    from modules import shared
    cond = getattr(sd_model, 'transformer', None)
    uncond = getattr(sd_model, 'unconditional_transformer', None)
    if uncond is None or uncond is cond:
        return []
    scale = getattr(shared.opts, 'model_ideogram4_uncond_lora_scale', 0.7)
    return [('unconditional_transformer', scale)]
