"""Chroma native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``chroma`` in ``allow_native``).

Entry points, one per family: :func:`try_load_lora` (plus DoRA),
:func:`try_load_lokr`, :func:`try_load_loha`, :func:`try_load_oft`.

Recognized key prefixes: ``diffusion_model.``, ``transformer.``,
``lora_unet_``, plus bare BFL paths (``double_blocks.`` / ``single_blocks.``)
and bare diffusers paths (``transformer_blocks.`` /
``single_transformer_blocks.`` / ``distilled_guidance_layer.``).

Chroma LoRAs are trained against the Flux block layout regardless of save
format. The diffusers ``ChromaTransformer2DModel`` exposes split-attention
modules; :func:`resolve_targets` rewrites Flux-layout keys to diffusers
paths and emits ``ChunkSpec`` entries for fused targets.

Fused weight handling:

- ``double_blocks.{i}.img_attn.qkv`` / ``txt_attn.qkv``: three equal Q/K/V
  chunks (``ChunkSpec(idx, total=3)``).
- ``single_blocks.{i}.linear1``: four chunks Q/K/V/proj_mlp at unequal dims
  ``[HIDDEN, HIDDEN, HIDDEN, MLP_HIDDEN]`` (``ChunkSpec(start, end)``).

Chroma's modulation generator is the central ``distilled_guidance_layer``
approximator (replacing Flux's per-block ``norm1.linear``). The pruned
``ChromaAdaLayerNormZeroPruned`` classes have no ``.linear`` submodule, so
any ``_mod_lin`` / ``_modulation_lin`` keys land in ``unmapped``. LoRAs
targeting the approximator pass through unchanged.
"""

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

BARE_FLUX_PREFIXES = ("double_blocks.", "single_blocks.")

BARE_DIFFUSERS_PREFIXES = (
    "transformer_blocks.", "single_transformer_blocks.",
    "distilled_guidance_layer.",
)


# === Fused weight dims ===
# Defaults match Chroma1-HD (``inner_dim = num_attention_heads *
# attention_head_dim = 24 * 128 = 3072``; ``mlp_hidden = 12288``). Tests
# patch these to the mock's scale via direct module-level assignment.

QKV_DIMS = [3072, 3072, 3072]
LINEAR1_DIMS = [3072, 3072, 3072, 12288]


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
    """Chroma-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_prefixes=BARE_FLUX_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """Chroma-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_prefixes=BARE_FLUX_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Rewrite Flux-layout keys to diffusers paths; emit ChunkSpec on fused targets.

    Dispatches on ``prefix_used``:

    - ``lora_unet_``: kohya underscore-flat Flux path; parse block type/index
      and module suffix, rename to diffusers.
    - ``diffusion_model.`` or bare BFL (None): dotted Flux path; same rewrite.
    - ``transformer.`` or bare-diffusers: already a diffusers path; passthrough.
    """
    if prefix_used == "transformer.":
        return [(base, None)]
    if prefix_used == BARE_DIFFUSERS_PREFIX_USED:
        return [(base, None)]
    if prefix_used == "lora_unet_":
        return _kohya_to_diffusers(base)
    if prefix_used in (None, "diffusion_model."):
        return _bfl_to_diffusers(base)
    return []


def _kohya_to_diffusers(base):
    """For kohya keys like ``double_blocks_0_img_attn_qkv`` or ``single_blocks_5_linear1``."""
    if base.startswith("double_blocks_"):
        rest = base[len("double_blocks_"):]
        idx, _, suffix = rest.partition("_")
        return _double_block_targets(idx, suffix)
    if base.startswith("single_blocks_"):
        rest = base[len("single_blocks_"):]
        idx, _, suffix = rest.partition("_")
        return _single_block_targets(idx, suffix)
    return [(base, None)]


def _bfl_to_diffusers(base):
    """For BFL dotted keys like ``double_blocks.0.img_attn.qkv``."""
    parts = base.split(".")
    if len(parts) < 3:
        return [(base, None)]
    block_type, block_idx, module_suffix = parts[0], parts[1], ".".join(parts[2:])
    # Normalize the dotted module suffix to the underscore form the dispatch tables use.
    suffix_key = module_suffix.replace(".", "_")
    if block_type == "double_blocks":
        return _double_block_targets(block_idx, suffix_key)
    if block_type == "single_blocks":
        return _single_block_targets(block_idx, suffix_key)
    return [(base, None)]


# Static non-fused renames (underscore-keyed for dispatch from either kohya or BFL paths).
_DOUBLE_STATIC = {
    "img_attn_proj": "attn.to_out.0",
    "txt_attn_proj": "attn.to_add_out",
    "img_mlp_0":     "ff.net.0.proj",
    "img_mlp_2":     "ff.net.2",
    "txt_mlp_0":     "ff_context.net.0.proj",
    "txt_mlp_2":     "ff_context.net.2",
}
_SINGLE_STATIC = {
    "linear2": "proj_out",
}


def _double_block_targets(block_idx, suffix_key):
    """Resolve a double_blocks.{i}.<suffix> target to diffusers paths."""
    if suffix_key == "img_attn_qkv":
        return _split_double_qkv(block_idx, ("to_q", "to_k", "to_v"))
    if suffix_key == "txt_attn_qkv":
        return _split_double_qkv(block_idx, ("add_q_proj", "add_k_proj", "add_v_proj"))
    if suffix_key in _DOUBLE_STATIC:
        return [(f"transformer_blocks.{block_idx}.{_DOUBLE_STATIC[suffix_key]}", None)]
    return []


def _single_block_targets(block_idx, suffix_key):
    """Resolve a single_blocks.{i}.<suffix> target to diffusers paths."""
    if suffix_key == "linear1":
        return _split_single_linear1(block_idx)
    if suffix_key in _SINGLE_STATIC:
        return [(f"single_transformer_blocks.{block_idx}.{_SINGLE_STATIC[suffix_key]}", None)]
    return []


def _split_double_qkv(block_idx, attn_keys):
    """Three equal-chunk ChunkSpec entries for fused img_attn.qkv / txt_attn.qkv."""
    return [
        (f"transformer_blocks.{block_idx}.attn.{k}", ChunkSpec(idx=i, total=len(attn_keys)))
        for i, k in enumerate(attn_keys)
    ]


def _split_single_linear1(block_idx):
    """Four unequal-range ChunkSpec entries for fused linear1 (Q/K/V/proj_mlp)."""
    target_keys = (
        f"single_transformer_blocks.{block_idx}.attn.to_q",
        f"single_transformer_blocks.{block_idx}.attn.to_k",
        f"single_transformer_blocks.{block_idx}.attn.to_v",
        f"single_transformer_blocks.{block_idx}.proj_mlp",
    )
    targets = []
    start = 0
    for d, target in zip(LINEAR1_DIMS, target_keys):
        targets.append((target, ChunkSpec(start=start, end=start + d)))
        start += d
    return targets


# === Native loaders (thin wrappers over native_adapter generics) ===


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_prefixes=BARE_FLUX_PREFIXES,
    bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    arch_name="chroma",
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
    """Run every Chroma family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(try_load_lora, try_load_lokr, try_load_loha, try_load_oft),
    )
