"""Anima native adapter loader.

Anima ships with three trainable surfaces:

- ``transformer`` (Cosmos 2.0-style DiT), stamped as ``lora_transformer_*``
- ``llm_adapter`` (custom MLP that projects Qwen3 hidden states into the DiT
  cross-attention dim), stamped as ``lora_llm_adapter_*``
- ``text_encoder`` (Qwen3Model), stamped as ``lora_te_*``

The loader recognizes four ``ss_network_module`` formats in the wild:

- kohya transformer: ``lora_unet_blocks_0_self_attn_q_proj.lora_down.weight``
- kohya text-encoder: ``lora_te_layers_0_self_attn_q_proj.lora_down.weight``
- BFL/AI-toolkit: ``diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight``
  with the adapter under ``diffusion_model.llm_adapter.*`` and the TE under
  ``text_encoders.qwen3_06b.transformer.model.*``
- LyCORIS Hadamard (LoHA): same kohya path structure but with
  ``hada_w1_a`` / ``hada_w1_b`` / ``hada_w2_a`` / ``hada_w2_b`` weights

All transformer paths route through :func:`cosmos_rename_flat` which mirrors
``diffusers.loaders.single_file_utils.convert_cosmos_transformer_checkpoint_to_diffusers``'s
``TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0``, transposed into flat (underscore)
form so str.replace produces a path that matches the diffusers module name
already stamped on the network_layer_mapping. Adapter and TE paths bypass the
rename and are flattened verbatim.

Network-key construction (transformer vs llm_adapter vs te) is parameterized
in :mod:`modules.lora.native_loader` via the ``network_prefix`` kwarg; this
module supplies :func:`network_prefix_for` to pick per ``prefix_used``.
Family-specific dispatch (LoRA, LoHA, LoKR, OFT, IA3, GLoRA, Norm, Full) is
inherited from native_loader's generics; alpha / scale / DoRA flow through
the standard ``NetworkWeights.w`` slots rather than being baked into the
factor weights at load time.
"""

from collections import OrderedDict

from modules.lora import native_loader


# === Arch-specific prefix configuration ===
#
# Order matters: longer / more-specific prefixes must precede shorter ones,
# because :func:`native_loader.parse_key` returns the first match. Both
# ``diffusion_model.llm_adapter.`` and ``text_encoders.qwen3_06b.transformer.model.``
# start with ``diffusion_model.`` / ``text_encoders.`` so they must be listed first.

ANIMA_PREFIXES = (
    "diffusion_model.llm_adapter.",
    "text_encoders.qwen3_06b.transformer.model.",
    "diffusion_model.",
    "lora_te_",
    "lora_unet_",
)


# === Re-exports for test / back-compat ===
# Tests address these through the anima_lora module surface; sibling pipelines
# do the same (see flux2_lora / zimage_lora / ernie_lora).

LORA_SUFFIXES = native_loader.LORA_SUFFIXES
LOKR_SUFFIXES = native_loader.LOKR_SUFFIXES
LOHA_SUFFIXES = native_loader.LOHA_SUFFIXES
OFT_SUFFIXES = native_loader.OFT_SUFFIXES
IA3_SUFFIXES = native_loader.IA3_SUFFIXES
GLORA_SUFFIXES = native_loader.GLORA_SUFFIXES
NORM_SUFFIXES = native_loader.NORM_SUFFIXES
FULL_SUFFIXES = native_loader.FULL_SUFFIXES

LORA_MARKERS = native_loader.LORA_MARKERS
LOKR_MARKERS = native_loader.LOKR_MARKERS
LOHA_MARKERS = native_loader.LOHA_MARKERS
OFT_MARKERS = native_loader.OFT_MARKERS
IA3_MARKERS = native_loader.IA3_MARKERS
GLORA_MARKERS = native_loader.GLORA_MARKERS
NORM_MARKERS = native_loader.NORM_MARKERS
FULL_MARKERS = native_loader.FULL_MARKERS

SUFFIX_NORMALIZE = native_loader.SUFFIX_NORMALIZE
BARE_DIFFUSERS_PREFIX_USED = native_loader.BARE_DIFFUSERS_PREFIX_USED
has_marker = native_loader.has_marker


def parse_key(key, suffixes):
    """Anima-bound :func:`native_loader.parse_key`."""
    return native_loader.parse_key(key, suffixes, prefixes=ANIMA_PREFIXES)


def group_by_suffixes(state_dict, suffixes):
    """Anima-bound :func:`native_loader.group_by_suffixes`."""
    return native_loader.group_by_suffixes(state_dict, suffixes, prefixes=ANIMA_PREFIXES)


# === Cosmos 2.0 path rename (transformer only) ===
#
# Applied to underscore-flattened paths. Order mirrors diffusers
# ``TRANSFORMER_KEYS_RENAME_DICT_COSMOS_2_0`` exactly: longer substrings must
# precede substrings nested inside them so ``str.replace`` does not consume a
# fragment that a later rule still needs to match. Numeric-suffixed entries
# (e.g. ``t_embedder_1``) come first because the bare ``t_embedder`` would
# otherwise eat the ``_1`` suffix.

COSMOS_2_FLAT_RENAME = OrderedDict([
    ("t_embedder_1", "time_embed_t_embedder"),
    ("t_embedding_norm", "time_embed_norm"),
    ("blocks", "transformer_blocks"),
    ("adaln_modulation_self_attn_1", "norm1_linear_1"),
    ("adaln_modulation_self_attn_2", "norm1_linear_2"),
    ("adaln_modulation_cross_attn_1", "norm2_linear_1"),
    ("adaln_modulation_cross_attn_2", "norm2_linear_2"),
    ("adaln_modulation_mlp_1", "norm3_linear_1"),
    ("adaln_modulation_mlp_2", "norm3_linear_2"),
    ("self_attn", "attn1"),
    ("cross_attn", "attn2"),
    ("q_proj", "to_q"),
    ("k_proj", "to_k"),
    ("v_proj", "to_v"),
    ("output_proj", "to_out_0"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("mlp_layer1", "ff_net_0_proj"),
    ("mlp_layer2", "ff_net_2"),
    ("x_embedder_proj_1", "patch_embed_proj"),
    ("final_layer_adaln_modulation_1", "norm_out_linear_1"),
    ("final_layer_adaln_modulation_2", "norm_out_linear_2"),
    ("final_layer_linear", "proj_out"),
])


def cosmos_rename_flat(flat):
    """Apply the Cosmos 2.0 path rename to a flattened (underscore) path."""
    for k, v in COSMOS_2_FLAT_RENAME.items():
        flat = flat.replace(k, v)
    return flat


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group.

    Anima has no fused QKV (every kohya / BFL key targets a single diffusers
    module), so ``ChunkSpec`` is always ``None`` and each call returns at most
    one target.

    The returned ``diffusers_path`` is already underscore-flattened; the
    generic loader's ``path.replace('.', '_')`` is a no-op on it. Routing into
    the right namespace (transformer / llm_adapter / te) is handled separately
    by :func:`network_prefix_for`.
    """
    if prefix_used == "lora_unet_":
        return [(cosmos_rename_flat(base), None)]
    if prefix_used == "diffusion_model.":
        return [(cosmos_rename_flat(base.replace(".", "_")), None)]
    if prefix_used == "diffusion_model.llm_adapter.":
        return [(base.replace(".", "_"), None)]
    if prefix_used in ("text_encoders.qwen3_06b.transformer.model.", "lora_te_"):
        return [(base.replace(".", "_"), None)]
    return []


def network_prefix_for(prefix_used):
    """Pick the ``network_layer_mapping`` namespace prefix.

    ``lora_convert.assign_network_names_to_compvis_modules`` stamps three
    namespaces on Anima modules; this picks the right one based on which arch
    prefix the key matched.
    """
    if prefix_used == "diffusion_model.llm_adapter.":
        return "lora_llm_adapter_"
    if prefix_used in ("text_encoders.qwen3_06b.transformer.model.", "lora_te_"):
        return "lora_te_"
    return "lora_transformer_"


# === Native loaders (thin wrappers over native_loader generics) ===

_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=ANIMA_PREFIXES,
    network_prefix=network_prefix_for,
    arch_name="anima",
)


def try_load_lora(name, network_on_disk, lora_scale):
    return native_loader.try_load_lora(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_lokr(name, network_on_disk, lora_scale):
    return native_loader.try_load_lokr(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_loha(name, network_on_disk, lora_scale):
    return native_loader.try_load_loha(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_oft(name, network_on_disk, lora_scale):
    return native_loader.try_load_oft(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_ia3(name, network_on_disk, lora_scale):
    return native_loader.try_load_ia3(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_glora(name, network_on_disk, lora_scale):
    return native_loader.try_load_glora(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_norm(name, network_on_disk, lora_scale):
    return native_loader.try_load_norm(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load_full(name, network_on_disk, lora_scale):
    return native_loader.try_load_full(name, network_on_disk, lora_scale, **_BIND_KWARGS)


def try_load(name, network_on_disk, lora_scale):
    """Run every Anima family loader, merge any that match."""
    return native_loader.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
