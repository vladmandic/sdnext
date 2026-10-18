"""Flux 3 native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``f3`` in ``allow_native``). Reads the safetensors directly and writes into
sdnext's ``network_layer_mapping``, returning a ``Network`` populated with
``NetworkModule*`` entries that ``network_activate`` applies.

The transformer module tree is the diffusers one: ``mode_blocks.image.N``,
``mode_blocks.image_cond.N``, ``txt_mode_blocks.N`` and
``single_transformer_blocks.N``, each with ``attn.to_q/to_k/to_v/to_out.0``
and ``ff.linear_in/linear_out``. Diffusers-PEFT keys (``transformer.`` prefix
or bare, as the diffusers DreamBooth trainer saves them) bind verbatim: every
projection is its own Linear, so there is no fused-QKV chunking. BFL keys
(``content_mode_blocks.``, ``single_blocks.``, ``q_proj``, ``attn_out``,
``mlp_in``...) are rewritten with the tables of the diffusers checkpoint
converter; the output adaLN swaps its scale/shift halves on the way. Kohya
flat-underscore keys are rebuilt into dotted paths by protecting the module
names that carry underscores of their own, in either spelling.
"""

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

# BFL block leaves -> diffusers block leaves, shared by the per-mode and the joint blocks
BFL_LEAF_MAP = {
    "q_proj": "attn.to_q",
    "k_proj": "attn.to_k",
    "v_proj": "attn.to_v",
    "attn_out": "attn.to_out.0",
    "mlp_in": "ff.linear_in",
    "mlp_out": "ff.linear_out",
}

# BFL block stacks -> diffusers stacks
BFL_STACK_MAP = {
    "content_mode_blocks": "mode_blocks",
    "txt_mode_blocks": "txt_mode_blocks",
    "single_blocks": "single_transformer_blocks",
}

# BFL non-block modules, exact match on the full base
BFL_EXTRA_MAP = {
    "emb_in.image": "x_embedders.image",
    "emb_in.image_cond": "x_embedders.image_cond",
    "txt_in": "context_embedder",
    "time_in.in_layer": "timestep_embedder.linear_1",
    "time_in.out_layer": "timestep_embedder.linear_2",
    "ref_time_in.in_layer": "target_timestep_embedder.linear_1",
    "ref_time_in.out_layer": "target_timestep_embedder.linear_2",
    "final_layer.image.linear": "proj_out",
}

# the original adaLN produces [shift, scale]; Flux3AdaLayerNorm expects [scale, shift]
BFL_ADALN = "final_layer.image.adaLN_modulation.1"
ADALN_TARGET = ("norm_out.linear", ChunkSpec(reorder=(1, 0)))

MODULATION_STACKS = ("early_stream_modulations", "single_stream_modulations")

# module names with underscores of their own, longest first so a name never splits inside a longer one
COMPOUND_NAMES = sorted((
    "content_mode_blocks", "txt_mode_blocks", "single_blocks", "mode_blocks", "single_transformer_blocks",
    "early_stream_modulations", "single_stream_modulations",
    "image_cond", "q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out",
    "to_q", "to_k", "to_v", "to_out", "linear_in", "linear_out", "linear_1", "linear_2",
    "emb_in", "txt_in", "ref_time_in", "time_in", "in_layer", "out_layer", "final_layer", "adaLN_modulation",
    "x_embedders", "context_embedder", "target_timestep_embedder", "timestep_embedder", "proj_out", "norm_out",
), key=len, reverse=True)


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group key.

    BFL paths are rewritten; diffusers paths pass. For the passthrough prefixes
    (``transformer.`` and the bare sentinel) an empty result defers to the
    verbatim fallback in :func:`native_adapter.resolve_group_targets`.
    """
    if prefix_used == "lora_unet_":
        base = underscore_to_dotted(base)
    if prefix_used in ("diffusion_model.", "lora_unet_"):
        return bfl_to_diffusers(base) or [(base, None)]
    if prefix_used in ("transformer.", native_adapter.BARE_DIFFUSERS_PREFIX_USED):
        return bfl_to_diffusers(base)
    return []


def bfl_to_diffusers(base):
    """Rewrite a BFL module path to the diffusers path; ``[]`` when the path is not BFL-shaped."""
    extra = BFL_EXTRA_MAP.get(base)
    if extra is not None:
        return [(extra, None)]
    if base == BFL_ADALN:
        return [ADALN_TARGET]
    parts = base.split(".")
    if parts[0] in MODULATION_STACKS and parts[-1] == "lin":
        return [(".".join([*parts[:-1], "linear"]), None)]
    stack = BFL_STACK_MAP.get(parts[0])
    leaf = BFL_LEAF_MAP.get(parts[-1])
    if stack is None or leaf is None or len(parts) < 3:
        return []
    return [(".".join([stack, *parts[1:-1], leaf]), None)]


def underscore_to_dotted(base):
    """Rebuild a dotted path from a kohya flat-underscore base, keeping compound names intact."""
    protected = base
    for name in COMPOUND_NAMES:
        protected = protected.replace(name, name.replace("_", "@"))
    return protected.replace("_", ".").replace("@", "_")


BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    arch_name="f3",
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
    """Run every Flux 3 family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
