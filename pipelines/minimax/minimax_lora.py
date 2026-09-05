"""MiniMax H3 native adapter loader.

MiniMax H3 is a modular video pipeline with a transformer and a text encoder.
This loader routes transformer keys to ``lora_transformer_`` and text-encoder
keys to ``lora_te_`` when native LoRAs are applied.

Published LoRAs target the reference module names. The mapping onto
``diffusers.MiniMaxH3Transformer3DModel`` follows the diffusers LoRA converter:
fused ``attn.qkv_proj`` splits into ``to_q``/``to_k``/``to_v``, and ``mlp.fc1``
lands on the fused SwiGLU projection with its two output halves swapped.
"""

import re

from modules.lora import native_adapter


KNOWN_PREFIXES = (
    "diffusion_model.text_encoders.",
    "diffusion_model.text_encoder.",
    "diffusion_model.transformer.",
    "diffusion_model.blocks.",
    "diffusion_model.transformer_blocks.",
    "diffusion_model.token_refiner.refiner_blocks.",
    "diffusion_model.token_refiner.",
    "diffusion_model.model.language_model.",
    "text_encoders.",
    "text_encoder.",
    "model.language_model.",
    "transformer.",
    "transformer_blocks.",
    "blocks.",
    "token_refiner.refiner_blocks.",
    "token_refiner.",
) + native_adapter.KNOWN_PREFIXES_DEFAULT

# Reference keys outside the block stacks carry no arch prefix in reference saves; the base is the whole module path.
BARE_PREFIXES = ("video_patch_proj.", "audio_patch_proj.", "condition_proj.", "time_embedder.", "final_layer.")


STANDALONE_RENAMES = {
    "video_patch_proj": "proj_in",
    "audio_patch_proj": "audio_proj_in",
    "condition_proj": "context_embedder",
    "time_embedder.proj_in": "time_embedder.linear_1",
    "time_embedder.proj_out": "time_embedder.linear_2",
    "final_layer.adaln_proj.linear": "norm_out.linear",
    "final_layer.video_out": "proj_out",
    "final_layer.audio_out": "audio_proj_out",
}

# Reference block leaves and where each lands in a diffusers block; a leaf not listed here binds verbatim.
BLOCK_LEAF_TARGETS = {
    "attn.qkv_proj": tuple((f"attn.{k}", native_adapter.ChunkSpec(idx=i, total=3)) for i, k in enumerate(("to_q", "to_k", "to_v"))),
    "attn.out_proj": (("attn.to_out.0", None),),
    "mlp.fc1": (("ff.net.0.proj", native_adapter.ChunkSpec(reorder=(1, 0))),), # reference [gate; value] onto the diffusers SwiGLU [value; gate]
    "mlp.fc2": (("ff.net.2", None),),
    "adaln_proj.linear": (("adaln_proj.linear", None),),
}

BLOCK_STACKS = {
    "blocks": "transformer_blocks",
    "token_refiner.blocks": "token_refiner.refiner_blocks",
}


# Re-export for tests / compatibility
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


MINIMAX_EXTRA_SUFFIXES = (".lora_down", ".lora_up", ".lora_A", ".lora_B")
MINIMAX_LORA_SUFFIXES = native_adapter.LORA_SUFFIXES + MINIMAX_EXTRA_SUFFIXES
MINIMAX_SUFFIX_NORMALIZE = {
    "lora_down": "lora_down.weight",
    "lora_up": "lora_up.weight",
    "lora_A": "lora_down.weight",
    "lora_B": "lora_up.weight",
}


def normalize_mini_max_suffix(suffix: str) -> str:
    return MINIMAX_SUFFIX_NORMALIZE.get(suffix, suffix)


def _flattened(dotted):
    return re.escape(dotted.replace(".", "_"))


# musubi-tuner flattens every "." to "_" under a lora_unet_ prefix; the reference module names carry underscores
# of their own, so the dotted path is recovered by matching the whole flattened name against the vocabulary above.
_FLATTENED_MODULES = [(re.compile(_flattened(name)), name) for name in STANDALONE_RENAMES] + [
    (re.compile(rf"{_flattened(stack)}_(\d+)_{_flattened(leaf)}"), f"{stack}.{{}}.{leaf}")
    for stack in BLOCK_STACKS for leaf in BLOCK_LEAF_TARGETS
]


def _unflatten_lora_unet_key(key: str) -> str | None:
    if not key.startswith("lora_unet_"):
        return None
    module_key, _, suffix = key[len("lora_unet_"):].partition(".")
    if not suffix:
        return None
    for pattern, dotted in _FLATTENED_MODULES:
        match = pattern.fullmatch(module_key)
        if match:
            return f"{dotted.format(*match.groups())}.{suffix}"
    return None


def parse_key(key, suffixes):
    """MiniMax-bound :func:`native_adapter.parse_key`."""
    unflattened = _unflatten_lora_unet_key(key)
    if unflattened is not None:
        key = unflattened
    key = native_adapter.unwrap_peft_wrapper(key)
    if key.startswith("dit."):
        key = "diffusion_model." + key[len("dit."):]
    parsed = native_adapter.parse_key(key, suffixes, prefixes=KNOWN_PREFIXES, bare_prefixes=BARE_PREFIXES)
    if parsed is None:
        return None
    prefix_used, base, suffix = parsed
    return prefix_used, base, normalize_mini_max_suffix(suffix)


def group_by_suffixes(state_dict, suffixes, *, prefixes=None, bare_prefixes=()): # pylint: disable=unused-argument
    """MiniMax-bound :func:`native_adapter.group_by_suffixes`."""
    groups: dict[tuple, dict[str, object]] = {}
    for key, value in state_dict.items():
        parsed = parse_key(key, suffixes)
        if parsed is None:
            continue
        prefix_used, base, suffix = parsed
        slot = groups.get((prefix_used, base))
        if slot is None:
            slot = {}
            groups[(prefix_used, base)] = slot
        slot[suffix] = value
    return groups


def _block_targets(target_stack, base):
    """Targets for ``<index>.<leaf>`` inside a reference block stack."""
    idx, _, leaf = base.partition(".")
    targets = BLOCK_LEAF_TARGETS.get(leaf)
    if targets is None:
        return [(f"{target_stack}.{base}", None)]
    return [(f"{target_stack}.{idx}.{path}", chunk) for path, chunk in targets]


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for MiniMax keys."""
    if prefix_used == "diffusion_model." or prefix_used is None:
        if base.startswith("transformer."):
            return [(base[len("transformer."):], None)]
        if base.startswith("text_encoder."):
            return [(base[len("text_encoder."):], None)]
        if base.startswith("text_encoders."):
            return [(base[len("text_encoders."):], None)]
        return [(STANDALONE_RENAMES.get(base, base), None)]
    if prefix_used in ("diffusion_model.transformer.", "transformer.", "lora_transformer_"):
        return [(base, None)]
    if prefix_used in ("diffusion_model.transformer_blocks.", "transformer_blocks."):
        return [(f"transformer_blocks.{base}", None)]
    if prefix_used in ("diffusion_model.blocks.", "blocks."):
        return _block_targets(BLOCK_STACKS["blocks"], base)
    if prefix_used in ("diffusion_model.token_refiner.refiner_blocks.", "token_refiner.refiner_blocks."):
        return [(f"token_refiner.refiner_blocks.{base}", None)]
    if prefix_used in ("diffusion_model.token_refiner.", "token_refiner."):
        if base.startswith("blocks."):
            return _block_targets(BLOCK_STACKS["token_refiner.blocks"], base[len("blocks."):])
        return [(f"token_refiner.refiner_{base}", None)]
    if prefix_used in (
            "diffusion_model.text_encoder.",
            "diffusion_model.text_encoders.",
            "text_encoder.",
            "text_encoders.",
            "lora_te_",
    ):
        return [(base, None)]
    if prefix_used in ("lora_unet_", "lycoris_"):
        return [(base, None)]
    return []


def network_prefix_for(prefix_used):
    """Choose the network prefix for a parsed MiniMax group."""
    if prefix_used == "lora_te_":
        return "lora_te_"
    if prefix_used in (
            "text_encoder.",
            "text_encoders.",
            "diffusion_model.text_encoder.",
            "diffusion_model.text_encoders.",
    ):
        return "lora_te_"
    return "lora_transformer_"


def file_alpha(network_on_disk):
    """The training alpha some trainers record in the safetensors metadata instead of per-key tensors, or None."""
    alpha = (getattr(network_on_disk, "metadata", None) or {}).get("alpha")
    if alpha is None:
        return None
    try:
        return float(alpha)
    except (TypeError, ValueError):
        return None


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_prefixes=BARE_PREFIXES,
    network_prefix=network_prefix_for,
    group_by_suffixes_fn=group_by_suffixes,
    arch_name="minimaxh3",
)


def try_load_lora(name, network_on_disk, lora_scale):
    return native_adapter.try_load_lora(name, network_on_disk, lora_scale, network_alpha=file_alpha(network_on_disk), **_BIND_KWARGS)


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
    """Run every MiniMax family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
