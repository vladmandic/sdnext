"""MiniMax H3 native adapter loader.

MiniMax H3 is a modular video pipeline with a transformer and a text encoder.
This loader routes transformer keys to ``lora_transformer_`` and text-encoder
keys to ``lora_te_`` when native LoRAs are applied.
"""

import re

from modules.lora import native_adapter


KNOWN_PREFIXES = (
    "diffusion_model.text_encoders.",
    "diffusion_model.text_encoder.",
    "diffusion_model.transformer.",
    "diffusion_model.blocks.",
    "diffusion_model.transformer_blocks.",
    "diffusion_model.token_refiner.",
    "diffusion_model.token_refiner.refiner_blocks.",
    "diffusion_model.final_layer.",
    "diffusion_model.video_patch_proj.",
    "diffusion_model.audio_patch_proj.",
    "diffusion_model.condition_proj.",
    "diffusion_model.time_embedder.proj_in.",
    "diffusion_model.time_embedder.proj_out.",
    "diffusion_model.model.language_model.",
    "text_encoders.",
    "text_encoder.",
    "model.language_model.",
    "transformer.",
    "transformer_blocks.",
    "blocks.",
    "token_refiner.",
    "token_refiner.refiner_blocks.",
    "final_layer.",
    "video_patch_proj.",
    "audio_patch_proj.",
    "condition_proj.",
    "time_embedder.proj_in.",
    "time_embedder.proj_out.",
) + native_adapter.KNOWN_PREFIXES_DEFAULT


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


def _unflatten_lora_unet_key(key: str) -> str | None:
    if not key.startswith("lora_unet_"):
        return None
    module_key, _, suffix = key[len("lora_unet_"):].partition(".")
    if not suffix:
        return None

    patterns = [
        (r"blocks_(\d+)_attn_out_proj", r"blocks.\1.attn.out_proj"),
        (r"blocks_(\d+)_attn_qkv_proj", r"blocks.\1.attn.qkv_proj"),
        (r"blocks_(\d+)_mlp_fc1", r"blocks.\1.mlp.fc1"),
        (r"blocks_(\d+)_mlp_fc2", r"blocks.\1.mlp.fc2"),        (r"token_refiner_blocks_(\d+)_attn_out_proj", r"token_refiner.blocks.\1.attn.out_proj"),
        (r"token_refiner_blocks_(\d+)_attn_qkv_proj", r"token_refiner.blocks.\1.attn.qkv_proj"),
        (r"token_refiner_blocks_(\d+)_mlp_fc1", r"token_refiner.blocks.\1.mlp.fc1"),
        (r"token_refiner_blocks_(\d+)_mlp_fc2", r"token_refiner.blocks.\1.mlp.fc2"),    ]

    for pattern, replacement in patterns:
        if re.fullmatch(pattern, module_key):
            return f"{re.sub(pattern, replacement, module_key)}.{suffix}"

    return None


def parse_key(key, suffixes):
    """MiniMax-bound :func:`native_adapter.parse_key`."""
    unflattened = _unflatten_lora_unet_key(key)
    if unflattened is not None:
        key = unflattened
    parsed = native_adapter.parse_key(key, suffixes, prefixes=KNOWN_PREFIXES)
    if parsed is None:
        return None
    prefix_used, base, suffix = parsed
    return prefix_used, base, normalize_mini_max_suffix(suffix)


def group_by_suffixes(state_dict, suffixes, *, prefixes=None, bare_prefixes=(), bare_diffusers_prefixes=()): # pylint: disable=unused-argument
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


def _split_qkv(base, target_prefix):
    if not base.endswith(".attn.qkv_proj"):
        return []
    stem = base[: -len(".attn.qkv_proj")]
    return [
        (f"{target_prefix}.{stem}.attn.{k}", native_adapter.ChunkSpec(idx=i, total=3))
        for i, k in enumerate(("to_q", "to_k", "to_v"))
    ]


def _transformer_block_targets(target_prefix, base):
    if base.endswith(".attn.qkv_proj"):
        return _split_qkv(base, target_prefix)
    if base.endswith(".attn.out_proj"):
        stem = base[: -len(".attn.out_proj")]
        return [(f"{target_prefix}.{stem}.attn.to_out.0", None)]
    if base.endswith(".mlp.fc1"):
        stem = base[: -len(".mlp.fc1")]
        return [(f"{target_prefix}.{stem}.ff.net.0.proj", None)]
    if base.endswith(".mlp.fc2"):
        stem = base[: -len(".mlp.fc2")]
        return [(f"{target_prefix}.{stem}.ff.net.2", None)]
    return [(f"{target_prefix}.{base}", None)]


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for MiniMax keys."""
    if prefix_used == "diffusion_model.":
        if base.startswith("transformer."):
            return [(base[len("transformer."):], None)]
        if base.startswith("text_encoder."):
            return [(base[len("text_encoder."):], None)]
        if base.startswith("text_encoders."):
            return [(base[len("text_encoders."):], None)]
        return [(base, None)]
    if prefix_used in ("diffusion_model.transformer.", "transformer.", "lora_transformer_"):
        return [(base, None)]
    if prefix_used in ("diffusion_model.transformer_blocks.", "transformer_blocks."):
        return [(f"transformer_blocks.{base}", None)]
    if prefix_used in ("diffusion_model.blocks.", "blocks."):
        return _transformer_block_targets("transformer_blocks", base)
    if prefix_used in ("diffusion_model.token_refiner.refiner_blocks.", "token_refiner.refiner_blocks."):
        return [(f"token_refiner.refiner_blocks.{base}", None)]
    if prefix_used in ("diffusion_model.token_refiner.", "token_refiner."):
        if base.startswith("blocks."):
            return _transformer_block_targets("token_refiner.refiner_blocks", base[len("blocks."):])
        return [(f"token_refiner.refiner_{base}", None)]
    if prefix_used in ("diffusion_model.final_layer.", "final_layer."):
        if base == "adaln_proj.linear":
            return [("norm_out.linear", None)]
        return [(base, None)]
    if prefix_used in ("diffusion_model.video_patch_proj.", "video_patch_proj."):
        return [("proj_in." + base.split(".", 1)[1], None)] if "." in base else [("proj_in", None)]
    if prefix_used in ("diffusion_model.audio_patch_proj.", "audio_patch_proj."):
        return [("audio_proj_in." + base.split(".", 1)[1], None)] if "." in base else [("audio_proj_in", None)]
    if prefix_used in ("diffusion_model.condition_proj.", "condition_proj."):
        return [("context_embedder." + base.split(".", 1)[1], None)] if "." in base else [("context_embedder", None)]
    if prefix_used in ("diffusion_model.time_embedder.proj_in.", "time_embedder.proj_in."):
        return [("time_embedder.linear_1." + base.split(".", 1)[1], None)] if "." in base else [("time_embedder.linear_1", None)]
    if prefix_used in ("diffusion_model.time_embedder.proj_out.", "time_embedder.proj_out."):
        return [("time_embedder.linear_2." + base.split(".", 1)[1], None)] if "." in base else [("time_embedder.linear_2", None)]
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


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    network_prefix=network_prefix_for,
    group_by_suffixes_fn=group_by_suffixes,
    arch_name="minimaxh3",
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
    """Run every MiniMax family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
