"""Flux2/Klein native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``f2`` in ``allow_native``). Reads the safetensors directly and writes into
sdnext's existing ``network_layer_mapping``, returning a ``Network`` populated
with ``NetworkModule*`` entries that ``network_activate`` will apply.

Recognized key prefixes for every family: ``diffusion_model.``,
``transformer.``, ``lora_unet_``, ``lycoris_``, ``base_model.model.``
(PEFT save wrapper), bare BFL paths (e.g. ``double_blocks.``), and bare
diffusers paths (``transformer_blocks.`` / ``single_transformer_blocks.``,
produced by ``Flux2Transformer2DModel.save_lora_adapter()``). Diffusers-PEFT
``lora_A``/``lora_B`` are normalized to ``lora_down``/``lora_up`` and a
``.<adapter_name>.`` infix between the suffix and ``.weight`` (e.g.
``.lora_A.default.weight``) is stripped to match the standard suffix table.

BFL/kohya keys are mapped to diffusers paths via ``F2_SINGLE_MAP`` /
``F2_DOUBLE_MAP`` / ``F2_QKV_MAP``. Fused QKV in double_blocks emits three
Q/K/V targets each carrying a :class:`modules.lora.native_loader.ChunkSpec`
that the generic loaders use to chunk the up-weight or instantiate the
appropriate ``NetworkModule*Chunk`` variant.

Per-family fused-QKV handling is inherited from
:mod:`modules.lora.native_loader`; see the loader-by-loader notes there.

LyCORIS algorithm coverage relative to upstream
``KohakuBlueleaf/LyCORIS/lycoris/modules/``:

- Native: LoRA, LoKR, LoHA, OFT, BOFT, IA3, GLoRA, Norm, Full.
- Saved as standard LoRA: LoCon and DyLoRA. Both ``custom_state_dict``
  outputs collapse to ``lora_up.weight``/``lora_down.weight``/``alpha``
  (LoCon bakes its ``scalar`` into ``lora_up``; DyLoRA concats its
  per-block slabs into a max-rank matrix), so ``try_load_lora`` loads
  them losslessly relative to upstream's own export.
- Deferred: TLoRA. The file saves only ``q_layer.weight`` /
  ``p_layer.weight`` / ``lambda_layer`` / ``alpha``; the base SVD
  reference (``base_q`` / ``base_p`` / ``base_lambda``) that the
  delta math subtracts is unsaved by upstream design and the
  ``sig_type`` selection mode is unrecoverable from the file, so
  any loader has a silent-correctness gap for ``sig_type != 'principal'``.
  Files fail cleanly with "not loaded".

Diffusers-PEFT fallback (used when ``lora_force_diffusers`` is on) is preserved
via :func:`apply_patch`, which monkey-patches ``Flux2LoraLoaderMixin.lora_state_dict``
to inject the ``diffusion_model.`` prefix for bare-BFL keys and bake kohya
``.alpha`` scaling into ``lora_down`` weights.
"""

import os

from modules.logger import log
from modules.lora import native_loader
from modules.lora.native_loader import ChunkSpec


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_loader.KNOWN_PREFIXES_DEFAULT + ("lycoris_",)

BARE_FLUX_PREFIXES = (
    "single_blocks.", "double_blocks.", "img_in.", "txt_in.",
    "final_layer.", "time_in.", "single_stream_modulation.",
    "double_stream_modulation_",
)

BARE_DIFFUSERS_PREFIXES = ("single_transformer_blocks.", "transformer_blocks.")


# === BFL to diffusers mapping ===

# Single-block (single_transformer_blocks.{i}.<target>) - both projections are
# single fused diffusers modules, so no chunking is needed for any family.
F2_SINGLE_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
}

# Double-block non-QKV targets (transformer_blocks.{i}.<target>).
F2_DOUBLE_MAP = {
    "img_attn.proj": "attn.to_out.0",
    "txt_attn.proj": "attn.to_add_out",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

# Double-block fused QKV targets - diffusers exposes Q/K/V as separate modules,
# so resolve_targets emits three (path, ChunkSpec(idx, total=3)) entries.
F2_QKV_MAP = {
    "img_attn.qkv": ("attn", ["to_q", "to_k", "to_v"]),
    "txt_attn.qkv": ("attn", ["add_q_proj", "add_k_proj", "add_v_proj"]),
}

# Kohya underscore suffix -> BFL dot suffix. Used to convert kohya key fragments
# to look up F2_DOUBLE_MAP / F2_QKV_MAP.
KOHYA_SUFFIX_MAP = {
    "img_attn_proj": "img_attn.proj",
    "txt_attn_proj": "txt_attn.proj",
    "img_attn_qkv": "img_attn.qkv",
    "txt_attn_qkv": "txt_attn.qkv",
    "img_mlp_0": "img_mlp.0",
    "img_mlp_2": "img_mlp.2",
    "txt_mlp_0": "txt_mlp.0",
    "txt_mlp_2": "txt_mlp.2",
}


# === Re-exports for backward compatibility ===
# The offline test suite addresses these via the flux2_lora module surface.
# Re-export rather than asking tests to import native_loader directly.

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
    """Flux2-bound :func:`native_loader.parse_key`. Returns ``(prefix_used, base, suffix)`` or ``None``."""
    return native_loader.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_prefixes=BARE_FLUX_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """Flux2-bound :func:`native_loader.group_by_suffixes`."""
    return native_loader.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_prefixes=BARE_FLUX_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group key.

    For ``lora_unet_`` prefix, applies ``KOHYA_SUFFIX_MAP`` then ``F2_*_MAP``.
    For BFL / bare-BFL, applies ``F2_*_MAP`` directly. For ``transformer.``,
    ``lycoris_``, and bare-diffusers, returns the base verbatim with no chunking.
    Unrecognized prefixes return an empty list.
    """
    if prefix_used == "lora_unet_":
        return _kohya_to_diffusers_targets(base)
    if prefix_used in (None, "diffusion_model."):
        return _bfl_to_diffusers_targets(base)
    if prefix_used == "transformer.":
        return [(base, None)]
    if prefix_used == BARE_DIFFUSERS_PREFIX_USED:
        return [(base, None)]
    if prefix_used == "lycoris_":
        # base is an already-underscored diffusers path (e.g.
        # 'transformer_blocks_0_attn_add_k_proj'). The caller's network_key
        # construction does base.replace('.', '_'); for already-underscored
        # paths that's a no-op, so the network_key matches the entry stamped
        # by lora_convert.assign_network_names_to_compvis_modules
        # (e.g. 'lora_transformer_transformer_blocks_0_attn_add_k_proj').
        return [(base, None)]
    return []


def _kohya_to_diffusers_targets(stripped):
    """For kohya keys like ``double_blocks_0_img_attn_proj`` or ``single_blocks_5_linear1``."""
    targets: list[tuple[str, ChunkSpec | None]] = []
    if stripped.startswith("single_blocks_"):
        rest = stripped[len("single_blocks_"):]
        idx, _, suffix = rest.partition("_")
        if suffix in F2_SINGLE_MAP:
            targets.append((f"single_transformer_blocks.{idx}.{F2_SINGLE_MAP[suffix]}", None))
    elif stripped.startswith("double_blocks_"):
        rest = stripped[len("double_blocks_"):]
        idx, _, kohya_suffix = rest.partition("_")
        bfl_suffix = KOHYA_SUFFIX_MAP.get(kohya_suffix)
        if bfl_suffix is None:
            return targets
        if bfl_suffix in F2_DOUBLE_MAP:
            targets.append((f"transformer_blocks.{idx}.{F2_DOUBLE_MAP[bfl_suffix]}", None))
        elif bfl_suffix in F2_QKV_MAP:
            attn_prefix, proj_keys = F2_QKV_MAP[bfl_suffix]
            for i, proj_key in enumerate(proj_keys):
                targets.append((
                    f"transformer_blocks.{idx}.{attn_prefix}.{proj_key}",
                    ChunkSpec(idx=i, total=len(proj_keys)),
                ))
    return targets


def _bfl_to_diffusers_targets(base):
    """For BFL keys like ``double_blocks.0.img_attn.proj`` or ``single_blocks.5.linear1``."""
    targets: list[tuple[str, ChunkSpec | None]] = []
    parts = base.split(".")
    if len(parts) < 3:
        return targets
    block_type, block_idx, module_suffix = parts[0], parts[1], ".".join(parts[2:])
    if block_type == "single_blocks" and module_suffix in F2_SINGLE_MAP:
        targets.append((f"single_transformer_blocks.{block_idx}.{F2_SINGLE_MAP[module_suffix]}", None))
    elif block_type == "double_blocks":
        if module_suffix in F2_DOUBLE_MAP:
            targets.append((f"transformer_blocks.{block_idx}.{F2_DOUBLE_MAP[module_suffix]}", None))
        elif module_suffix in F2_QKV_MAP:
            attn_prefix, proj_keys = F2_QKV_MAP[module_suffix]
            for i, proj_key in enumerate(proj_keys):
                targets.append((
                    f"transformer_blocks.{block_idx}.{attn_prefix}.{proj_key}",
                    ChunkSpec(idx=i, total=len(proj_keys)),
                ))
    return targets


# === Native loaders (thin wrappers over native_loader generics) ===


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_prefixes=BARE_FLUX_PREFIXES,
    bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    arch_name="f2",
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
    """Single dispatcher entry point: run every family loader, merge any that match."""
    return native_loader.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )


# === Diffusers-PEFT path helpers (used when lora_force_diffusers is on) ===


def apply_lora_alphas(state_dict):
    """Bake kohya-format ``.alpha`` scaling into ``lora_down`` weights and remove alpha keys.

    Diffusers' Flux2 converter only handles ``lora_A``/``lora_B`` (or
    ``lora_down``/``lora_up``) keys. Kohya-format LoRAs store per-layer alpha
    values as separate ``.alpha`` keys that the converter does not consume,
    causing a ``ValueError`` on leftover keys. This matches the approach used
    by ``_convert_kohya_flux_lora_to_diffusers`` for Flux 1.
    """
    alpha_keys = [k for k in state_dict if k.endswith(".alpha")]
    if not alpha_keys:
        return state_dict
    for alpha_key in alpha_keys:
        base = alpha_key[:-len(".alpha")]
        down_key = f"{base}.lora_down.weight"
        if down_key not in state_dict:
            continue
        down_weight = state_dict[down_key]
        rank = down_weight.shape[0]
        alpha = state_dict.pop(alpha_key).item()
        scale = alpha / rank
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        state_dict[down_key] = down_weight * scale_down
        up_key = f"{base}.lora_up.weight"
        if up_key in state_dict:
            state_dict[up_key] = state_dict[up_key] * scale_up
    remaining = [k for k in state_dict if k.endswith(".alpha")]
    if remaining:
        log.debug(f'Network load: type=LoRA stripped {len(remaining)} orphaned alpha keys')
        for k in remaining:
            del state_dict[k]
    return state_dict


def preprocess_f2_keys(state_dict):
    """Add ``diffusion_model.`` prefix to bare BFL-format keys so
    ``Flux2LoraLoaderMixin``'s format detection routes them to the converter."""
    if any(k.startswith("diffusion_model.") or k.startswith("base_model.model.") for k in state_dict):
        return state_dict
    if any(k.startswith(p) for k in state_dict for p in BARE_FLUX_PREFIXES):
        log.debug("Network load: type=LoRA adding diffusion_model prefix for bare BFL-format keys")
        state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}
    return state_dict


patched = False


def apply_patch():
    """Patch ``Flux2LoraLoaderMixin.lora_state_dict`` to handle bare BFL-format keys.

    When a LoRA file has bare BFL keys (no ``diffusion_model.`` prefix), the
    original ``lora_state_dict`` won't detect them as AI toolkit format. This
    patch checks for bare keys after the original returns and adds the prefix +
    re-runs conversion. Used only on the diffusers-PEFT fallback path.
    """
    global patched # pylint: disable=global-statement
    if patched:
        return
    patched = True

    from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin
    original_lora_state_dict = Flux2LoraLoaderMixin.lora_state_dict.__func__

    @classmethod # pylint: disable=no-self-argument
    def patched_lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = preprocess_f2_keys(pretrained_model_name_or_path_or_dict)
            pretrained_model_name_or_path_or_dict = apply_lora_alphas(pretrained_model_name_or_path_or_dict)
        elif isinstance(pretrained_model_name_or_path_or_dict, (str, os.PathLike)):
            path = str(pretrained_model_name_or_path_or_dict)
            if path.endswith(".safetensors"):
                try:
                    from safetensors import safe_open
                    with safe_open(path, framework="pt") as f:
                        keys = list(f.keys())
                    needs_load = (
                        any(k.endswith(".alpha") for k in keys)
                        or (not any(k.startswith("diffusion_model.") or k.startswith("base_model.model.") for k in keys)
                            and any(k.startswith(p) for k in keys for p in BARE_FLUX_PREFIXES))
                    )
                    if needs_load:
                        from safetensors.torch import load_file
                        sd = load_file(path)
                        sd = preprocess_f2_keys(sd)
                        pretrained_model_name_or_path_or_dict = apply_lora_alphas(sd)
                except Exception:
                    pass
        return original_lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs)

    Flux2LoraLoaderMixin.lora_state_dict = patched_lora_state_dict
