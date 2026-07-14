"""Z-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``zimage`` in ``allow_native``).

Entry points, one per family: :func:`try_load_lora` (plus DoRA),
:func:`try_load_lokr`, :func:`try_load_loha`, :func:`try_load_oft`,
:func:`try_load_ia3`, :func:`try_load_glora`, :func:`try_load_norm`,
:func:`try_load_full`.

Recognized key prefixes: ``diffusion_model.``, ``transformer.``,
``lora_unet_``, or bare paths starting with the known block-level prefixes
(``layers.``, ``noise_refiner.``, ``context_refiner.``).

Checkpoint names that differ from the diffusers module tree are rewritten by
:func:`resolve_targets`: qk-norms (``attention.q_norm`` / ``k_norm`` ->
``attention.norm_q`` / ``norm_k``) and the non-block targets in
``ZIMAGE_EXTRA_MAP`` (``x_embedder`` and ``final_layer.*`` live in ModuleDicts
keyed by ``"{patch_size}-{f_patch_size}"``, read from the live model by
:func:`patch_keys`). ``t_embedder.mlp.N`` and ``cap_embedder.N`` already match
and pass through.

Pre-refactor Z-Image attention layouts (fused ``attention.qkv``, bare
``attention.out`` / ``attention.wo``) are rewritten by :func:`resolve_targets`
to the current diffusers ``to_q``/``to_k``/``to_v`` and ``to_out.0``. For
LoRA the fused qkv up-weight is chunked at load time; for LoKR the split is
deferred to apply time via :class:`network_lokr.NetworkModuleLokrChunk`.

Fused ``attention.qkv`` for LoHA and OFT is skipped with a warning: no
slice variant exists for LoHA's Hadamard product on Linear targets, and an
OFT block structure is tied to the target module's ``out_features`` so a
Q/K/V split is not a drop-in.
"""

from modules.lora import native_adapter
from modules.lora.native_adapter import ChunkSpec


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

BARE_DIFFUSERS_PREFIXES = ("layers.", "noise_refiner.", "context_refiner.")

# Checkpoint qk-norm names vs the diffusers attention module names.
ZIMAGE_NORM_ALIASES = {
    ".attention.q_norm": ".attention.norm_q",
    ".attention.k_norm": ".attention.norm_k",
}

# Non-block targets. The patch embedder and the final layer live in ModuleDicts keyed by
# "{patch_size}-{f_patch_size}", so the diffusers path carries a key the checkpoint does not:
# {key} is filled from the live model. t_embedder.mlp.N and cap_embedder.N already match and
# pass through verbatim.
ZIMAGE_EXTRA_MAP = {
    "x_embedder": "all_x_embedder.{key}",
    "final_layer.linear": "all_final_layer.{key}.linear",
    "final_layer.adaLN_modulation.1": "all_final_layer.{key}.adaLN_modulation.1",
}

ZIMAGE_EXTRA_KOHYA_MAP = {k.replace(".", "_"): v for k, v in ZIMAGE_EXTRA_MAP.items()}

# Both shipped Z-Image variants (Base, Turbo) build a single entry; read from the live model and
# fall back to it. Tests patch this directly.
PATCH_KEYS_DEFAULT = ["2-1"]


def patch_keys():
    """ModuleDict keys of the live transformer's ``all_x_embedder``."""
    try:
        from modules import shared
        pipe = getattr(shared.sd_model, "pipe", shared.sd_model)
        embedder = getattr(getattr(pipe, "transformer", None), "all_x_embedder", None)
        keys = list(embedder.keys()) if embedder is not None else []
        if keys:
            return keys
    except Exception:
        pass
    return PATCH_KEYS_DEFAULT


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
    """Z-Image-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """Z-Image-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group key.

    Handles two legacy Z-Image attention layouts:

    - Fused ``attention.qkv`` is split into three Q/K/V targets, each carrying
      a :class:`ChunkSpec` for the loader to apply.
    - ``attention.out`` / ``attention.out.0`` / ``attention.wo`` are aliased to
      the current diffusers ``attention.to_out.0`` path.

    Everything else (modern split-attention paths, MLP, norms, embedders) is
    returned verbatim.

    Universal passthrough prefixes are handled upstream by
    :func:`native_adapter.resolve_group_targets`.
    """
    if prefix_used == "lora_unet_":
        return _underscore_to_diffusers_targets(base)
    if prefix_used in (None, "diffusion_model."):
        return _dotted_to_diffusers_targets(base)
    return []


def _dotted_to_diffusers_targets(base):
    """For BFL / bare-BFL keys like ``layers.0.attention.qkv``."""
    extra = ZIMAGE_EXTRA_MAP.get(base)
    if extra is not None:
        return [(extra.format(key=key), None) for key in patch_keys()]
    if base.endswith(".attention.qkv"):
        stem = base[:-len(".attention.qkv")]
        return [
            (f"{stem}.attention.to_q", ChunkSpec(idx=0, total=3)),
            (f"{stem}.attention.to_k", ChunkSpec(idx=1, total=3)),
            (f"{stem}.attention.to_v", ChunkSpec(idx=2, total=3)),
        ]
    for alias, renamed in ZIMAGE_NORM_ALIASES.items():
        if base.endswith(alias):
            return [(base[:-len(alias)] + renamed, None)]
    for alias in (".attention.out.0", ".attention.out", ".attention.wo"):
        if base.endswith(alias):
            stem = base[:-len(alias)]
            return [(f"{stem}.attention.to_out.0", None)]
    return [(base, None)]


def _underscore_to_diffusers_targets(base):
    """For kohya flat-underscore keys like ``layers_0_attention_qkv``."""
    extra = ZIMAGE_EXTRA_KOHYA_MAP.get(base)
    if extra is not None:
        return [(extra.format(key=key), None) for key in patch_keys()]
    if base.endswith("_attention_qkv"):
        stem = base[:-len("_attention_qkv")]
        return [
            (f"{stem}_attention_to_q", ChunkSpec(idx=0, total=3)),
            (f"{stem}_attention_to_k", ChunkSpec(idx=1, total=3)),
            (f"{stem}_attention_to_v", ChunkSpec(idx=2, total=3)),
        ]
    for alias, renamed in ZIMAGE_NORM_ALIASES.items():
        underscored = alias.replace(".", "_")
        if base.endswith(underscored):
            return [(base[:-len(underscored)] + renamed.replace(".", "_"), None)]
    for alias in ("_attention_out_0", "_attention_out", "_attention_wo"):
        if base.endswith(alias):
            stem = base[:-len(alias)]
            return [(f"{stem}_attention_to_out_0", None)]
    return [(base, None)]


# === Native loaders (thin wrappers over native_adapter generics) ===


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    arch_name="zimage",
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
    """Run every Z-Image family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
