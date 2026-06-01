"""Z-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``zimage`` in ``allow_native``).

Entry points, one per family: :func:`try_load_lora` (plus DoRA),
:func:`try_load_lokr`, :func:`try_load_loha`, :func:`try_load_oft`.

Recognized key prefixes: ``diffusion_model.``, ``transformer.``,
``lora_unet_``, or bare paths starting with the known block-level prefixes
(``layers.``, ``noise_refiner.``, ``context_refiner.``).

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
    """
    if prefix_used == "lora_unet_":
        return _underscore_to_diffusers_targets(base)
    if prefix_used == "transformer.":
        return [(base, None)]
    if prefix_used == BARE_DIFFUSERS_PREFIX_USED:
        return [(base, None)]
    if prefix_used in (None, "diffusion_model."):
        return _dotted_to_diffusers_targets(base)
    return []


def _dotted_to_diffusers_targets(base):
    """For BFL / bare-BFL keys like ``layers.0.attention.qkv``."""
    if base.endswith(".attention.qkv"):
        stem = base[:-len(".attention.qkv")]
        return [
            (f"{stem}.attention.to_q", ChunkSpec(idx=0, total=3)),
            (f"{stem}.attention.to_k", ChunkSpec(idx=1, total=3)),
            (f"{stem}.attention.to_v", ChunkSpec(idx=2, total=3)),
        ]
    for alias in (".attention.out.0", ".attention.out", ".attention.wo"):
        if base.endswith(alias):
            stem = base[:-len(alias)]
            return [(f"{stem}.attention.to_out.0", None)]
    return [(base, None)]


def _underscore_to_diffusers_targets(base):
    """For kohya flat-underscore keys like ``layers_0_attention_qkv``."""
    if base.endswith("_attention_qkv"):
        stem = base[:-len("_attention_qkv")]
        return [
            (f"{stem}_attention_to_q", ChunkSpec(idx=0, total=3)),
            (f"{stem}_attention_to_k", ChunkSpec(idx=1, total=3)),
            (f"{stem}_attention_to_v", ChunkSpec(idx=2, total=3)),
        ]
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


def try_load(name, network_on_disk, lora_scale):
    """Run every Z-Image family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(try_load_lora, try_load_lokr, try_load_loha, try_load_oft),
    )
