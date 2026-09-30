"""Qwen-Image 2.1 native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``qwen21`` in ``allow_native``).

Entry points, one per family: :func:`try_load_lora` (plus DoRA),
:func:`try_load_lokr`, :func:`try_load_loha`, :func:`try_load_oft`,
:func:`try_load_ia3`, :func:`try_load_glora`, :func:`try_load_norm`,
:func:`try_load_full`.

``QwenImage21Transformer2DModel`` keeps attention split (``to_q`` / ``to_k`` /
``to_v`` / ``to_out.0``) and the SwiGLU split (``img_mlp.gate_layer`` /
``img_mlp.proj`` / ``img_mlp.out``), so diffusers-named keys bind verbatim
under every prefix. ComfyUI checkpoints fuse the SwiGLU input as
``img_mlp.gate_up`` with rows ``[gate_layer; proj]``, and so do the LoRAs
ai-toolkit and musubi-tuner train against them; :func:`resolve_targets` splits
that projection into the two diffusers modules.

Parallel decoding exports (alibaba-pai Fun-Acc) carry their own sigma grid, so
:data:`PDD` needs no scheduler mapping: the pin hands the pipeline that grid.
"""

from modules.lora import native_adapter, network_pdd
from modules.lora.native_adapter import ChunkSpec


# One output projection on the pipeline scheduler; with explicit sigmas the step count is the interval count.
PDD = network_pdd.ArchSpec()


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT


# Fused SwiGLU input in dotted and kohya-flattened form, and the split modules its row halves land on.
GATE_UP_SUFFIXES = (".img_mlp.gate_up", "_img_mlp_gate_up")
GATE_UP_TARGETS = (("gate_layer", ChunkSpec(idx=0, total=2)), ("proj", ChunkSpec(idx=1, total=2)))


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
    """Qwen-Image 2.1-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """Qwen-Image 2.1-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Return ``[(diffusers_path, ChunkSpec | None), ...]`` for a parsed group key.

    A fused ``img_mlp.gate_up`` splits into ``gate_layer`` and ``proj`` under any
    prefix, dotted or flattened. Everything else is already a diffusers path:
    ``diffusion_model.`` and ``lora_unet_`` bind here, the universal passthrough
    prefixes bind upstream in :func:`native_adapter.resolve_group_targets`.
    """
    for fused in GATE_UP_SUFFIXES:
        if base.endswith(fused):
            sep = fused[0]
            stem = base[:-len(fused)]
            return [(f"{stem}{sep}img_mlp{sep}{leaf}", chunk) for leaf, chunk in GATE_UP_TARGETS]
    if prefix_used in ("diffusion_model.", "lora_unet_"):
        return [(base, None)]
    return []


# === Native loaders (thin wrappers over native_adapter generics) ===


BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    arch_name="qwen21",
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
    """Run every Qwen-Image 2.1 family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
            network_pdd.try_load,
        ),
    )
