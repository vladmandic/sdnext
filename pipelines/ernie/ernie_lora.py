"""ERNIE-Image native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``ernieimage`` in ``allow_native``).

Entry points, one per family: :func:`try_load_lora` (plus DoRA),
:func:`try_load_lokr`, :func:`try_load_loha`, :func:`try_load_oft`.

Recognized key prefixes: ``diffusion_model.``, ``transformer.``,
``lora_unet_``, plus bare diffusers paths (``layers.``, ``adaLN_modulation.``,
``final_norm.``, ``final_linear.``).

``ErnieImageAttention`` has fully split ``to_q`` / ``to_k`` / ``to_v`` Linear
modules (no fused QKV) and ``ErnieImageFeedForward`` exposes ``gate_proj``,
``up_proj``, ``linear_fc2`` separately. resolve_targets is therefore a
straight passthrough; no chunking, no renames, no dispatch table.
"""

from modules.lora import native_adapter


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_adapter.KNOWN_PREFIXES_DEFAULT

BARE_DIFFUSERS_PREFIXES = (
    "layers.", "adaLN_modulation.", "final_norm.", "final_linear.",
)


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
    """ERNIE-bound :func:`native_adapter.parse_key`."""
    return native_adapter.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """ERNIE-bound :func:`native_adapter.group_by_suffixes`."""
    return native_adapter.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Passthrough for every recognized prefix. ERNIE has no fused targets or path
    renames; the base path is already the diffusers module path."""
    if prefix_used in ("diffusion_model.", "transformer.", "lora_unet_",
                       BARE_DIFFUSERS_PREFIX_USED, None):
        return [(base, None)]
    return []


# === Native loaders (thin wrappers over native_adapter generics) ===


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    arch_name="ernieimage",
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
    """Run every ERNIE family loader, merge any that match."""
    return native_adapter.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(try_load_lora, try_load_lokr, try_load_loha, try_load_oft),
    )
