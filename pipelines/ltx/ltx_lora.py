"""LTX-Video native adapter loader.

Runs when :func:`modules.lora.lora_overrides.get_method` returns ``'native'``
(``lora_force_diffusers`` off and ``ltxvideo`` in ``allow_native``).

Entry points: :func:`try_load_lora` (plus DoRA), :func:`try_load_lokr`,
:func:`try_load_loha`, :func:`try_load_oft`, :func:`try_load_ia3`,
:func:`try_load_glora`, :func:`try_load_norm`, :func:`try_load_full`.

Recognized key prefixes: ``diffusion_model.``, ``transformer.``,
``lora_unet_``, plus the bare diffusers paths the LTX transformer exposes.

Both ``LTXVideoTransformer3DModel`` (0.9.x) and ``LTX2VideoTransformer3DModel``
(2.x) construct attention with fully split ``to_q`` / ``to_k`` / ``to_v`` /
``to_out.0`` Linear modules (no fused QKV) and feed-forward via ``ff.net.0.proj``
/ ``ff.net.2``. resolve_targets is therefore a straight passthrough; no
chunking, no renames, no dispatch table. Targets that only exist on one family
(``audio_*`` / ``video_to_audio_attn`` on 2.x, decoder-style modules on 0.9.x)
land in ``unmapped`` on the other family without affecting the rest of the load.
"""

from modules.lora import native_loader


# === Arch-specific prefix configuration ===

KNOWN_PREFIXES = native_loader.KNOWN_PREFIXES_DEFAULT

BARE_DIFFUSERS_PREFIXES = (
    "transformer_blocks.",
    "adaln_single.", "audio_adaln_single.", "audio_prompt_adaln_single.",
    "patchify_proj.", "audio_patchify_proj.",
    "proj_out.", "audio_proj_out.",
    "caption_projection.",
    "rope.",
)


# === Re-exports for test/back-compat ===

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
    """LTX-bound :func:`native_loader.parse_key`."""
    return native_loader.parse_key(
        key, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


def group_by_suffixes(state_dict, suffixes):
    """LTX-bound :func:`native_loader.group_by_suffixes`."""
    return native_loader.group_by_suffixes(
        state_dict, suffixes,
        prefixes=KNOWN_PREFIXES,
        bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    )


# === Target resolution (arch-specific) ===


def resolve_targets(prefix_used, base):
    """Passthrough for every recognized prefix. LTX has no fused targets or
    path renames; the base path is already the diffusers module path."""
    if prefix_used in ("diffusion_model.", "transformer.", "lora_unet_",
                       BARE_DIFFUSERS_PREFIX_USED, None):
        return [(base, None)]
    return []


# === Native loaders (thin wrappers over native_loader generics) ===


_BIND_KWARGS = dict(
    resolve_targets=resolve_targets,
    prefixes=KNOWN_PREFIXES,
    bare_diffusers_prefixes=BARE_DIFFUSERS_PREFIXES,
    arch_name="ltxvideo",
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
    """Run every LTX family loader, merge any that match."""
    return native_loader.try_load_chain(
        name, network_on_disk, lora_scale,
        family_loaders=(
            try_load_lora, try_load_lokr, try_load_loha, try_load_oft,
            try_load_ia3, try_load_glora, try_load_norm, try_load_full,
        ),
    )
