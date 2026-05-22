"""Shared scaffolding for native adapter loaders.

The four native adapter loaders (z-image, chroma, ernie, flux2) all implement
the same algorithm:

1. Read the safetensors state dict
2. Test for family-specific markers; bail out if absent
3. Resolve the diffusers ``network_layer_mapping``
4. Group state-dict entries by ``(prefix, base)``
5. For each group, ask the arch to resolve targets (diffusers paths + chunk specs)
6. Instantiate a :class:`network.NetworkModule*` per resolved target
7. Return a populated :class:`network.Network` (or ``None`` if no matches)

This module holds the parts of that algorithm that don't vary between
architectures. Constants (suffix and marker tables), helpers (``has_marker``,
``resolve_mapping``, ``new_network``, ``finalize_network``, ``shapes_match``),
the parameterized parsing primitives (``parse_key``, ``group_by_suffixes``),
and the two cross-arch key normalizations (``unwrap_peft_wrapper`` and
``strip_peft_adapter_name``) live here.

The variance that does remain is captured by ``ChunkSpec`` (how a row range
of a fused weight is described) and the per-arch ``resolve_targets`` callable
each loader passes in (how a parsed ``(prefix, base)`` maps to one or more
diffusers paths plus optional chunk descriptors).

Per-arch loader modules import this module and pass their own ``prefixes``,
``bare_prefixes``, ``bare_diffusers_prefixes``, and ``resolve_targets`` to the
generic helpers. Loader business logic itself lands in subsequent commits.
"""

import os
import time
from dataclasses import dataclass

import torch

from modules import shared, sd_models
from modules.logger import log
from modules.lora import (
    lora_convert, network, network_boft, network_full, network_glora,
    network_hada, network_ia3, network_lokr, network_lora, network_norm,
    network_oft,
)
from modules.lora import lora_common as l


# Universal prefix list shared by every native arch loader. Per-arch loaders
# extend this with arch-specific entries (e.g. flux2 adds ``"lycoris_"``).
KNOWN_PREFIXES_DEFAULT = ("diffusion_model.", "transformer.", "lora_unet_")


# Sentinel ``prefix_used`` value emitted by :func:`parse_key` when a bare path
# starting with a member of ``bare_diffusers_prefixes`` matches. Loader
# ``resolve_targets`` callables dispatch on this string to pass the base path
# through verbatim (no rename required, the path is already in diffusers form).
BARE_DIFFUSERS_PREFIX_USED = "bare_diffusers"


# Default network-key prefix. Single-component arches (flux2, zimage, chroma,
# ernie) keep this default; multi-component arches (anima: transformer plus
# llm_adapter plus text_encoder) pass a callable that picks per ``prefix_used``.
NETWORK_PREFIX_DEFAULT = "lora_transformer_"


def _resolve_prefix(network_prefix, prefix_used):
    """Return the network-key prefix for one parsed group.

    ``network_prefix`` is either a literal string (single-component arches) or
    a ``Callable[[str | None], str]`` that picks based on which arch prefix
    was matched (multi-component arches route to ``lora_te_`` / ``lora_llm_adapter_``
    / ``lora_transformer_`` etc.).
    """
    return network_prefix(prefix_used) if callable(network_prefix) else network_prefix


SUFFIX_NORMALIZE = {
    "lora_A.weight": "lora_down.weight",
    "lora_B.weight": "lora_up.weight",
}


# === Family suffix tables ===
# Alpha / scale / bias / dora_scale flow into ``weights.w`` via the base
# ``network.NetworkModule.__init__`` and are listed here so they survive the
# suffix-filter pass in :func:`parse_key`.

LORA_SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight", ".lora_mid.weight",
    ".lora_A.weight",    ".lora_B.weight",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
LOKR_SUFFIXES = (
    ".lokr_w1", ".lokr_w2",
    ".lokr_w1_a", ".lokr_w1_b",
    ".lokr_w2_a", ".lokr_w2_b",
    ".lokr_t2",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
LOHA_SUFFIXES = (
    ".hada_w1_a", ".hada_w1_b",
    ".hada_w2_a", ".hada_w2_b",
    ".hada_t1",   ".hada_t2",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
OFT_SUFFIXES = (
    ".oft_blocks", ".oft_diag",
    ".alpha", ".dora_scale", ".bias", ".scale",
)
IA3_SUFFIXES = (
    ".weight", ".on_input",
    ".alpha", ".scale",
)
GLORA_SUFFIXES = (
    ".a1.weight", ".a2.weight",
    ".b1.weight", ".b2.weight",
    ".alpha", ".dora_scale", ".scale",
)
NORM_SUFFIXES = (
    ".w_norm", ".b_norm",
    ".alpha", ".scale",
)
FULL_SUFFIXES = (
    ".diff", ".diff_b",
    ".alpha", ".scale",
)


# === Family marker tables ===
# Presence of any marker substring anywhere in a key triggers a try-load attempt
# for that family. Markers are deliberately narrower than suffixes (e.g. IA3's
# ``.on_input`` rather than ``.weight``) so :func:`has_marker` does not light up
# on accidental overlaps with other families.

LORA_MARKERS = (
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight",
    # PEFT named-adapter saves embed the slot name as ``.lora_A.<name>.weight``;
    # the trailing-dot forms catch every variant.
    ".lora_A.", ".lora_B.",
)
LOKR_MARKERS = (".lokr_w1", ".lokr_w2")
LOHA_MARKERS = (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b")
OFT_MARKERS = (".oft_blocks", ".oft_diag")
IA3_MARKERS = (".on_input",)  # NOT .weight - too generic, overlaps every other family
GLORA_MARKERS = (".a1.weight", ".a2.weight", ".b1.weight", ".b2.weight")
NORM_MARKERS = (".w_norm",)
FULL_MARKERS = (".diff",)


# === Chunk descriptor ===


@dataclass(frozen=True)
class ChunkSpec:
    """How to slice a fused weight along dim 0 for one target module.

    Two forms supported:

    - Equal chunks (``idx`` + ``total``): fused QKV split into Q/K/V via
      ``torch.chunk(up, total, dim=0)[idx]``. Used by flux2 / z-image where
      Q, K and V have the same ``out_features``.
    - Row range (``start`` + ``end``): asymmetric partition via
      ``up[start:end]``. Used by chroma's single-block ``linear1`` which
      fuses Q / K / V / proj_mlp at unequal sizes
      (``[3072, 3072, 3072, 12288]``).

    Generic loaders check :attr:`is_equal_chunks` to decide between the two
    forms and select the appropriate ``NetworkModule*Chunk`` /
    ``NetworkModule*SliceChunk`` variant.
    """
    idx: int | None = None
    total: int | None = None
    start: int | None = None
    end: int | None = None

    @property
    def is_equal_chunks(self) -> bool:
        return self.idx is not None and self.total is not None


# === Key normalizations (applied universally by parse_key) ===


def unwrap_peft_wrapper(key):
    """Strip the ``base_model.model.`` prefix added by ``peft.save_pretrained``.

    PeftModel.save_pretrained prepends this wrapper to every adapter key. The
    content underneath can be any of the standard prefixes (BFL, diffusers PEFT,
    kohya, bare); a single strip lets the rest of :func:`parse_key` handle the
    unwrapped key normally. Mirrors the diffusers ``Flux2LoraLoaderMixin``
    behavior of renaming ``base_model.model.`` to ``diffusion_model.`` before
    feeding the converter.
    """
    if key.startswith("base_model.model."):
        return key[len("base_model.model."):]
    return key


def strip_peft_adapter_name(key):
    """Normalize ``.lora_[AB].<adapter_name>.weight`` to ``.lora_[AB].weight``.

    ``peft.PeftModel`` and the diffusers ``save_lora_adapter`` exporter embed
    the adapter slot name into the saved key (``"default"`` when not explicitly
    set). Strip a single non-dotted name segment so the suffix table matches
    without having to list every plausible adapter name.
    """
    for inner in (".lora_A.", ".lora_B."):
        idx = key.find(inner)
        if idx == -1:
            continue
        rest = key[idx + len(inner):]
        if rest == "weight" or not rest.endswith(".weight"):
            continue
        adapter_name = rest[:-len(".weight")]
        if adapter_name and "." not in adapter_name:
            return key[:idx] + inner + "weight"
    return key


# === Core helpers ===


def has_marker(state_dict, markers):
    """Substring scan: does any key in ``state_dict`` contain any marker?"""
    return any(any(m in k for m in markers) for k in state_dict)


def resolve_mapping():
    """Ensure ``network_layer_mapping`` is populated, return it (or empty dict)."""
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
    lora_convert.assign_network_names_to_compvis_modules(sd_model)
    return getattr(shared.sd_model, "network_layer_mapping", {}) or {}


def new_network(name, network_on_disk):
    """Construct an empty :class:`network.Network` with the file's mtime stamped."""
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    return net


def finalize_network(net, name, family, lora_scale, t0, unmapped=0, mismatch=0, skipped=0):
    """Emit the standard debug log line and return the populated network (or ``None``).

    Returns ``None`` when no modules were bound. Logs at debug only; loader
    callers can surface higher-level outcomes at info if needed.
    """
    if len(net.modules) == 0:
        if unmapped or mismatch or skipped:
            log.debug(
                f'Network load: type={family} name="{name}" native no-match'
                f' unmapped={unmapped} mismatch={mismatch} skipped={skipped}'
            )
        return None
    log.debug(
        f'Network load: type={family} name="{name}" native modules={len(net.modules)}'
        f' unmapped={unmapped} mismatch={mismatch} skipped={skipped} scale={lora_scale}'
    )
    l.timer.activate += time.time() - t0
    return net


def shapes_match(sd_module, down_w: torch.Tensor, up_w: torch.Tensor) -> bool:
    """LoRA-style rank-and-dim sanity check against the live module weight.

    Honors SDNQ-quantized modules by reading the original shape from the
    dequantizer rather than the packed weight tensor.
    """
    if not hasattr(sd_module, "weight"):
        return False
    if hasattr(sd_module, "sdnq_dequantizer"):
        mod_shape = sd_module.sdnq_dequantizer.original_shape
    else:
        mod_shape = sd_module.weight.shape
    if len(mod_shape) < 2 or len(down_w.shape) < 2 or len(up_w.shape) < 2:
        return False
    return down_w.shape[1] == mod_shape[1] and up_w.shape[0] == mod_shape[0]


# === Parsing primitives ===


def parse_key(key, suffixes, *, prefixes=KNOWN_PREFIXES_DEFAULT, bare_prefixes=(), bare_diffusers_prefixes=()):
    """Return ``(prefix_used, base, suffix_normalized)`` or ``None``.

    ``prefix_used`` is the matched element of ``prefixes``, ``BARE_DIFFUSERS_PREFIX_USED``
    if a member of ``bare_diffusers_prefixes`` matched, or ``None`` for a key
    that matched a member of ``bare_prefixes``. ``base`` is the path with prefix
    and suffix removed. ``suffix_normalized`` is the suffix (without the leading
    dot) after applying :data:`SUFFIX_NORMALIZE` (e.g. ``lora_A.weight`` becomes
    ``lora_down.weight``).

    Always applies :func:`unwrap_peft_wrapper` and :func:`strip_peft_adapter_name`
    to the raw key before format detection so callers do not have to opt in.
    """
    key = unwrap_peft_wrapper(key)
    key = strip_peft_adapter_name(key)
    prefix_used = None
    stripped = key
    for p in prefixes:
        if key.startswith(p):
            prefix_used = p
            stripped = key[len(p):]
            break
    if prefix_used is None:
        if any(key.startswith(p) for p in bare_diffusers_prefixes):
            prefix_used = BARE_DIFFUSERS_PREFIX_USED
        elif not any(key.startswith(p) for p in bare_prefixes):
            return None

    matched_suffix = None
    split_at = -1
    for marker in suffixes:
        if stripped.endswith(marker):
            split_at = len(stripped) - len(marker)
            matched_suffix = marker.lstrip(".")
            break
    if split_at < 0:
        return None

    base = stripped[:split_at]
    if not base:
        return None

    suffix = SUFFIX_NORMALIZE.get(matched_suffix, matched_suffix)
    return prefix_used, base, suffix


def group_by_suffixes(state_dict, suffixes, *, prefixes=KNOWN_PREFIXES_DEFAULT, bare_prefixes=(), bare_diffusers_prefixes=()):
    """Group state-dict entries by ``(prefix_used, base)``.

    Returns ``{(prefix_used, base): {suffix: tensor, ...}}`` where each suffix
    is the normalized form produced by :func:`parse_key`. Per-family loaders
    apply their own key-presence gates on each group (e.g. LoRA requires both
    ``lora_down.weight`` and ``lora_up.weight``).
    """
    groups: dict[tuple, dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        parsed = parse_key(
            key, suffixes,
            prefixes=prefixes,
            bare_prefixes=bare_prefixes,
            bare_diffusers_prefixes=bare_diffusers_prefixes,
        )
        if parsed is None:
            continue
        prefix_used, base, suffix = parsed
        slot = groups.get((prefix_used, base))
        if slot is None:
            slot = {}
            groups[(prefix_used, base)] = slot
        slot[suffix] = value
    return groups


# Surface ``sd_models.read_state_dict`` here so loader modules don't have to
# import ``sd_models`` directly; keeps the per-arch wrapper imports compact.
read_state_dict = sd_models.read_state_dict


# === Generic family loaders ===
#
# Each loader takes a per-arch ``resolve_targets`` callable returning a list of
# ``(diffusers_path, ChunkSpec | None)`` tuples for each parsed group. The
# loader builds the network key as ``network_prefix + path.replace(".", "_")``
# where ``network_prefix`` defaults to ``"lora_transformer_"`` and may be
# either a literal string (single-component arches) or a callable that picks
# per ``prefix_used`` (multi-component arches such as Anima that route to
# ``lora_te_`` or ``lora_llm_adapter_`` based on which arch prefix matched).
# The loader then instantiates the appropriate ``network.NetworkModule*`` subclass.
#
# Fused-target handling per family:
#
# - LoRA: chunk at load time (lora_up is sliced along dim 0). Both equal and
#   unequal ChunkSpec shapes are supported.
# - LoKR: defer to NetworkModuleLokrChunk (equal) or NetworkModuleLokrSliceChunk
#   (unequal) which materialize the Kronecker product once and return the
#   designated slice.
# - LoHA: only equal ChunkSpec is supported via NetworkModuleHadaChunk. Unequal
#   slices and Tucker-decomposed LoHAs on fused targets are skipped with a
#   warning (no slice variant exists, and Tucker keys cannot arise on Linear
#   layers per LyCORIS upstream — see network_hada.NetworkModuleHadaChunk).
# - OFT/BOFT: no chunk variant exists; fused targets are skipped with a
#   warning. Discrimination is by ``oft_blocks.ndim`` (3-D OFT, 4-D BOFT),
#   mirroring upstream LyCORIS ``algo_check``.


def _slice_lora_chunk(w, chunk: ChunkSpec):
    """Return a shallow copy of ``w`` with ``lora_up.weight`` sliced per ``chunk``.

    Equal-chunks form uses ``torch.chunk`` (faster for the symmetric case);
    row-range form uses tensor slicing for arbitrary partitions.
    """
    up = w["lora_up.weight"]
    if chunk.is_equal_chunks:
        sliced = torch.chunk(up, chunk.total, dim=0)[chunk.idx].contiguous()
    else:
        sliced = up[chunk.start:chunk.end].contiguous()
    out = dict(w)
    out["lora_up.weight"] = sliced
    return out


def try_load_lora(name, network_on_disk, lora_scale, *,
                  resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                  bare_prefixes=(), bare_diffusers_prefixes=(),
                  network_prefix=NETWORK_PREFIX_DEFAULT,
                  arch_name="generic"):
    """Generic LoRA loader (handles DoRA via the universal ``finalize_updown`` hook).

    Fused targets are chunked at load time by slicing ``lora_up`` along dim 0;
    the down-side is shared across the resolved targets.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, LORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, LORA_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    mismatch = 0
    for (prefix, base), w in groups.items():
        if "lora_down.weight" not in w or "lora_up.weight" not in w:
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, chunk in resolve_targets(prefix, base):
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue

            target_w = _slice_lora_chunk(w, chunk) if chunk is not None else w

            if not shapes_match(sd_module, target_w["lora_down.weight"], target_w["lora_up.weight"]):
                log.warning(
                    f'Network load: type=LoRA name="{name}" arch={arch_name} key={network_key}'
                    f' lora={target_w["lora_down.weight"].shape[1]}x{target_w["lora_up.weight"].shape[0]}'
                    f' module={getattr(sd_module, "weight", None).shape if hasattr(sd_module, "weight") else "?"}'
                    f' shape mismatch'
                )
                mismatch += 1
                continue

            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=target_w, sd_module=sd_module)
            net.modules[network_key] = network_lora.NetworkModuleLora(net, nw)

    return finalize_network(net, name, "LoRA", lora_scale, t0, unmapped=unmapped, mismatch=mismatch)


def try_load_lokr(name, network_on_disk, lora_scale, *,
                  resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                  bare_prefixes=(), bare_diffusers_prefixes=(),
                  network_prefix=NETWORK_PREFIX_DEFAULT,
                  arch_name="generic"):
    """Generic LoKR loader.

    Stores only the compact LoKR factors and dispatches to
    :class:`network_lokr.NetworkModuleLokrChunk` (equal chunks) or
    :class:`network_lokr.NetworkModuleLokrSliceChunk` (row range) at apply
    time. Both materialize ``kron(w1, w2)`` once per forward pass and return
    the designated slice; full materialization happens lazily inside the
    module rather than at load.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, LOKR_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, LOKR_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    for (prefix, base), w in groups.items():
        has_1 = "lokr_w1" in w or ("lokr_w1_a" in w and "lokr_w1_b" in w)
        has_2 = "lokr_w2" in w or ("lokr_w2_a" in w and "lokr_w2_b" in w)
        if not (has_1 and has_2):
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, chunk in resolve_targets(prefix, base):
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            if chunk is None:
                net.modules[network_key] = network_lokr.NetworkModuleLokr(net, nw)
            elif chunk.is_equal_chunks:
                net.modules[network_key] = network_lokr.NetworkModuleLokrChunk(net, nw, chunk.idx, chunk.total)
            else:
                net.modules[network_key] = network_lokr.NetworkModuleLokrSliceChunk(net, nw, chunk.start, chunk.end)

    return finalize_network(net, name, "LoKR", lora_scale, t0, unmapped=unmapped)


def try_load_loha(name, network_on_disk, lora_scale, *,
                  resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                  bare_prefixes=(), bare_diffusers_prefixes=(),
                  network_prefix=NETWORK_PREFIX_DEFAULT,
                  arch_name="generic"):
    """Generic LoHA (Hadamard product) loader.

    Standard non-Tucker LoHA on fused targets uses
    :class:`network_hada.NetworkModuleHadaChunk` for equal-chunks dispatch.
    Tucker-decomposed LoHAs on fused targets and any unequal-chunks dispatch
    are skipped with a warning: no slice variant exists, and Tucker keys
    cannot arise on Linear layers per LyCORIS upstream.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, LOHA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, LOHA_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not all(k in w for k in ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")):
            continue
        is_tucker = "hada_t1" in w or "hada_t2" in w
        targets = resolve_targets(prefix, base)
        is_fused = any(t[1] is not None for t in targets)
        if is_fused and is_tucker:
            log.warning(f'Network load: type=LoHA name="{name}" arch={arch_name} key={base} Tucker fused QKV skipped (unsupported)')
            skipped += 1
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, chunk in targets:
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            if chunk is None:
                net.modules[network_key] = network_hada.NetworkModuleHada(net, nw)
            elif chunk.is_equal_chunks:
                net.modules[network_key] = network_hada.NetworkModuleHadaChunk(net, nw, chunk.idx, chunk.total)
            else:
                log.warning(f'Network load: type=LoHA name="{name}" arch={arch_name} key={network_key} unequal fused chunks unsupported')
                skipped += 1

    return finalize_network(net, name, "LoHA", lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_oft(name, network_on_disk, lora_scale, *,
                 resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                 bare_prefixes=(), bare_diffusers_prefixes=(),
                 network_prefix=NETWORK_PREFIX_DEFAULT,
                 arch_name="generic"):
    """Generic OFT/BOFT loader.

    OFT and BOFT share the ``oft_blocks`` save key and are discriminated by
    tensor dimensionality (3-D OFT, 4-D BOFT), mirroring upstream
    ``algo_check``. Both kohya (``oft_blocks`` + alpha-as-constraint) and
    LyCORIS (``oft_diag``) OFT layouts route through
    :class:`network_oft.NetworkModuleOFT`; BOFT routes through
    :class:`network_boft.NetworkModuleBOFT`.

    Fused targets are skipped with a warning for both algorithms: OFT block
    structure (and BOFT's per-stage block partition) is tied to the target
    module's ``out_features``, so a per-Q/K/V split would require re-deriving
    the rotations per chunk.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, OFT_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, OFT_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not ("oft_blocks" in w or "oft_diag" in w):
            continue
        is_boft = "oft_blocks" in w and w["oft_blocks"].ndim == 4
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type={"BOFT" if is_boft else "OFT"} name="{name}" arch={arch_name} key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, _ in targets:
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            if is_boft:
                net.modules[network_key] = network_boft.NetworkModuleBOFT(net, nw)
            else:
                net.modules[network_key] = network_oft.NetworkModuleOFT(net, nw)

    return finalize_network(net, name, "OFT", lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_ia3(name, network_on_disk, lora_scale, *,
                 resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                 bare_prefixes=(), bare_diffusers_prefixes=(),
                 network_prefix=NETWORK_PREFIX_DEFAULT,
                 arch_name="generic"):
    """Generic IA3 loader.

    IA3 stores a per-row or per-column scale vector keyed under ``.weight``
    plus an ``.on_input`` flag selecting which axis. ``.weight`` alone is too
    generic for the marker scan (it overlaps every other family's
    ``.lora_down.weight`` / ``.hada_w*`` keys), so the marker gate insists on
    ``.on_input`` while the suffix table includes both.

    Fused targets are skipped with a warning. There is no real-world IA3-on-DiT
    prevalence to justify the asymmetry between ``on_input=True`` (which would
    replicate cleanly across Q/K/V) and ``on_input=False`` (which would need
    output-axis slicing).
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, IA3_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, IA3_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not ("weight" in w and "on_input" in w):
            continue
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type=IA3 name="{name}" arch={arch_name} key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, _ in targets:
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_ia3.NetworkModuleIa3(net, nw)

    return finalize_network(net, name, "IA3", lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_glora(name, network_on_disk, lora_scale, *,
                   resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                   bare_prefixes=(), bare_diffusers_prefixes=(),
                   network_prefix=NETWORK_PREFIX_DEFAULT,
                   arch_name="generic"):
    """Generic GLoRA loader.

    GLoRA stores four low-rank components (``a1`` / ``a2`` / ``b1`` / ``b2``)
    and computes ``W_delta = w2b @ w1b + (target @ w2a) @ w1a``; the second
    term is target-dependent. Fused targets are skipped: the target-dependent
    term doesn't slice cleanly across projections without redirecting
    calc_updown to a fused proxy weight, and the file pattern is vanishingly
    rare on DiT architectures.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, GLORA_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, GLORA_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if not all(k in w for k in ("a1.weight", "a2.weight", "b1.weight", "b2.weight")):
            continue
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type=GLoRA name="{name}" arch={arch_name} key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, _ in targets:
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_glora.NetworkModuleGLora(net, nw)

    return finalize_network(net, name, "GLoRA", lora_scale, t0, unmapped=unmapped, skipped=skipped)


def try_load_norm(name, network_on_disk, lora_scale, *,
                  resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                  bare_prefixes=(), bare_diffusers_prefixes=(),
                  network_prefix=NETWORK_PREFIX_DEFAULT,
                  arch_name="generic"):
    """Generic Norm (LayerNorm / RMSNorm weight + bias delta) loader.

    Norm targets are never fused, so the chunk dispatch is dropped.

    Loader-local stamping: ``lora_convert.assign_network_names_to_compvis_modules``
    deliberately skips setting ``module.network_layer_name`` for transformer
    norm modules (except SD3) because of legacy CompVis UNet collisions. This
    loader bypasses the guard locally: for each target it actually binds, it
    sets ``network_layer_name`` directly on the host module so
    ``network_activate`` will apply the delta. Stamping is idempotent and only
    touches modules a Norm adapter explicitly targets.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, NORM_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, NORM_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    for (prefix, base), w in groups.items():
        if "w_norm" not in w:
            continue
        targets = resolve_targets(prefix, base)
        if not targets:
            unmapped += 1
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, chunk in targets:
            if chunk is not None:
                continue  # norm targets are not fused
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            if not getattr(sd_module, "network_layer_name", None):
                sd_module.network_layer_name = network_key
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_norm.NetworkModuleNorm(net, nw)

    return finalize_network(net, name, "Norm", lora_scale, t0, unmapped=unmapped)


def try_load_full(name, network_on_disk, lora_scale, *,
                  resolve_targets, prefixes=KNOWN_PREFIXES_DEFAULT,
                  bare_prefixes=(), bare_diffusers_prefixes=(),
                  network_prefix=NETWORK_PREFIX_DEFAULT,
                  arch_name="generic"):
    """Generic Full (full-rank weight delta) loader.

    Full adapters carry a complete weight delta (``diff``, same shape as the
    host weight) and an optional bias delta (``diff_b``). Fused targets are
    skipped with a warning: ``diff`` has the host weight's full shape and
    row-slicing across three projections is well-defined arithmetically, but
    no chunk class exists.
    """
    t0 = time.time()
    state_dict = read_state_dict(network_on_disk.filename, what="network")
    if not has_marker(state_dict, FULL_MARKERS):
        return None

    mapping = resolve_mapping()
    net = new_network(name, network_on_disk)
    groups = group_by_suffixes(
        state_dict, FULL_SUFFIXES,
        prefixes=prefixes,
        bare_prefixes=bare_prefixes,
        bare_diffusers_prefixes=bare_diffusers_prefixes,
    )

    unmapped = 0
    skipped = 0
    for (prefix, base), w in groups.items():
        if "diff" not in w:
            continue
        targets = resolve_targets(prefix, base)
        if any(t[1] is not None for t in targets):
            log.warning(f'Network load: type=Full name="{name}" arch={arch_name} key={base} fused QKV skipped (unsupported)')
            skipped += 1
            continue
        arch_prefix = _resolve_prefix(network_prefix, prefix)
        for diffusers_path, _ in targets:
            network_key = arch_prefix + diffusers_path.replace(".", "_")
            sd_module = mapping.get(network_key)
            if sd_module is None:
                unmapped += 1
                continue
            nw = network.NetworkWeights(network_key=network_key, sd_key=network_key, w=w, sd_module=sd_module)
            net.modules[network_key] = network_full.NetworkModuleFull(net, nw)

    return finalize_network(net, name, "Full", lora_scale, t0, unmapped=unmapped, skipped=skipped)


# === Per-arch umbrella ===


def try_load_chain(name, network_on_disk, lora_scale, family_loaders):
    """Run each family loader in order and merge any non-None results.

    Per-arch loader modules expose a ``try_load(name, nod, scale)`` entry point
    that the dispatcher in ``modules.lora.lora_load`` calls. That entry point
    is a thin wrapper around this helper: it passes ``family_loaders`` as a
    tuple of partial-applied generic loaders, each already bound to the arch's
    ``resolve_targets`` and prefix tuples.
    """
    net = None
    for try_fn in family_loaders:
        sub = try_fn(name, network_on_disk, lora_scale)
        if sub is None:
            continue
        if net is None:
            net = sub
        else:
            net.modules.update(sub.modules)
    return net
