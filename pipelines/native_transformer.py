"""Generic native loader for DiT transformers and bundled sibling components.

Loads a single-safetensors file into a diffusers (or custom) transformer class
when the user selects an override via the UNET dropdown (``shared.opts.sd_unet``).
Bypasses :func:`diffusers.loaders.FromOriginalModelMixin.from_single_file` so
sdnext owns prefix detection, optional sibling partitioning, dtype/quant/offload
handling, and explicit validation of missing/unexpected keys.

The per-arch knobs are captured in :class:`TransformerSpec`. Each pipeline
defines its spec in ``pipelines/<arch>/__init__.py`` and passes it explicitly
to :func:`load` (or to :func:`pipelines.generic.load_transformer` via the
``native_spec`` kwarg). No class-keyed registry: two pipelines may share a
transformer class but need different specs (e.g. Anima vs raw Cosmos both
use ``CosmosTransformer3DModel`` but Anima has a bundled ``llm_adapter``
sibling). Pipelines without a custom spec fall back to
:func:`make_default_spec`, which opportunistically picks up a real converter
from diffusers' ``SINGLE_FILE_LOADABLE_CLASSES`` table.

Algorithm:

1. Read the safetensors state dict (.gguf and .pth are rejected up front).
2. Drop keys belonging to known companion component families (all-in-one
   exports bundle the text encoder and VAE under prefixes like
   ``cond_stage_model.`` / ``first_stage_model.``; sdnext sources those
   components elsewhere).
3. Detect and strip one of the spec's known prefixes (raises on mixed prefixes).
4. Check forbidden markers (catches structural mismatches like Cosmos 1.0 keys
   in a Cosmos 2.0 loader).
5. Partition off sibling component keys (e.g. Anima's bundled ``llm_adapter.*``).
6. Run the spec's converter if present (else pass through unchanged).
7. Fetch ``<subfolder>/config.json`` from the base repo, instantiate via
   ``cls.from_config``, ``load_state_dict(strict=False)``, validate, dtype-cast,
   quantize, and offload-place.
8. Repeat the build for each populated sibling (no converter, no quant by
   default; sibling weights are read raw from the bundled file).

Returns ``(transformer, sibling_components_dict)``. The dict is empty for
arches with no siblings; for Anima it carries the ``llm_adapter`` if the
community file bundled one, else the empty dict and the caller falls back to
loading the adapter from the base repo.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Callable

import huggingface_hub as hf
import torch

from modules import shared, devices, sd_models, model_quant, errors
from modules.logger import log, console


DEFAULT_PREFIXES: tuple[str, ...] = (
    "model.diffusion_model.",
    "diffusion_model.",
    "net.",
)
DEFAULT_ACCEPTABLE_MISSING: tuple[str, ...] = (
    "rope.",
    "pos_embedder.",
    "learnable_pos_embed.",
)
DEFAULT_IGNORED_PREFIXES: tuple[str, ...] = (
    "cond_stage_model.",
    "conditioner.",
    "first_stage_model.",
    "text_encoders.",
    "vae.",
)
COMFY_QUANT_MARKER = ".comfy_quant"
COMFY_QUANT_FORMATS: dict[str, str] = {  # comfy_quant format string -> SDNQ weights_dtype
    "int8_tensorwise": "int8",
    "float8_e4m3fn": "float8_e4m3fn",
    "nvfp4": "float4_e2m1fn",
}
NVFP4_GROUP_SIZE = 16  # fixed by the format definition; markers restate it


class OverrideArchMismatch(Exception):
    """Raised when a user-selected UNET/DiT override cannot be loaded as the
    spec's transformer class: the arch-specific converter rejects its keys, or
    ``load_state_dict`` reports unexpected or missing keys.

    :func:`pipelines.generic_transformer.load_transformer` catches this to drop
    the override and load the base repo transformer instead, so a stale or
    wrong-arch UNET selection degrades to the base model rather than crashing
    inside a converter.

    A tensor shape mismatch on otherwise-matching keys is not raised as this; it
    surfaces as the native ``load_state_dict`` error, which already names the
    conflicting shapes, so it stays a hard load error rather than a fall back.
    """


@dataclass(frozen=True)
class SiblingSpec:
    """Describes a non-transformer component that may ship inline in the same
    safetensors as the transformer (e.g. Anima's ``llm_adapter``).

    ``subfolder`` names the base repo subfolder holding the canonical config and
    weights when the sibling is NOT bundled inline; ``inline_prefix`` is the
    key prefix that identifies the sibling's weights within the bundled file
    (after the transformer's prefix has already been stripped).
    """

    subfolder: str
    inline_prefix: str
    acceptable_missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransformerSpec:
    """Per-arch configuration for the native loader.

    Most arches only need to override ``cls`` (and rely on the default
    prefixes, no converter, no siblings). Arches with bundled-sibling
    components (Anima) or unusual key conventions (custom converters,
    Cosmos-style structural markers) override the relevant fields.

    ``zero_init_missing`` lists key prefixes for a zero-initialized residual
    branch the class defines but some checkpoints omit. Unlike
    ``acceptable_missing`` (buffer-only keys left at their init), these are
    zero-filled on load so the branch stays a no-op, matching a base model
    that ships the branch dormant (output projection all-zeros).

    ``ignored_prefixes`` names companion component families (text encoder,
    VAE) that all-in-one exports bundle alongside the transformer; their keys
    are dropped before prefix detection rather than treated as a malformed
    file.

    ``converter_handles_quant`` runs the converter before comfy_quant
    detection; such converters must translate marker/scale sidecar keys along
    with the weights. Float-oriented converters keep the default.
    """

    cls: type
    subfolder: str = "transformer"
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES
    ignored_prefixes: tuple[str, ...] = DEFAULT_IGNORED_PREFIXES
    converter: Callable[[dict], dict] | None = None
    converter_handles_quant: bool = False
    siblings: dict[str, SiblingSpec] = field(default_factory=dict)
    acceptable_missing: tuple[str, ...] = DEFAULT_ACCEPTABLE_MISSING
    zero_init_missing: tuple[str, ...] = ()
    forbidden_markers: tuple[tuple[str, str], ...] = ()


def make_default_spec(cls: type) -> TransformerSpec:
    """Synthesize a default spec for ``cls``: default prefixes, no siblings,
    no forbidden markers, and a converter picked up automatically from
    diffusers' ``SINGLE_FILE_LOADABLE_CLASSES`` table if one exists (and is
    not the no-op identity lambda that ``QwenImageTransformer2DModel`` and
    a few other classes register).

    Used by callers (notably :func:`pipelines.generic.load_transformer`) when
    a pipeline does not supply a custom ``TransformerSpec`` of its own.
    """
    return TransformerSpec(cls=cls, converter=auto_pickup_converter(cls))


def auto_pickup_converter(cls: type) -> Callable[[dict], dict] | None:
    """Pull a checkpoint converter from diffusers for ``cls`` when one exists.

    Skipped for the no-op identity lambda some classes register
    (notably ``QwenImageTransformer2DModel``), because using it would silently
    accept whatever key naming the file happens to have.
    """
    try:
        from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES
    except ImportError:
        return None
    entry = SINGLE_FILE_LOADABLE_CLASSES.get(cls.__name__)
    if entry is None:
        return None
    fn = entry.get("checkpoint_mapping_fn")
    if fn is None or is_noop_converter(fn):
        return None
    return fn


def is_noop_converter(fn: Callable) -> bool:
    """Detect ``lambda checkpoint, **kwargs: checkpoint`` and equivalents.

    Strips inline ``#`` comments from the source line before inspecting the
    body, so that diagnostic markers like ``# noqa`` on the lambda's source
    line do not defeat the detection.
    """
    try:
        import inspect
        src = inspect.getsource(fn).strip()
    except (OSError, TypeError):
        return False
    if "lambda" not in src:
        return False
    if "#" in src:
        src = src.split("#", 1)[0].rstrip()
    body = src.split(":", 1)[-1].strip().rstrip(",").rstrip(")")
    return body.endswith("checkpoint")


def resolve_path() -> str | None:
    """Return the absolute path of the UNET dropdown selection, or None if
    no selection is active or the file is unresolvable.
    """
    sel = shared.opts.sd_unet
    if sel is None or sel in ("Default", "None"):
        return None
    from modules import sd_unet
    if sel not in list(sd_unet.unet_dict):
        log.error(f'Load module: type=transformer file="{sel}" not found')
        return None
    path = sd_unet.unet_dict[sel]
    if not os.path.exists(path):
        log.error(f'Load module: type=transformer path="{path}" does not exist')
        return None
    return path


def load(
    local_file: str,
    repo_id: str,
    spec: TransformerSpec,
    diffusers_cfg: dict | None = None,
    sibling_classes: dict[str, type] | None = None,
    *,
    allow_quant: bool = True,
    dtype=None,
    modules_to_not_convert: list | None = None,
    modules_dtype_dict: dict | None = None,
    quant_args: dict | None = None,
    quant_type: str | None = None,
    **kwargs,
) -> tuple[object, dict[str, object]]:
    """Load the transformer (and any bundled siblings) from ``local_file``.

    ``sibling_classes`` supplies the runtime class for each sibling named in
    ``spec.siblings``. Required when a sibling has a dynamic class (e.g.
    Anima's ``AnimaLLMAdapter`` is loaded from remote_code at runtime).
    Missing sibling classes raise ``ValueError`` if the corresponding sibling
    keys are present in the bundled file.

    Keyword-only arguments ``allow_quant``, ``dtype``, ``modules_to_not_convert``,
    and ``modules_dtype_dict`` mirror the corresponding kwargs of
    :func:`pipelines.generic.load_transformer` and are forwarded unchanged.
    ``quant_args`` and ``quant_type`` are precomputed by the caller; when
    ``None`` they are derived here via ``model_quant.get_dit_args``. Extra
    ``**kwargs`` reach the transformer's ``cls.from_config``; siblings do not
    receive them.

    Returns ``(transformer, siblings_dict)``. ``siblings_dict`` is keyed by
    sibling name and is empty for non-sibling specs, or for sibling specs
    whose keys are absent from the bundled file.
    """
    if diffusers_cfg is None:
        diffusers_cfg = {}
    if sibling_classes is None:
        sibling_classes = {}

    t0 = time.time()
    if not local_file.lower().endswith(".safetensors"):
        raise ValueError(
            f"Load model: type={spec.cls.__name__} custom transformer requires .safetensors, "
            f'got "{local_file}"'
        )

    if quant_args is None:
        _, quant_args = model_quant.get_dit_args(
            diffusers_cfg, module="Model", device_map=True,
            allow_quant=allow_quant,
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict,
        )
        quant_type = model_quant.get_quant_type(quant_args)

    state_dict = sd_models.read_state_dict(local_file, what="transformer")
    metadata_layers = read_quantization_metadata(local_file)
    state_dict = drop_companion_keys(state_dict, spec.ignored_prefixes, spec.cls.__name__)
    state_dict, detected_prefix = strip_prefix(state_dict, spec.prefixes, spec.cls.__name__)
    if metadata_layers and detected_prefix:
        # header metadata names mirror the file's tensor naming, so they carry the same prefix
        metadata_layers = {name[len(detected_prefix):] if name.startswith(detected_prefix) else name: meta for name, meta in metadata_layers.items()}
    check_forbidden_markers(state_dict, spec.forbidden_markers, spec.cls.__name__, local_file)
    transformer_sd, sibling_sds = partition_siblings(state_dict, spec.siblings)
    del state_dict

    sibling_counts = {name: len(sd) for name, sd in sibling_sds.items() if sd}
    prefix_disp = f'"{detected_prefix}"' if detected_prefix else "bare"
    log.info(
        f'cls={spec.cls.__name__} custom="{os.path.basename(local_file)}" '
        f"prefix={prefix_disp} transformer_keys={len(transformer_sd)} "
        f"siblings={sibling_counts or '{}'}"
    )

    effective_dtype = dtype if dtype is not None else devices.dtype
    transformer_cfg = fetch_component_config(repo_id, spec.subfolder)
    transformer = build_component(
        component_name="transformer",
        state_dict=transformer_sd,
        config=transformer_cfg,
        cls=spec.cls,
        converter=spec.converter,
        acceptable_missing=spec.acceptable_missing,
        zero_init_missing=spec.zero_init_missing,
        quant_args=quant_args,
        quant_type=quant_type,
        dtype=effective_dtype,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
        converter_handles_quant=spec.converter_handles_quant,
        metadata_layers=metadata_layers,
        **kwargs,
    )
    del transformer_sd
    devices.torch_gc()

    loaded_siblings: dict[str, object] = {}
    for name, sibling_sd in sibling_sds.items():
        if not sibling_sd:
            continue
        sibling_spec = spec.siblings[name]
        sibling_cls = sibling_classes.get(name)
        if sibling_cls is None:
            raise ValueError(
                f"Load model: type={spec.cls.__name__} bundled sibling '{name}' present in "
                f"file but no class was supplied via sibling_classes"
            )
        sibling_cfg = fetch_component_config(repo_id, sibling_spec.subfolder)
        loaded_siblings[name] = build_component(
            component_name=name,
            state_dict=sibling_sd,
            config=sibling_cfg,
            cls=sibling_cls,
            converter=None,
            acceptable_missing=sibling_spec.acceptable_missing,
            quant_args={},
            quant_type=None,
            dtype=effective_dtype,
        )

    sd_models.allow_post_quant = False
    devices.torch_gc()
    log.debug(f"Load model: type={spec.cls.__name__} native_transformer time={time.time() - t0:.2f}")
    return transformer, loaded_siblings


def drop_companion_keys(state_dict: dict, ignored_prefixes: tuple[str, ...], type_name: str) -> dict:
    """Remove keys belonging to known non-transformer component families.

    All-in-one exports bundle the text encoder and VAE alongside the
    transformer under LDM/ComfyUI-style family prefixes (``cond_stage_model.``,
    ``first_stage_model.``, ``text_encoders.``, ``vae.``). Only the transformer
    (plus any spec-declared inline siblings) is wanted here; sdnext sources the
    other components from the base repo or their own override dropdowns.
    Dropped families are logged so the skip is visible in the load log.

    Raises ValueError when nothing remains after filtering, meaning the
    selected file holds no transformer at all.
    """
    if not ignored_prefixes:
        return state_dict
    dropped: dict[str, int] = {}
    kept: dict = {}
    for key, value in state_dict.items():
        prefix = next((p for p in ignored_prefixes if key.startswith(p)), None)
        if prefix is None:
            kept[key] = value
        else:
            dropped[prefix] = dropped.get(prefix, 0) + 1
    if not dropped:
        return state_dict
    counts = " ".join(f"{prefix.rstrip('.')}={count}" for prefix, count in dropped.items())
    if not kept:
        raise ValueError(
            f"Load model: type={type_name} native_transformer has no transformer keys ({counts})"
        )
    log.info(f"Load model: type={type_name} native_transformer skipping bundled components: {counts}")
    return kept


def strip_prefix(state_dict: dict, prefixes: tuple[str, ...], type_name: str) -> tuple[dict, str]:
    """Detect and uniformly strip the most common known prefix from every key.

    Order matters: longer prefixes win over shorter ones with the same suffix
    (e.g. ``model.diffusion_model.`` beats ``diffusion_model.``). If some keys
    match the dominant prefix and others do not, raises ValueError because
    mixed prefixes indicate a malformed file rather than a recoverable export
    quirk.

    Returns the stripped state dict and the detected prefix (empty string when
    the keys are already bare) so the caller can report it in one line.
    """
    sorted_prefixes = sorted(prefixes, key=len, reverse=True)
    counts: dict[str, int] = {}
    seen = 0
    for key in state_dict:
        for prefix in sorted_prefixes:
            if key.startswith(prefix):
                counts[prefix] = counts.get(prefix, 0) + 1
                seen += 1
                break
    total = len(state_dict)
    if seen == 0:
        return state_dict, ""
    dominant = max(counts, key=counts.get)
    if counts[dominant] != total:
        raise ValueError(
            f"Load model: type={type_name} native_transformer has mixed prefixes "
            f"(total={total} {dominant}={counts[dominant]})"
        )
    offset = len(dominant)
    return {key[offset:]: value for key, value in state_dict.items()}, dominant


def check_forbidden_markers(
    state_dict: dict,
    forbidden_markers: tuple[tuple[str, str], ...],
    type_name: str,
    local_file: str,
) -> None:
    """Raise if any forbidden marker key is present in the state_dict.

    Catches structural mismatches that pass prefix detection but indicate the
    file is from an incompatible architecture variant (e.g. Cosmos 1.0 keys
    showing up in a Cosmos 2.0 loader path).
    """
    for marker, description in forbidden_markers:
        if marker in state_dict:
            raise ValueError(
                f"Load model: type={type_name} native_transformer rejects "
                f'"{os.path.basename(local_file)}" ({description}; marker key {marker!r})'
            )


def read_quantization_metadata(local_file: str) -> dict[str, dict] | None:
    """Read per-layer quantization info from the safetensors header metadata.

    Newer comfy_quant checkpoints record layer formats in the header
    ``__metadata__`` under ``_quantization_metadata`` (a JSON string with a
    ``layers`` map keyed by tensor naming) instead of per-layer marker
    tensors. Returns the layers map, or ``None`` when the header carries no
    such entry. Raises :class:`OverrideArchMismatch` when the entry is
    present but malformed, so the caller's base-repo fallback engages.
    """
    from modules.model_probe import read_safetensors_header
    try:
        header = read_safetensors_header(local_file)
    except Exception as e:
        log.debug(f'Load model: file="{local_file}" header metadata unreadable ({e})')
        return None
    meta = (header.get("__metadata__") or {}).get("_quantization_metadata")
    if meta is None:
        return None
    try:
        parsed = json.loads(meta) if isinstance(meta, str) else meta
        layers = parsed["layers"]
    except Exception as e:
        raise OverrideArchMismatch(
            f'Load model: file="{local_file}" native_transformer _quantization_metadata '
            f"is malformed ({type(e).__name__}: {e})"
        ) from e
    if not isinstance(layers, dict) or not all(isinstance(entry, dict) for entry in layers.values()):
        raise OverrideArchMismatch(
            f'Load model: file="{local_file}" native_transformer _quantization_metadata '
            f"layers map is malformed"
        )
    return layers


def transcode_quant_metadata(state_dict: dict, metadata_layers: dict[str, dict]) -> tuple[dict, str]:
    """Synthesize per-layer marker tensors from header quantization metadata.

    Header metadata and marker tensors describe the same per-layer format
    dicts; converging on markers lets detection, converters, and remapping
    handle both forms identically. Entries without a matching ``.weight``
    (companion components filtered earlier) are skipped; existing markers
    are overwritten by their header entry. Returns a new dict plus the
    detection source (``header`` or ``both``) for logging.
    """
    sd = dict(state_dict)
    had_markers = any(key.endswith(COMFY_QUANT_MARKER) for key in sd)
    count = 0
    for name, meta in metadata_layers.items():
        if f"{name}.weight" not in sd:
            continue
        sd[f"{name}{COMFY_QUANT_MARKER}"] = torch.tensor(list(json.dumps(meta).encode("utf-8")), dtype=torch.uint8)
        count += 1
    if count == 0:
        return state_dict, "markers"
    return sd, "both" if had_markers else "header"


def detect_comfy_quant(state_dict: dict, type_name: str) -> tuple[dict[str, dict], str] | None:
    """Detect ``comfy_quant`` pre-quantized layers in a state dict.

    ComfyUI-format quantized checkpoints mark each quantized layer with a
    ``<name>.comfy_quant`` uint8 tensor whose bytes are a JSON object naming
    the storage format, alongside a ``<name>.weight_scale`` tensor. Returns
    a mapping of marked module names to their parsed marker JSON plus the
    file's format string, or ``None`` when no markers are present. ConvRot
    markers (``convrot``/``convrot_groupsize``) are accepted per layer: the
    rotation is the same regular Hadamard SDNQ implements, so flagged layers
    map onto ``use_hadamard`` (regular Hadamards only exist for power-of-4
    group sizes). Raises :class:`OverrideArchMismatch` for
    malformed markers, unsupported formats or group sizes, or a file mixing
    formats across layers, so the caller's base-repo fallback engages.
    """
    marker_keys = [key for key in state_dict if key.endswith(COMFY_QUANT_MARKER)]
    if not marker_keys:
        return None
    marked: dict[str, dict] = {}
    formats: set[str] = set()
    for key in marker_keys:
        name = key[: -len(COMFY_QUANT_MARKER)]
        try:
            meta = json.loads(state_dict[key].cpu().numpy().tobytes())
            fmt = meta["format"]
        except Exception as e:
            raise OverrideArchMismatch(
                f"Load model: type={type_name} native_transformer comfy_quant marker "
                f"for {name!r} is malformed ({type(e).__name__}: {e})"
            ) from e
        if meta.get("convrot"):
            if fmt == "nvfp4":
                raise OverrideArchMismatch(
                    f"Load model: type={type_name} native_transformer comfy_quant layer "
                    f"{name!r} combines nvfp4 with convrot, which the format does not define"
                )
            group_size = int(meta.get("convrot_groupsize", 256))
            is_pow4 = group_size >= 4 and (group_size & (group_size - 1)) == 0 and (group_size.bit_length() & 1) == 1
            if not is_pow4:
                raise OverrideArchMismatch(
                    f"Load model: type={type_name} native_transformer comfy_quant layer "
                    f"{name!r} convrot_groupsize={group_size} is not a power of 4"
                )
        if fmt == "nvfp4" and int(meta.get("group_size", NVFP4_GROUP_SIZE)) != NVFP4_GROUP_SIZE:
            raise OverrideArchMismatch(
                f"Load model: type={type_name} native_transformer comfy_quant layer "
                f"{name!r} nvfp4 group_size={meta.get('group_size')} is not {NVFP4_GROUP_SIZE}"
            )
        marked[name] = meta
        formats.add(fmt)
    unsupported = sorted(formats - set(COMFY_QUANT_FORMATS))
    if unsupported:
        log.error(
            f'Load model: type={type_name} quant=comfy format={",".join(unsupported)} not supported '
            f'(supported: {",".join(COMFY_QUANT_FORMATS)})'
        )
        raise OverrideArchMismatch(
            f"Load model: type={type_name} native_transformer comfy_quant format "
            f"{', '.join(unsupported)} not supported"
        )
    if len(formats) > 1:
        raise OverrideArchMismatch(
            f"Load model: type={type_name} native_transformer comfy_quant mixes formats "
            f"across layers ({', '.join(sorted(formats))})"
        )
    return marked, next(iter(formats))


def remap_comfy_quant(state_dict: dict, marked_names: set[str], defer_scales: bool = False) -> dict:
    """Translate comfy_quant tensor naming to SDNQ naming.

    Renames ``<name>.weight_scale`` to ``<name>.scale`` (reshaped to 2-D,
    ``[]`` becomes ``[1, 1]``, since SDNQ transposes scales in place) and
    drops the ``<name>.comfy_quant`` markers plus optional
    ``<name>.input_scale`` activation-calibration sidecars (SDNQ re-derives
    activation scales dynamically). With ``defer_scales`` the scale rename is
    skipped: block-scaled formats (nvfp4) carry swizzled scale tensors whose
    transform needs the layer dimensions, so the prequantized builder handles
    them per layer. Returns a new dict; the input (which may be the cached
    state dict) is not mutated.
    """
    scale_suffix = ".weight_scale"
    input_scale_suffix = ".input_scale"
    remapped: dict = {}
    for key, value in state_dict.items():
        if key.endswith(COMFY_QUANT_MARKER) and key[: -len(COMFY_QUANT_MARKER)] in marked_names:
            continue
        if key.endswith(input_scale_suffix) and key[: -len(input_scale_suffix)] in marked_names:
            continue
        if not defer_scales and key.endswith(scale_suffix) and key[: -len(scale_suffix)] in marked_names:
            remapped[f"{key[: -len(scale_suffix)]}.scale"] = value.reshape(-1, 1)
            continue
        remapped[key] = value
    return remapped


def unswizzle_block_scales(scales: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    """Invert the cuBLAS 2D block-scaling factor layout back to row-major
    ``[rows, groups]``, dropping the tile alignment padding.

    Block-scaled comfy_quant checkpoints store per-group scales pre-tiled for
    the hardware kernels: the ``[roundup(rows, 128), roundup(groups, 4)]``
    grid is rearranged into 32x16 tiles as described in
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    row_blocks = -(rows // -128)
    col_blocks = -(groups // -4)
    tiles = scales.reshape(-1, 32, 4, 4).transpose(1, 2)
    tiles = tiles.reshape(row_blocks, col_blocks, 4, 32, 4).reshape(row_blocks, col_blocks, 128, 4)
    return tiles.permute(0, 2, 1, 3).reshape(row_blocks * 128, col_blocks * 4)[:rows, :groups]


def adopt_nvfp4_layer(sd: dict, name: str, linear: torch.nn.Linear, component_name: str, meta: dict) -> None:
    """Rewrite one nvfp4 layer's tensors in place into SDNQ layout.

    Slices tile-alignment padding off the packed weight, swaps the nibble
    order (the container packs the even element into the high nibble, SDNQ
    unpacks low-first), unswizzles the e4m3 block scales, and folds the fp32
    global scale into them as ``[out, groups, 1]`` fp32 grouped scales.
    """
    out_features, in_features = linear.out_features, linear.in_features
    if in_features % NVFP4_GROUP_SIZE != 0:
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} comfy_quant marked module "
            f"{name!r} nvfp4 group size {NVFP4_GROUP_SIZE} does not divide in_features={in_features}"
        )
    orig_shape = meta.get("orig_shape")
    if orig_shape is not None and tuple(orig_shape) != (out_features, in_features):
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} comfy_quant marked module "
            f"{name!r} stores orig_shape={tuple(orig_shape)} but the model expects {(out_features, in_features)}"
        )
    weight = sd.get(f"{name}.weight")
    scales = sd.pop(f"{name}.weight_scale", None)
    global_scale = sd.pop(f"{name}.weight_scale_2", None)
    if scales is None or global_scale is None:
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} comfy_quant marked module "
            f"{name!r} is missing nvfp4 weight_scale/weight_scale_2 tensors"
        )
    packed_columns = in_features // 2
    if weight is None or weight.ndim != 2 or weight.shape[0] < out_features or weight.shape[1] < packed_columns:
        found = tuple(weight.shape) if weight is not None else "missing"
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} comfy_quant marked module "
            f"{name!r} packed weight {found} cannot hold {(out_features, packed_columns)}"
        )
    groups = in_features // NVFP4_GROUP_SIZE
    expected_scales = -(out_features // -128) * 128 * -(groups // -4) * 4
    if scales.numel() != expected_scales:
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} comfy_quant marked module "
            f"{name!r} block scales hold {scales.numel()} values, expected {expected_scales}"
        )
    weight = weight[:out_features, :packed_columns]
    weight = torch.bitwise_or(torch.bitwise_left_shift(torch.bitwise_and(weight, 15), 4), torch.bitwise_right_shift(weight, 4))
    scales = unswizzle_block_scales(scales, out_features, groups)
    sd[f"{name}.weight"] = weight
    sd[f"{name}.scale"] = scales.to(torch.float32).mul_(global_scale.to(torch.float32)).unsqueeze(-1)


def partition_siblings(
    state_dict: dict,
    siblings: dict[str, SiblingSpec],
) -> tuple[dict, dict[str, dict]]:
    """Split state_dict into (transformer_sd, {sibling_name: sibling_sd}).

    Keys matching a sibling's ``inline_prefix`` go into that sibling's dict
    with the prefix stripped; everything else stays in the transformer dict.
    Sibling names with no matching keys still appear in the output dict but
    map to an empty dict, so the caller can iterate uniformly.
    """
    sibling_sds: dict[str, dict] = {name: {} for name in siblings}
    transformer_sd: dict = {}
    if not siblings:
        return state_dict, sibling_sds
    sibling_lookups = [(name, siblings[name].inline_prefix) for name in siblings]
    for key, value in state_dict.items():
        matched = False
        for name, prefix in sibling_lookups:
            if key.startswith(prefix):
                sibling_sds[name][key[len(prefix):]] = value
                matched = True
                break
        if not matched:
            transformer_sd[key] = value
    return transformer_sd, sibling_sds


def fetch_component_config(repo_id: str, subfolder: str) -> dict:
    """Download and parse ``<subfolder>/config.json`` from the base repo."""
    relative_path = f"{subfolder}/config.json"
    try:
        local = hf.hf_hub_download(repo_id, filename=relative_path, cache_dir=shared.opts.diffusers_dir)
    except Exception as e:
        log.error(f' path="{relative_path}" repo="{repo_id}" failed to download: {e}')
        raise RuntimeError('') from e
    return shared.readfile(local, as_type="dict")


def build_component_quantized(
    *,
    component_name: str,
    state_dict: dict,
    config: dict,
    cls: type,
    quant_args: dict,
    dtype,
    acceptable_missing: tuple[str, ...],
    zero_init_missing: tuple[str, ...] = (),
    **kwargs,
) -> object:
    """Build a component with per-tensor SDNQ quantization during load.

    Mirrors the per-tensor loop in
    :func:`diffusers.models.model_loading_utils.load_model_dict_into_meta`:
    iterates the (already-converted) state_dict and dispatches each tensor
    through ``SDNQQuantizer.check_if_quantized_param`` /
    ``create_quantized_param`` so Linear/Conv/Embed weights are packed to
    uint4 as they land, while biases, LayerNorms, and other non-quantizable
    parameters go through ``accelerate.set_module_tensor_to_device``.

    The component is constructed inside ``init_empty_weights(include_buffers=
    False)`` so parameter slots are meta tensors (no full bf16 instantiation
    upfront) while computed buffers like ``rope.freqs`` materialize normally
    during ``cls.from_config(config)``. After the loop, the quantizer's
    ``_process_model_after_weight_loading`` hook attaches
    ``quantization_config`` and handles the CPU offload move.

    Caller is responsible for running the converter and prefix stripping
    before passing ``state_dict``.
    """
    import rich.progress as rp
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from modules.sdnq.quantizer import SDNQQuantizer

    quantization_config = quant_args.get("quantization_config")
    if quantization_config is None:
        raise ValueError(
            f"Load model: transformer=native {component_name} "
            f"per-tensor quantization requires quant_args['quantization_config']"
        )

    target_dtype = dtype if dtype is not None else devices.dtype
    quantizer = SDNQQuantizer(quantization_config, pre_quantized=False)
    quantizer.torch_dtype = target_dtype

    with init_empty_weights(include_buffers=False):
        component = cls.from_config(config, **kwargs)

    quantizer._process_model_before_weight_loading(component, device_map=None)  # pylint: disable=protected-access

    target_device = (
        devices.cpu if shared.opts.diffusers_offload_mode != "none"
        else devices.device
    )

    expected_keys = set(component.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected: list[str] = []
    total = len(state_dict)

    pbar = rp.Progress(
        rp.TextColumn(f'[cyan]Load {component_name}:'),
        rp.BarColumn(),
        rp.MofNCompleteColumn(),
        rp.TaskProgressColumn(),
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
        rp.TextColumn('[cyan]{task.description}'),
        console=console,
    )
    with pbar:
        task = pbar.add_task(total=total, description=cls.__name__)
        for name, value in state_dict.items():
            if name in expected_keys:
                if torch.is_floating_point(value):
                    value = value.to(target_dtype)
                if quantizer.check_if_quantized_param(component, value, name):
                    quantizer.create_quantized_param(component, value, name, target_device, dtype=target_dtype)
                else:
                    set_module_tensor_to_device(component, name, target_device, value=value, dtype=target_dtype)
                loaded_keys.add(name)
            else:
                unexpected.append(name)
            pbar.update(task, advance=1)

    missing = sorted(expected_keys - loaded_keys)
    missing = materialize_zero_init(
        component, missing, zero_init_missing, device=target_device, dtype=target_dtype
    )
    validate_state_dict_load(component_name, missing, unexpected, acceptable_missing)

    component = quantizer._process_model_after_weight_loading(component)  # pylint: disable=protected-access
    return component


def build_component_prequantized(
    *,
    component_name: str,
    state_dict: dict,
    config: dict,
    cls: type,
    marker_meta: dict[str, dict],
    comfy_format: str,
    dtype,
    acceptable_missing: tuple[str, ...],
    zero_init_missing: tuple[str, ...] = (),
    **kwargs,
) -> object:
    """Build a component from a comfy_quant pre-quantized state dict, mapping
    the marked layers onto SDNQ layers without dequantizing.

    The supported formats are subsets of SDNQ's symmetric quantization: same
    dequant math (``weight * scale``, no zero point), so tensors are adopted
    bit-exact. The 8-bit formats share the storage layout directly (unpacked
    ``[out, in]``); convrot layers map onto SDNQ's Hadamard support, whose
    identical regular Hadamard construction lets the dequantizer undo the
    stored rotation; nvfp4 layers keep their packed 4-bit codes (nibble order
    swapped once) and land on SDNQ's grouped quantization with the block and
    global scales folded into fp32 per-group scales.
    The file dictates which layers are quantized, independent of the user's
    quantization settings, and floating-point SDNQ params are not cast to the
    target dtype (the fp32 scales must survive). Layers are assembled in
    canonical dequant layout; ``apply_sdnq_options_to_model`` then applies
    the user's quantized-matmul settings, matching
    ``modules.sdnq.loader.load_sdnq_model``.
    """
    import rich.progress as rp
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers.utils import get_module_from_name
    from modules.sdnq.common import dtype_dict, check_torch_compile, is_fp8_compile_supported
    from modules.sdnq.quantizer import SDNQConfig, SDNQQuantizer
    from modules.sdnq.dequantizer import SDNQDequantizer
    from modules.sdnq.layers import get_sdnq_wrapper_class
    from modules.sdnq.forward import get_forward_func
    from modules.sdnq.loader import apply_sdnq_options_to_model

    weights_dtype = COMFY_QUANT_FORMATS[comfy_format]
    matmul_dtype = "int8" if dtype_dict[weights_dtype]["is_integer"] else "float8_e4m3fn"
    storage_dtype = dtype_dict[weights_dtype]["storage_dtype"]
    target_dtype = dtype if dtype is not None else devices.dtype
    is_nvfp4 = comfy_format == "nvfp4"
    # on hardware where compiled graphs cannot touch e4m3 tensors, adopt fp8 weights
    # through the uint8-backed codec: same values (NaN codes decode as +/-480), and
    # compiled dequant beats the eager native-fp8 fallback by ~6x
    remap_fp8_storage = weights_dtype == "float8_e4m3fn" and check_torch_compile() and not is_fp8_compile_supported
    if remap_fp8_storage:
        weights_dtype = "float8_e4m3fn_sdnq"
    marked_names = set(marker_meta)
    sd = remap_comfy_quant(state_dict, marked_names, defer_scales=is_nvfp4)

    # Civitai relabels these containers freely; trust the marker only as far
    # as the stored tensors actually match it.
    for name in marked_names:
        weight = sd.get(f"{name}.weight")
        if weight is None or weight.dtype != storage_dtype:
            found = weight.dtype if weight is not None else "missing"
            raise OverrideArchMismatch(
                f"Load model: transformer=native {component_name} comfy_quant format "
                f"{comfy_format} expects {storage_dtype} weights but {name!r} has {found}"
            )
        if remap_fp8_storage:
            sd[f"{name}.weight"] = weight.view(torch.uint8)

    # layers flagged full_precision_matrix_mult stay on the dequant path even
    # when the user enables quantized matmul (exact-name match on the config list)
    full_precision_mm = [f"{name}.weight" for name in sorted(marked_names) if marker_meta[name].get("full_precision_matrix_mult")]
    quantization_config = SDNQConfig(
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=matmul_dtype,
        group_size=NVFP4_GROUP_SIZE if is_nvfp4 else -1,
        use_quantized_matmul=shared.opts.sdnq_use_quantized_matmul,
        dequantize_fp32=shared.opts.sdnq_dequantize_fp32,
        add_skip_keys=False,
        modules_to_not_convert=[],
        modules_to_not_use_matmul=full_precision_mm,
    )
    quantizer = SDNQQuantizer(quantization_config, pre_quantized=True)
    quantizer.torch_dtype = target_dtype

    with init_empty_weights(include_buffers=False):
        component = cls.from_config(config, **kwargs)

    dequant_forward = get_forward_func("Linear", matmul_dtype, False)
    for name in sorted(marked_names):
        try:
            parent, child = get_module_from_name(component, name)
            linear = getattr(parent, child)
        except (AttributeError, ValueError) as e:
            raise OverrideArchMismatch(
                f"Load model: transformer=native {component_name} comfy_quant marked "
                f"module {name!r} not found in {cls.__name__}"
            ) from e
        if not isinstance(linear, torch.nn.Linear):
            raise OverrideArchMismatch(
                f"Load model: transformer=native {component_name} comfy_quant marked "
                f"module {name!r} is {linear.__class__.__name__}, expected Linear"
            )
        # convrot weights are stored rotated by SDNQ's regular Hadamard; the dequantizer undoes it
        use_hadamard = bool(marker_meta[name].get("convrot"))
        hadamard_group_size = int(marker_meta[name].get("convrot_groupsize", 256)) if use_hadamard else 256
        if use_hadamard and linear.in_features % hadamard_group_size != 0:
            raise OverrideArchMismatch(
                f"Load model: transformer=native {component_name} comfy_quant marked "
                f"module {name!r} convrot_groupsize={hadamard_group_size} does not divide in_features={linear.in_features}"
            )
        if is_nvfp4:
            adopt_nvfp4_layer(sd, name, linear, component_name, marker_meta[name])
        layer_shape = torch.Size((linear.out_features, linear.in_features))
        linear.sdnq_dequantizer = SDNQDequantizer(
            result_dtype=target_dtype,
            result_shape=layer_shape if is_nvfp4 else None,
            original_shape=layer_shape,
            original_stride=(linear.in_features, 1),
            quantized_weight_shape=torch.Size((linear.out_features, linear.in_features // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)) if is_nvfp4 else layer_shape,
            weights_dtype=weights_dtype,
            quantized_matmul_dtype=matmul_dtype,
            hadamard_group_size=hadamard_group_size,
            group_size=NVFP4_GROUP_SIZE if is_nvfp4 else -1,
            svd_rank=32,
            svd_steps=8,
            use_quantized_matmul=False,
            re_quantize_for_matmul=is_nvfp4,
            use_stochastic_rounding=False,
            use_hadamard=use_hadamard,
            layer_class_name="Linear",
        )
        wrapped = get_sdnq_wrapper_class(linear, dequant_forward)
        wrapped.scale = torch.nn.Parameter(torch.empty((1, 1), dtype=torch.float32, device="meta"), requires_grad=False)
        wrapped.zero_point = None
        wrapped.svd_up = None
        wrapped.svd_down = None
        setattr(parent, child, wrapped)

    target_device = (
        devices.cpu if shared.opts.diffusers_offload_mode != "none"
        else devices.device
    )

    expected_keys = set(component.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected: list[str] = []
    total = len(sd)

    pbar = rp.Progress(
        rp.TextColumn(f'[cyan]Load {component_name}:'),
        rp.BarColumn(),
        rp.MofNCompleteColumn(),
        rp.TaskProgressColumn(),
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
        rp.TextColumn('[cyan]{task.description}'),
        console=console,
    )
    with pbar:
        task = pbar.add_task(total=total, description=cls.__name__)
        for name, value in sd.items():
            if name in expected_keys:
                if quantizer.check_if_quantized_param(component, value, name):
                    quantizer.create_quantized_param(component, value, name, target_device, dtype=target_dtype)
                else:
                    if torch.is_floating_point(value):
                        value = value.to(target_dtype)
                    set_module_tensor_to_device(component, name, target_device, value=value, dtype=target_dtype)
                loaded_keys.add(name)
            else:
                unexpected.append(name)
            pbar.update(task, advance=1)

    missing = sorted(expected_keys - loaded_keys)
    missing = materialize_zero_init(
        component, missing, zero_init_missing, device=target_device, dtype=target_dtype
    )
    validate_state_dict_load(component_name, missing, unexpected, acceptable_missing)

    component = quantizer._process_model_after_weight_loading(component)  # pylint: disable=protected-access
    component = apply_sdnq_options_to_model(
        component,
        dtype=target_dtype,
        dequantize_fp32=shared.opts.sdnq_dequantize_fp32,
        use_quantized_matmul=shared.opts.sdnq_use_quantized_matmul,
    )
    return component


def apply_converter(converter: Callable[[dict], dict], state_dict: dict, cls: type, component_name: str) -> dict:
    """Run a spec converter, wrapping any failure as
    :class:`OverrideArchMismatch` (with the original chained) so a wrong-arch
    file degrades to the base-repo fallback instead of a raw converter crash.
    """
    log.debug(f'Load model: transformer=native {component_name} converter={converter.__name__} keys={len(state_dict)}')
    try:
        return converter(state_dict)
    except Exception as e:
        raise OverrideArchMismatch(
            f"Load model: type={cls.__name__} native_transformer converter "
            f"{converter.__name__} rejected the override ({type(e).__name__}: {e}); "
            f"file does not look like a {cls.__name__} checkpoint"
        ) from e


def build_component(
    *,
    component_name: str,
    state_dict: dict,
    config: dict,
    cls: type,
    converter: Callable[[dict], dict] | None,
    acceptable_missing: tuple[str, ...],
    zero_init_missing: tuple[str, ...] = (),
    quant_args: dict,
    quant_type: str | None,
    dtype=None,
    modules_to_not_convert: list | None = None,
    modules_dtype_dict: dict | None = None,
    converter_handles_quant: bool = False,
    metadata_layers: dict[str, dict] | None = None,
    **kwargs,
) -> object:
    """Convert (if needed), instantiate, load weights, dtype-cast, quantize,
    and offload-place a single component. Raises on any hard failure.

    Transformer state dicts carrying ``comfy_quant`` markers dispatch
    to :func:`build_component_prequantized` (``quant_args`` are bypassed: the
    file is already quantized). ``metadata_layers`` (header-metadata quant
    info) is transcoded into markers first, so both container forms share
    one path; a ``converter_handles_quant`` converter runs
    before that detection, float-oriented converters after it. Under SDNQ the
    transformer uses the per-tensor pre-mode path in
    :func:`build_component_quantized` so quantization is applied in flight.
    All other cases (siblings, non-quantized loads, NVIDIAModelOptConfig,
    layerwise quant) go through the standard load_state_dict + post-quantize
    path.

    ``dtype`` overrides ``devices.dtype`` when supplied; otherwise the global
    default is used. ``modules_to_not_convert`` and ``modules_dtype_dict``
    are forwarded to :func:`apply_quant` for the post-mode path; pre-mode
    receives them via the SDNQConfig in ``quant_args``. Extra ``**kwargs``
    reach ``cls.from_config`` for both construction paths.
    """
    try:
        quant_source = "markers"
        if component_name == "transformer" and metadata_layers:
            state_dict, quant_source = transcode_quant_metadata(state_dict, metadata_layers)

        if converter is not None and converter_handles_quant and component_name == "transformer":
            state_dict = apply_converter(converter, state_dict, cls, component_name)
            converter = None  # consumed; must not run again on the non-comfy path below

        comfy_quant = detect_comfy_quant(state_dict, cls.__name__) if component_name == "transformer" else None
        if comfy_quant is not None:
            marker_meta, comfy_format = comfy_quant
            convrot_count = sum(1 for meta in marker_meta.values() if meta.get("convrot"))
            log.info(
                f'Load model: transformer=native {component_name} quant=comfy '
                f'format={comfy_format} layers={len(marker_meta)} convrot={convrot_count} source={quant_source} keys={len(state_dict)} cls={cls.__name__}'
            )
            component = build_component_prequantized(
                component_name=component_name,
                state_dict=state_dict,
                config=config,
                cls=cls,
                marker_meta=marker_meta,
                comfy_format=comfy_format,
                dtype=dtype,
                acceptable_missing=acceptable_missing,
                zero_init_missing=zero_init_missing,
                **kwargs,
            )
            devices.torch_gc()
            return component

        if converter is not None:
            sd = apply_converter(converter, state_dict, cls, component_name)
        else:
            sd = state_dict

        if component_name == "transformer" and quant_type == "SDNQConfig":
            component = build_component_quantized(
                component_name=component_name,
                state_dict=sd,
                config=config,
                cls=cls,
                quant_args=quant_args,
                dtype=dtype,
                acceptable_missing=acceptable_missing,
                zero_init_missing=zero_init_missing,
                **kwargs,
            )
            del sd
            devices.torch_gc()
            return component

        log.debug(f'Load model: transformer=native {component_name} loading keys={len(sd)} cls={cls.__name__}')
        component = cls.from_config(config, **kwargs)
        missing, unexpected = component.load_state_dict(sd, strict=False)
        missing = materialize_zero_init(component, missing, zero_init_missing)
        validate_state_dict_load(component_name, missing, unexpected, acceptable_missing)
        del sd
        devices.torch_gc()
        target_dtype = dtype if dtype is not None else devices.dtype
        log.debug(f'Load model: transformer=native {component_name} cast dtype={target_dtype}')
        component = component.to(dtype=target_dtype)
    except OverrideArchMismatch:
        raise
    except Exception as e:
        log.error(f"Load model: transformer=native {component_name} load failed: {e}")
        errors.display(e, "Load")
        raise

    if component_name == "transformer":
        apply_quant(
            component,
            quant_type,
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict,
        )

    if shared.opts.diffusers_offload_mode != "none":
        sd_models.move_model(component, devices.cpu)

    if not hasattr(component, "quantization_config"):
        if hasattr(component, "config") and hasattr(component.config, "quantization_config"):
            component.quantization_config = component.config.quantization_config
        elif quant_type is not None and quant_args.get("quantization_config") is not None:
            component.quantization_config = quant_args.get("quantization_config")
    return component


def validate_state_dict_load(
    component_name: str,
    missing: list[str],
    unexpected: list[str],
    acceptable_missing: tuple[str, ...],
) -> None:
    """Raise :class:`OverrideArchMismatch` if load_state_dict produced
    unexpected keys or non-acceptable missing keys, which means the override's
    weights do not fit the target class. Buffer-only missing keys matching the
    ``acceptable_missing`` prefix list are logged at debug level and ignored.
    """
    if unexpected:
        sample = ", ".join(unexpected[:5])
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} has {len(unexpected)} "
            f"unexpected keys (sample: {sample})"
        )
    hard_missing = [
        k for k in missing if not any(k.startswith(p) for p in acceptable_missing)
    ]
    if hard_missing:
        sample = ", ".join(hard_missing[:5])
        raise OverrideArchMismatch(
            f"Load model: transformer=native {component_name} missing "
            f"{len(hard_missing)} required keys (sample: {sample})"
        )
    if missing:
        log.debug(
            f"Load model: transformer=native {component_name} ignored "
            f"{len(missing)} buffer-only missing keys"
        )


def materialize_zero_init(
    component: object,
    missing: list[str],
    zero_init_missing: tuple[str, ...],
    *,
    device=None,
    dtype=None,
) -> list[str]:
    """Zero-fill missing weights that belong to a zero-initialized residual
    branch the class defines but the override file predates.

    A base model may ship a residual branch dormant: its output projection is
    all-zeros, so ``out + up(down(x))`` reduces to ``out`` and the branch is a
    no-op. A checkpoint made before the branch existed omits those keys. Left
    unmaterialized they keep random ``from_config`` init (non-quant path) or
    stay on the meta device (per-tensor quant path); either injects garbage
    into the output. Zero-filling reproduces the base's dormant behavior.

    Returns ``missing`` with the handled keys removed, so
    :func:`validate_state_dict_load` does not treat them as a hard mismatch.
    ``device``/``dtype`` override the target placement (the quant path passes
    the real device since its params are on meta); when omitted each key's own
    parameter device/dtype is used. Keys matching a prefix but absent from the
    component's parameters are left in ``missing`` untouched.
    """
    if not zero_init_missing:
        return missing
    from accelerate.utils import set_module_tensor_to_device
    params = dict(component.named_parameters())
    handled: list[str] = []
    remaining: list[str] = []
    for key in missing:
        param = params.get(key)
        if param is not None and any(key.startswith(p) for p in zero_init_missing):
            tgt_device = device if device is not None else param.device
            tgt_dtype = dtype if dtype is not None else param.dtype
            set_module_tensor_to_device(
                component, key, tgt_device,
                value=torch.zeros(param.shape, dtype=tgt_dtype), dtype=tgt_dtype,
            )
            handled.append(key)
        else:
            remaining.append(key)
    if handled:
        log.debug(
            f"Load model: transformer=native zero-init {len(handled)} dormant "
            f"residual key(s): {', '.join(handled)}"
        )
    return remaining


def apply_quant(
    transformer: object,
    quant_type: str | None,
    modules_to_not_convert: list | None = None,
    modules_dtype_dict: dict | None = None,
) -> None:
    """Apply post-load quantization to a fully-loaded transformer.

    Used as a fallback for cases that don't go through the per-tensor
    pre-mode path in :func:`build_component_quantized`: ``layerwise_quantization``
    (via ``do_post_load_quant``), and the no-quant case (no-op). SDNQ post
    mode also reaches this path because pre-mode dispatch only triggers
    when an SDNQConfig is present (which itself only happens under modes
    ``pre`` or ``auto``).

    ``modules_to_not_convert`` and ``modules_dtype_dict`` are forwarded
    to :func:`sdnq_quantize_model` so per-call skip lists set by the
    caller are honored.
    """
    if quant_type == "NVIDIAModelOptConfig":
        log.warning(
            "Load model: transformer=native quant=TRT not supported on native path, skipping"
        )
    elif quant_type == "SDNQConfig":
        model_quant.sdnq_quantize_model(
            transformer,
            op="transformer",
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict,
        )
    model_quant.do_post_load_quant(transformer, allow=False)
