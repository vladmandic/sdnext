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
COMFY_SUPPORTED_FORMATS: tuple[str, ...] = ("int8_tensorwise",)


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
    """

    cls: type
    subfolder: str = "transformer"
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES
    ignored_prefixes: tuple[str, ...] = DEFAULT_IGNORED_PREFIXES
    converter: Callable[[dict], dict] | None = None
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
    state_dict = drop_companion_keys(state_dict, spec.ignored_prefixes, spec.cls.__name__)
    state_dict, detected_prefix = strip_prefix(state_dict, spec.prefixes, spec.cls.__name__)
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


def detect_comfy_quant(state_dict: dict, type_name: str) -> tuple[set[str], str] | None:
    """Detect ComfyUI ``comfy_quant`` pre-quantized layers in a state dict.

    ComfyUI's quantized checkpoints mark each quantized layer with a
    ``<name>.comfy_quant`` uint8 tensor whose bytes are a JSON object naming
    the storage format, alongside a ``<name>.weight_scale`` tensor. Returns
    the set of marked module names and the format string, or ``None`` when no
    markers are present. Raises :class:`OverrideArchMismatch` for malformed
    markers or unsupported formats so the caller's base-repo fallback engages.
    """
    marker_keys = [key for key in state_dict if key.endswith(COMFY_QUANT_MARKER)]
    if not marker_keys:
        return None
    marked: set[str] = set()
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
        marked.add(name)
        formats.add(fmt)
    unsupported = sorted(formats - set(COMFY_SUPPORTED_FORMATS))
    if unsupported:
        log.error(
            f'Load model: type={type_name} quant=comfy format={",".join(unsupported)} not supported '
            f'(supported: {",".join(COMFY_SUPPORTED_FORMATS)})'
        )
        raise OverrideArchMismatch(
            f"Load model: type={type_name} native_transformer comfy_quant format "
            f"{', '.join(unsupported)} not supported"
        )
    return marked, next(iter(formats))


def remap_comfy_quant(state_dict: dict, marked_names: set[str]) -> dict:
    """Translate comfy_quant tensor naming to SDNQ naming.

    Renames ``<name>.weight_scale`` to ``<name>.scale`` (reshaped to 2-D,
    ``[]`` becomes ``[1, 1]``, since SDNQ transposes scales in place) and
    drops the ``<name>.comfy_quant`` markers. Returns a new dict; the input
    (which may be the cached state dict) is not mutated.
    """
    scale_suffix = ".weight_scale"
    remapped: dict = {}
    for key, value in state_dict.items():
        if key.endswith(COMFY_QUANT_MARKER) and key[: -len(COMFY_QUANT_MARKER)] in marked_names:
            continue
        if key.endswith(scale_suffix) and key[: -len(scale_suffix)] in marked_names:
            remapped[f"{key[: -len(scale_suffix)]}.scale"] = value.reshape(-1, 1)
            continue
        remapped[key] = value
    return remapped


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
    marked_names: set[str],
    dtype,
    acceptable_missing: tuple[str, ...],
    zero_init_missing: tuple[str, ...] = (),
    **kwargs,
) -> object:
    """Build a component from a comfy_quant pre-quantized state dict, mapping
    the marked layers onto SDNQ int8 layers without dequantizing.

    ComfyUI ``int8_tensorwise`` is a strict subset of SDNQ symmetric int8:
    same storage layout (unpacked int8 ``[out, in]``), same dequant math
    (``weight * scale``, no zero point), so the file's tensors are adopted
    bit-exact. The model is built under ``init_empty_weights`` and each marked
    Linear is swapped for an SDNQ wrapper with per-tensor dequant geometry
    (``group_size=-1``, scalar scale); the file dictates which layers are
    quantized, independent of the user's quantization settings. Weights load
    through ``SDNQQuantizer``'s pre-quantized path, which preserves the int8
    weight dtype and the fp32 scale (unlike :func:`build_component_quantized`,
    floating-point SDNQ params are deliberately not cast to the target dtype).

    Layers are assembled in canonical dequant layout first;
    ``apply_sdnq_options_to_model`` then enables quantized matmul per the
    user's settings (transposing eligible layers), matching the order used by
    ``modules.sdnq.loader.load_sdnq_model``.

    Caller is responsible for prefix stripping; the state dict arrives with
    its comfy marker keys intact and is remapped here.
    """
    import rich.progress as rp
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers.utils import get_module_from_name
    from modules.sdnq.quantizer import SDNQConfig, SDNQQuantizer
    from modules.sdnq.dequantizer import SDNQDequantizer
    from modules.sdnq.layers import get_sdnq_wrapper_class
    from modules.sdnq.forward import get_forward_func
    from modules.sdnq.loader import apply_sdnq_options_to_model

    target_dtype = dtype if dtype is not None else devices.dtype
    sd = remap_comfy_quant(state_dict, marked_names)

    quantization_config = SDNQConfig(
        weights_dtype="int8",
        quantized_matmul_dtype="int8",
        group_size=-1,
        use_quantized_matmul=shared.opts.sdnq_use_quantized_matmul,
        dequantize_fp32=shared.opts.sdnq_dequantize_fp32,
        add_skip_keys=False,
        modules_to_not_convert=[],
    )
    quantizer = SDNQQuantizer(quantization_config, pre_quantized=True)
    quantizer.torch_dtype = target_dtype

    with init_empty_weights(include_buffers=False):
        component = cls.from_config(config, **kwargs)

    dequant_forward = get_forward_func("Linear", "int8", False)
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
        linear.sdnq_dequantizer = SDNQDequantizer(
            result_dtype=target_dtype,
            result_shape=None,
            original_shape=torch.Size((linear.out_features, linear.in_features)),
            original_stride=(linear.in_features, 1),
            quantized_weight_shape=torch.Size((linear.out_features, linear.in_features)),
            weights_dtype="int8",
            quantized_matmul_dtype="int8",
            hadamard_group_size=256,
            group_size=-1,
            svd_rank=32,
            svd_steps=8,
            use_quantized_matmul=False,
            re_quantize_for_matmul=False,
            use_stochastic_rounding=False,
            use_hadamard=False,
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
    **kwargs,
) -> object:
    """Convert (if needed), instantiate, load weights, dtype-cast, quantize,
    and offload-place a single component. Raises on any hard failure.

    Transformer state dicts carrying ComfyUI ``comfy_quant`` markers are
    dispatched to :func:`build_component_prequantized`, which adopts the
    file's int8 tensors as SDNQ layers regardless of quantization settings
    (the converter and ``quant_args`` are bypassed: the file is already
    quantized). For the transformer component under SDNQ, the per-tensor
    pre-mode path in :func:`build_component_quantized` is used so
    quantization is applied in flight (one layer's worth of bf16 in memory
    at a time). All other cases (siblings, non-quantized loads,
    NVIDIAModelOptConfig, layerwise quant) go through the standard
    load_state_dict + post-quantize path.

    ``dtype`` overrides ``devices.dtype`` when supplied; otherwise the global
    default is used. ``modules_to_not_convert`` and ``modules_dtype_dict``
    are forwarded to :func:`apply_quant` for the post-mode path; pre-mode
    receives them via the SDNQConfig in ``quant_args``. Extra ``**kwargs``
    reach ``cls.from_config`` for both construction paths.
    """
    try:
        comfy_quant = detect_comfy_quant(state_dict, cls.__name__) if component_name == "transformer" else None
        if comfy_quant is not None:
            marked_names, comfy_format = comfy_quant
            log.info(
                f'Load model: transformer=native {component_name} quant=comfy '
                f'format={comfy_format} layers={len(marked_names)} keys={len(state_dict)} cls={cls.__name__}'
            )
            component = build_component_prequantized(
                component_name=component_name,
                state_dict=state_dict,
                config=config,
                cls=cls,
                marked_names=marked_names,
                dtype=dtype,
                acceptable_missing=acceptable_missing,
                zero_init_missing=zero_init_missing,
                **kwargs,
            )
            devices.torch_gc()
            return component

        if converter is not None:
            log.debug(f'Load model: transformer=native {component_name} converter={converter.__name__} keys={len(state_dict)}')
            try:
                sd = converter(state_dict)
            except Exception as e:
                raise OverrideArchMismatch(
                    f"Load model: type={cls.__name__} native_transformer converter "
                    f"{converter.__name__} rejected the override ({type(e).__name__}: {e}); "
                    f"file does not look like a {cls.__name__} checkpoint"
                ) from e
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
