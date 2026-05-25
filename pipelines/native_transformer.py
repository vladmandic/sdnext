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
2. Detect and strip one of the spec's known prefixes (raises on mixed prefixes).
3. Check forbidden markers (catches structural mismatches like Cosmos 1.0 keys
   in a Cosmos 2.0 loader).
4. Partition off sibling component keys (e.g. Anima's bundled ``llm_adapter.*``).
5. Run the spec's converter if present (else pass through unchanged).
6. Fetch ``<subfolder>/config.json`` from the base repo, instantiate via
   ``cls.from_config``, ``load_state_dict(strict=False)``, validate, dtype-cast,
   quantize, and offload-place.
7. Repeat the build for each populated sibling (no converter, no quant by
   default; sibling weights are read raw from the bundled file).

Returns ``(transformer, sibling_components_dict)``. The dict is empty for
arches with no siblings; for Anima it carries the ``llm_adapter`` if the
community file bundled one, else the empty dict and the caller falls back to
loading the adapter from the base repo.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import huggingface_hub as hf

from modules import shared, devices, sd_models, model_quant, errors
from modules.logger import log


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
    """

    cls: type
    subfolder: str = "transformer"
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES
    converter: Callable[[dict], dict] | None = None
    siblings: dict[str, SiblingSpec] = field(default_factory=dict)
    acceptable_missing: tuple[str, ...] = DEFAULT_ACCEPTABLE_MISSING
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
) -> tuple[object, dict[str, object]]:
    """Load the transformer (and any bundled siblings) from ``local_file``.

    ``sibling_classes`` supplies the runtime class for each sibling named in
    ``spec.siblings``. Required when a sibling has a dynamic class (e.g.
    Anima's ``AnimaLLMAdapter`` is loaded from remote_code at runtime).
    Missing sibling classes raise ``ValueError`` if the corresponding sibling
    keys are present in the bundled file.

    Keyword-only arguments ``allow_quant``, ``dtype``, ``modules_to_not_convert``,
    and ``modules_dtype_dict`` mirror the corresponding kwargs of
    :func:`pipelines.generic.load_transformer` so the dispatch from there can
    plumb the caller's intent through unchanged.

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

    _, quant_args = model_quant.get_dit_args(
        diffusers_cfg, module="Model", device_map=True,
        allow_quant=allow_quant,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
    )
    quant_type = model_quant.get_quant_type(quant_args)

    state_dict = sd_models.read_state_dict(local_file, what="transformer")
    state_dict = strip_prefix(state_dict, spec.prefixes, spec.cls.__name__)
    check_forbidden_markers(state_dict, spec.forbidden_markers, spec.cls.__name__, local_file)
    transformer_sd, sibling_sds = partition_siblings(state_dict, spec.siblings)
    del state_dict

    sibling_counts = {name: len(sd) for name, sd in sibling_sds.items() if sd}
    log.info(
        f'Load model: type={spec.cls.__name__} custom="{os.path.basename(local_file)}" '
        f"transformer_keys={len(transformer_sd)} siblings={sibling_counts or '{}'}"
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
        quant_args=quant_args,
        quant_type=quant_type,
        dtype=effective_dtype,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
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


def strip_prefix(state_dict: dict, prefixes: tuple[str, ...], type_name: str) -> dict:
    """Detect and uniformly strip the most common known prefix from every key.

    Order matters: longer prefixes win over shorter ones with the same suffix
    (e.g. ``model.diffusion_model.`` beats ``diffusion_model.``). If some keys
    match the dominant prefix and others do not, raises ValueError because
    mixed prefixes indicate a malformed file rather than a recoverable export
    quirk.
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
        log.debug(f"Load model: type={type_name} native_transformer prefix=bare")
        return state_dict
    dominant = max(counts, key=counts.get)
    if counts[dominant] != total:
        raise ValueError(
            f"Load model: type={type_name} native_transformer has mixed prefixes "
            f"(total={total} {dominant}={counts[dominant]})"
        )
    log.debug(f'Load model: type={type_name} native_transformer prefix="{dominant}"')
    offset = len(dominant)
    return {key[offset:]: value for key, value in state_dict.items()}


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
        local = hf.hf_hub_download(
            repo_id, filename=relative_path, cache_dir=shared.opts.diffusers_dir,
        )
    except Exception as e:
        raise RuntimeError(
            f'Load model: native_transformer failed to download {relative_path} '
            f'from repo="{repo_id}": {e}'
        ) from e
    return shared.readfile(local, as_type="dict")


def build_component(
    *,
    component_name: str,
    state_dict: dict,
    config: dict,
    cls: type,
    converter: Callable[[dict], dict] | None,
    acceptable_missing: tuple[str, ...],
    quant_args: dict,
    quant_type: str | None,
    dtype=None,
    modules_to_not_convert: list | None = None,
    modules_dtype_dict: dict | None = None,
) -> object:
    """Convert (if needed), instantiate, load weights, dtype-cast, quantize,
    and offload-place a single component. Raises on any hard failure.

    ``dtype`` overrides ``devices.dtype`` when supplied; otherwise the global
    default is used. ``modules_to_not_convert`` and ``modules_dtype_dict``
    are forwarded to :func:`apply_quant` for the transformer component
    (ignored for siblings).
    """
    try:
        sd = converter(state_dict) if converter is not None else state_dict
        component = cls.from_config(config)
        missing, unexpected = component.load_state_dict(sd, strict=False)
        validate_state_dict_load(component_name, missing, unexpected, acceptable_missing)
        del sd
        devices.torch_gc()
        component = component.to(dtype=dtype if dtype is not None else devices.dtype)
    except Exception as e:
        log.error(f"Load model: native_transformer {component_name} load failed: {e}")
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
    """Raise ValueError if load_state_dict produced unexpected keys or
    non-acceptable missing keys. Buffer-only missing keys matching the
    ``acceptable_missing`` prefix list are logged at debug level and ignored.
    """
    if unexpected:
        sample = ", ".join(unexpected[:5])
        raise ValueError(
            f"Load model: native_transformer {component_name} has {len(unexpected)} "
            f"unexpected keys (sample: {sample})"
        )
    hard_missing = [
        k for k in missing if not any(k.startswith(p) for p in acceptable_missing)
    ]
    if hard_missing:
        sample = ", ".join(hard_missing[:5])
        raise ValueError(
            f"Load model: native_transformer {component_name} missing "
            f"{len(hard_missing)} required keys (sample: {sample})"
        )
    if missing:
        log.debug(
            f"Load model: native_transformer {component_name} ignored "
            f"{len(missing)} buffer-only missing keys"
        )


def apply_quant(
    transformer: object,
    quant_type: str | None,
    modules_to_not_convert: list | None = None,
    modules_dtype_dict: dict | None = None,
) -> None:
    """Apply SDNQ / layerwise quantization to the bare transformer.

    SDNQ 'pre' and 'auto' would normally route through ``quantization_config``
    at ``from_pretrained`` time; the native path bypasses that boundary, so
    we call the per-module quant path directly. SDNQ 'post' and
    ``layerwise_quantization`` go through ``do_post_load_quant`` as usual.
    NVIDIAModelOptConfig (TRT) is not supported on this path.

    ``modules_to_not_convert`` and ``modules_dtype_dict`` are forwarded
    to :func:`sdnq_quantize_model` so per-call skip lists set by the
    caller are honored on the native path.
    """
    if quant_type == "NVIDIAModelOptConfig":
        log.warning(
            "Load model: native_transformer quant=TRT not supported on native path, skipping"
        )
    elif quant_type == "SDNQConfig":
        if shared.opts.sdnq_quantize_mode == "pre":
            log.info(
                "Load model: native_transformer quant=SDNQ pre-mode applied post-load "
                "on native path"
            )
        model_quant.sdnq_quantize_model(
            transformer,
            op="transformer",
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict,
        )
    model_quant.do_post_load_quant(transformer, allow=False)
