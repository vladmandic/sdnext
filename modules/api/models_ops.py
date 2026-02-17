"""Model operations API — analyze, save, list, merge, replace, download, extract LoRA.

Provides REST endpoints for model management operations that were previously
only accessible through the Gradio UI.
"""

import os
import collections
import inspect
from modules import shared


# ---------------------------------------------------------------------------
# Phase 1 — Current model & list
# ---------------------------------------------------------------------------

def get_analyze():
    """
    Analyze the currently loaded model.

    Returns model name, type, class, hash, file size, metadata, and a breakdown
    of all sub-modules with their device, dtype, quantization, and parameter counts.
    """
    from modules import modelstats
    model = modelstats.analyze()
    if model is None:
        return {}
    return {
        "name": model.name,
        "type": model.type,
        "class": model.cls,
        "hash": model.hash or None,
        "size": model.size,
        "mtime": str(model.mtime) if model.mtime else None,
        "meta": model.meta or {},
        "modules": [
            {
                "name": m.name,
                "cls": m.cls,
                "device": str(m.device) if m.device else None,
                "dtype": str(m.dtype) if m.dtype else None,
                "quant": str(m.quant) if m.quant else None,
                "params": m.params,
                "modules": m.modules,
                "config": dict(m.config) if m.config and hasattr(m.config, 'items') else None,
            }
            for m in model.modules
        ],
    }


def post_save(name: str, path: str = None, shard: str = None, overwrite: bool = False):
    """
    Save the currently loaded model to disk.

    Saves the active pipeline under ``name``. Optional ``path`` overrides the output
    directory, ``shard`` sets the shard size, and ``overwrite`` allows replacing
    an existing file.
    """
    from modules import sd_models
    result = sd_models.save_model(name=name, path=path, shard=shard, overwrite=overwrite)
    return {"status": result}


def get_list_detail():
    """
    List all checkpoints with detection info.

    Returns every registered checkpoint with its filename, file type, auto-detected
    model type, matching pipeline class, hash, file size, and modification time.
    """
    from modules import sd_checkpoint, sd_detect, modelstats
    rows = []
    for ckpt in sd_checkpoint.checkpoints_list.values():
        try:
            f = ckpt.filename
            stat_size, stat_mtime = modelstats.stat(f)
            if os.path.isfile(f):
                typ = os.path.splitext(f)[1][1:]
            elif os.path.isdir(f):
                typ = "diffusers"
            else:
                typ = "unknown"
            guess = "Diffusion"
            guess = sd_detect.guess_by_size(f, guess)
            guess = sd_detect.guess_by_name(f, guess)
            guess, pipeline = sd_detect.guess_by_diffusers(f, guess)
            guess = sd_detect.guess_variant(f, guess)
            if pipeline is None:
                pipeline = sd_detect.shared_items.get_pipelines().get(guess, None)
            rows.append({
                "model_name": ckpt.model_name,
                "filename": ckpt.filename,
                "type": typ,
                "detected_type": guess,
                "pipeline": pipeline.__name__ if pipeline else None,
                "hash": ckpt.shorthash,
                "size": stat_size,
                "mtime": str(stat_mtime) if stat_mtime else None,
            })
        except Exception as e:
            shared.log.error(f"Model list-detail: {e}")
    return rows


def post_update_hashes():
    """
    Recalculate hashes for all registered checkpoints.

    Iterates over every checkpoint and recomputes its short hash. Returns a list
    of updated entries with name, type, and new hash.
    """
    from modules import sd_checkpoint
    updated = []
    for _html in sd_checkpoint.update_model_hashes():
        pass  # consume the generator
    for ckpt in sd_checkpoint.checkpoints_list.values():
        if ckpt.shorthash:
            updated.append({"name": ckpt.model_name, "type": ckpt.type, "hash": ckpt.shorthash})
    return {"updated": updated}


# ---------------------------------------------------------------------------
# Phase 2 — HuggingFace, CivitAI, Metadata
# ---------------------------------------------------------------------------

def get_hf_search(keyword: str = ""):
    """
    Search HuggingFace Hub for models.

    Returns matching models with their repo ID, pipeline tag, tags, download count,
    last modified date, and URL.
    """
    from modules import models_hf
    results = models_hf.hf_search(keyword)
    return [
        {"id": r[0], "pipeline_tag": r[1], "tags": r[2], "downloads": r[3], "last_modified": r[4], "url": r[5]}
        for r in results
    ]


def post_hf_download(hub_id: str, token: str = "", variant: str = "", revision: str = "", mirror: str = "", custom_pipeline: str = ""):
    """
    Download a model from HuggingFace Hub.

    Downloads the model identified by ``hub_id``. Optional parameters control
    authentication token, variant (e.g., fp16), revision, mirror URL, and custom pipeline.
    """
    from modules import models_hf
    result = models_hf.hf_download_model(hub_id, token, variant, revision, mirror, custom_pipeline)
    return {"status": result}


def post_civitai_download(url: str, name: str = "", path: str = "", model_type: str = "", token: str = None):
    """
    Download a model from CivitAI.

    Queues the download and returns the download ID for progress tracking via
    the WebSocket or ``GET /sdapi/v1/civitai/download/status``.
    """
    from modules.civitai.download_civitai import download_manager
    from modules.civitai.filemanage_civitai import get_type_folder
    if not url:
        return {"status": "Error: no url provided"}
    if not path:
        folder = str(get_type_folder(model_type or 'Checkpoint'))
    elif os.path.isabs(path):
        folder = path
    else:
        from modules import paths
        folder = os.path.join(paths.models_path, path)
    item = download_manager.enqueue(
        url=url,
        folder=folder,
        filename=name or "Unknown",
        model_type=model_type,
        token=token,
    )
    return {"status": "queued", "download_id": item.id, "url": url}


def post_metadata_scan():
    """
    Scan local models against CivitAI metadata.

    Checks all registered checkpoints for matching CivitAI entries and returns
    scan results with model ID, name, hash, versions, and status.
    """
    from modules.civitai import metadata_civitai
    results = []
    for batch in metadata_civitai.civit_search_metadata(raw=True):
        if isinstance(batch, list):
            results = batch
    return {"results": results}


def post_metadata_update():
    """
    Update local model metadata from CivitAI.

    Fetches the latest metadata from CivitAI for all matched models and updates
    local records. Returns per-model update results.
    """
    from modules.civitai import metadata_civitai
    items = []
    for batch in metadata_civitai.civit_update_metadata(raw=True):
        if isinstance(batch, list):
            items = batch
    results = []
    for item in items:
        results.append({
            "file": getattr(item, "file", None),
            "id": getattr(item, "id", None),
            "name": getattr(item, "name", None),
            "sha": getattr(item, "sha", None),
            "versions": getattr(item, "versions", None),
            "latest": getattr(item, "latest_name", None),
            "status": getattr(item, "status", None),
        })
    return {"results": results}


# ---------------------------------------------------------------------------
# Phase 3 — Merge & Replace
# ---------------------------------------------------------------------------

def get_merge_methods():
    """
    List available model merge methods and block-weight presets.

    Returns method names, which methods support beta/triple parameters,
    per-method documentation, and SD 1.5 / SDXL block-weight presets.
    """
    from modules.merging import merge_methods
    from modules.merging.merge_utils import BETA_METHODS, TRIPLE_METHODS
    from modules.merging.merge_presets import BLOCK_WEIGHTS_PRESETS, SDXL_BLOCK_WEIGHTS_PRESETS
    docs = {}
    for name in merge_methods.__all__:
        fn = getattr(merge_methods, name, None)
        docs[name] = (fn.__doc__ or "").strip() if fn else ""
    return {
        "methods": list(merge_methods.__all__),
        "beta_methods": list(BETA_METHODS),
        "triple_methods": list(TRIPLE_METHODS),
        "docs": docs,
        "presets": dict(BLOCK_WEIGHTS_PRESETS),
        "sdxl_presets": dict(SDXL_BLOCK_WEIGHTS_PRESETS),
    }


def post_merge(
    custom_name: str,
    primary_model_name: str,
    secondary_model_name: str,
    merge_mode: str,
    tertiary_model_name: str = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    alpha_preset: str = None,
    alpha_preset_lambda: float = None,
    alpha_base: str = None,
    alpha_in_blocks: str = None,
    alpha_mid_block: str = None,
    alpha_out_blocks: str = None,
    beta_preset: str = None,
    beta_preset_lambda: float = None,
    beta_base: str = None,
    beta_in_blocks: str = None,
    beta_mid_block: str = None,
    beta_out_blocks: str = None,
    precision: str = "fp16",
    checkpoint_format: str = "safetensors",
    save_metadata: bool = True,
    weights_clip: bool = False,
    prune: bool = False,
    re_basin: bool = False,
    re_basin_iterations: int = 0,
    device: str = "cpu",
    unload: bool = True,
    overwrite: bool = False,
    bake_in_vae: str = None,
):
    """
    Merge two or three checkpoint models.

    Combines ``primary_model_name`` and ``secondary_model_name`` using the specified
    ``merge_mode``. Supports block-weight presets, precision control, re-basin
    alignment, and optional VAE bake-in. Saves the result as ``custom_name``.
    """
    from modules import extras, sd_models, errors
    kwargs = {k: v for k, v in locals().items() if v not in [None, "None", "", 0, []]}
    if not custom_name:
        return {"status": "Error: no output model name specified"}
    if not primary_model_name or not secondary_model_name:
        return {"status": "Error: primary and secondary models are required"}
    try:
        results = extras.run_modelmerger(None, **kwargs)
        status = results[-1] if isinstance(results, list) else str(results)
    except Exception as e:
        errors.display(e, "Merge")
        sd_models.list_models()
        status = f"Error merging: {e}"
    return {"status": status}


def post_replace(
    model_type: str,
    model_name: str,
    custom_name: str,
    comp_unet: str = "",
    comp_vae: str = "",
    comp_te1: str = "",
    comp_te2: str = "",
    precision: str = "fp16",
    comp_scheduler: str = "",
    comp_prediction: str = "",
    comp_lora: str = "",
    comp_fuse: float = 0.0,
    meta_author: str = "",
    meta_version: str = "",
    meta_license: str = "",
    meta_desc: str = "",
    meta_hint: str = "",
    create_diffusers: bool = True,
    create_safetensors: bool = False,
    debug: bool = False,
):
    """
    Replace model components and save as a new model.

    Swap UNET, VAE, text encoders, scheduler, or fuse a LoRA into the base model.
    Saves the result as ``custom_name`` in Diffusers and/or safetensors format
    with optional metadata (author, version, license, description).
    """
    from modules import extras
    status = "Unknown"
    for msg in extras.run_model_modules(
        model_type, model_name, custom_name,
        comp_unet, comp_vae, comp_te1, comp_te2,
        precision, comp_scheduler, comp_prediction,
        comp_lora, comp_fuse,
        meta_author, meta_version, meta_license, meta_desc, meta_hint,
        None,  # meta_thumbnail (PIL image — not applicable via API)
        create_diffusers, create_safetensors, debug,
    ):
        status = msg
    return {"status": status}


# ---------------------------------------------------------------------------
# Phase 4 — Loader & Extract LoRA
# ---------------------------------------------------------------------------

def get_loader_pipelines():
    """
    List available pipeline types for the model loader.

    Returns pipeline class names (e.g., StableDiffusionPipeline, FluxPipeline)
    that can be used with the loader/components and loader/load endpoints.
    """
    from modules import shared_items
    names = list(shared_items.pipelines)
    names = ["Current" if x.startswith("Custom") else x for x in names]
    return {"pipelines": names}


def post_loader_components(model_type: str):
    """
    Inspect pipeline components for a given model type.

    Returns the pipeline class name, default HuggingFace repo, and a list of
    loadable components with their IDs, class names, local/remote paths,
    dtype, and quantization settings.
    """
    from modules import shared_items, ui_models_load
    import diffusers as _diffusers
    if model_type == "Current":
        cls = shared.sd_model.__class__ if shared.sd_loaded else None
    else:
        cls = shared_items.pipelines.get(model_type, None)
    if cls is None:
        cls = _diffusers.AutoPipelineForText2Image
    name = cls.__name__
    repo = shared_items.get_repo(name) or shared_items.get_repo(model_type)
    # Build components via signature inspection
    ui_models_load.components.clear()
    signature = inspect.signature(cls.__init__, follow_wrapped=True)
    for param in signature.parameters.values():
        if param.name in ("self", "args", "kwargs"):
            continue
        component = ui_models_load.Component(param)
        ui_models_load.components.append(component)
    result = []
    for c in ui_models_load.components:
        result.append({
            "id": c.id,
            "name": c.name,
            "loadable": c.loadable,
            "default": str(c.val) if c.val is not None else None,
            "class_name": c.str,
            "local": c.local,
            "remote": c.remote,
            "dtype": c.dtype,
            "quant": c.quant,
        })
    return {"class": name, "repo": repo, "components": result}


def post_loader_load(model_type: str, repo: str, components: list = None):
    """
    Load a model with custom component configuration.

    Loads the pipeline for ``model_type`` from ``repo``, optionally overriding
    individual component paths, dtypes, and quantization via the ``components`` list.
    Call loader/components first to discover available components.
    """
    from modules import ui_models_load
    cls_name = None
    # Ensure components are populated — call post_loader_components first
    if not ui_models_load.components:
        info = post_loader_components(model_type)
        cls_name = info.get("class")
    if cls_name is None:
        import diffusers as _diffusers
        from modules import shared_items
        if model_type == "Current":
            _cls = shared.sd_model.__class__ if shared.sd_loaded else None
        else:
            _cls = shared_items.pipelines.get(model_type, None)
        if _cls is None:
            _cls = _diffusers.AutoPipelineForText2Image
        cls_name = _cls.__name__
    # Update components from provided data
    if components:
        for comp_data in components:
            matches = [c for c in ui_models_load.components if c.id == comp_data.get("id")]
            if matches:
                c = matches[0]
                if "local" in comp_data:
                    c.local = (comp_data["local"] or "").strip()
                if "remote" in comp_data:
                    c.remote = (comp_data["remote"] or "").strip()
                    if c.remote:
                        c.repo, c.subfolder, c.local, c.download = ui_models_load.process_huggingface_url(c.remote)
                if "dtype" in comp_data:
                    c.dtype = comp_data["dtype"]
                if "quant" in comp_data:
                    c.quant = comp_data["quant"]
    dataframes = [c.dataframe() for c in ui_models_load.components]
    result = ui_models_load.load_model(model_type, cls_name, repo, dataframes)
    return {"status": result}


def get_lora_loaded():
    """
    List LoRAs currently loaded in the active pipeline.

    Returns names of all LoRA networks that are fused or applied to the current model.
    """
    from modules.lora import lora_extract
    result = lora_extract.loaded_lora()
    if isinstance(result, str):
        return {"loras": []}
    return {"loras": result}


def post_lora_extract(filename: str, max_rank: int = 64, auto_rank: bool = False, rank_ratio: float = 0.5, modules: list = None, overwrite: bool = False):
    """
    Extract a LoRA from the currently loaded model.

    Creates a LoRA file by comparing the current model weights against the base.
    ``max_rank`` controls decomposition rank, ``modules`` selects which parts
    to extract (defaults to ["te", "unet"]).
    """
    from modules.lora import lora_extract
    if modules is None:
        modules = ["te", "unet"]
    status = "Unknown"
    for msg in lora_extract.make_lora(filename, max_rank, auto_rank, rank_ratio, modules, overwrite):
        status = msg
    return {"status": status}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_api():
    api = shared.api
    # Phase 1
    api.add_api_route("/sdapi/v1/model/analyze", get_analyze, methods=["GET"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/save", post_save, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/list-detail", get_list_detail, methods=["GET"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/update-hashes", post_update_hashes, methods=["POST"], tags=["Models"])
    # Phase 2
    api.add_api_route("/sdapi/v1/model/hf/search", get_hf_search, methods=["GET"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/hf/download", post_hf_download, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/civitai/download", post_civitai_download, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/metadata/scan", post_metadata_scan, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/metadata/update", post_metadata_update, methods=["POST"], tags=["Models"])
    # Phase 3
    api.add_api_route("/sdapi/v1/model/merge/methods", get_merge_methods, methods=["GET"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/merge", post_merge, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/replace", post_replace, methods=["POST"], tags=["Models"])
    # Phase 4
    api.add_api_route("/sdapi/v1/model/loader/pipelines", get_loader_pipelines, methods=["GET"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/loader/components", post_loader_components, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/loader/load", post_loader_load, methods=["POST"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/lora/loaded", get_lora_loaded, methods=["GET"], tags=["Models"])
    api.add_api_route("/sdapi/v1/model/lora/extract", post_lora_extract, methods=["POST"], tags=["Models"])
