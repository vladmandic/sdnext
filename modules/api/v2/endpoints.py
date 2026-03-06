import asyncio
from datetime import datetime
from typing import Any
from fastapi import APIRouter, HTTPException, Query
from modules import shared
from modules.api.v2.models import (
    ItemExtraNetworkV2, ResExtraNetworksV2,
    ItemModelV2, ResModelsV2,
    ItemSamplerV2,
    ItemHistoryV2, ResHistoryV2,
    ResCheckpointV2, ReqSetCheckpointV2, ResSetCheckpointV2,
    ItemScriptV2, ResScriptsV2,
    ItemVaeV2, ItemUpscalerV2,
    ItemEmbeddingV2, ResEmbeddingsV2,
    ItemPromptStyleV2, ResRefreshNetworksV2,
    OptionUpdateItemV2, ResSetOptionsV2,
    ResOptionsInfoV2, ItemSecretStatusV2,
    ItemPreprocessorV2, ReqPreprocessV2, ResPreprocessV2,
    ResLogV2, ResLogClearV2,
    ReqPngInfoV2, ResPngInfoV2,
    ItemExtensionV2,
    ItemDetailerV2,
)

router = APIRouter(prefix="/sdapi/v2")


def _format_tags(raw_tags) -> list[str]:
    if isinstance(raw_tags, dict):
        return list(raw_tags.keys()) if raw_tags else []
    if isinstance(raw_tags, str) and raw_tags:
        return [t.strip() for t in raw_tags.split('|') if t.strip()]
    if isinstance(raw_tags, list):
        return raw_tags
    return []


def _format_mtime(mtime) -> str | None:
    if isinstance(mtime, datetime):
        return mtime.isoformat()
    if isinstance(mtime, (int, float)):
        return datetime.fromtimestamp(mtime).isoformat()
    if mtime is not None:
        return str(mtime)
    return None


# --- Extra Networks ---

@router.get("/extra-networks", response_model=ResExtraNetworksV2, tags=["Enumerators"])
async def get_extra_networks_v2(
    page: str | None = Query(default=None, description="Filter by page name (lora, model, embedding, etc.)"),
    search: str | None = Query(default=None, description="Case-insensitive search across name, title, filename, tags"),
    subfolder: str | None = Query(default=None, description="Filter by subfolder component in filename"),
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1, description="Max items to return (omit for all)"),
):
    """List extra networks with optional filtering and pagination."""
    matched = []
    lower_search = search.lower() if search else None
    for pg in shared.extra_networks:
        if page is not None and pg.name != page.lower():
            continue
        for item in pg.items:
            tags = _format_tags(item.get('tags', None))
            name = item.get('name', '')
            title = item.get('title', '') or ''
            filename = item.get('filename', '') or ''
            if lower_search:
                haystack = f"{name} {title} {filename} {' '.join(tags)}".lower()
                if lower_search not in haystack:
                    continue
            if subfolder and subfolder not in filename:
                continue
            mtime = _format_mtime(item.get('mtime', None))
            matched.append(ItemExtraNetworkV2(
                name=name,
                type=pg.name,
                title=item.get('title', None),
                fullname=item.get('fullname', None),
                filename=item.get('filename', None),
                hash=item.get('shorthash', None) or item.get('hash'),
                preview=item.get('preview', None),
                version=item.get('version', None),
                tags=tags,
                size=item.get('size', None),
                mtime=mtime,
            ))
    total = len(matched)
    page_items = matched[offset:offset + limit] if limit is not None else matched[offset:]
    return ResExtraNetworksV2(items=page_items, total=total, offset=offset, limit=limit or total)


# --- Options ---

@router.get("/options", tags=["Server"])
async def get_options_v2(
    keys: str | None = Query(default=None, description="Comma-separated list of keys to return"),
) -> dict[str, Any]:
    """Return server options, optionally filtered by key names."""
    from modules.api.server import get_config
    options = get_config()
    if keys:
        requested = [k.strip() for k in keys.split(',') if k.strip()]
        return {k: v for k, v in options.items() if k in requested}
    return options


@router.post("/options", response_model=ResSetOptionsV2, tags=["Server"])
async def set_options_v2(req: dict[str, Any]):
    """Update one or more application options and persist them to disk."""
    from modules.api.server import set_config
    result = await asyncio.to_thread(set_config, req)
    updated = []
    for item in result.get("updated", []):
        for k, v in item.items():
            updated.append(OptionUpdateItemV2(key=k, changed=bool(v)))
    return ResSetOptionsV2(ok=True, updated=updated)


@router.get("/options-info", response_model=ResOptionsInfoV2, tags=["Server"])
async def get_options_info_v2():
    """Return metadata for all application settings."""
    from modules.api.server import get_options_info
    return get_options_info()


@router.get("/secrets-status", response_model=dict[str, ItemSecretStatusV2], tags=["Server"])
async def get_secrets_status_v2():
    """Return configuration status for all secret options."""
    from modules.api.server import get_secrets_status
    return get_secrets_status()


# --- Control ---

@router.get("/control-models", response_model=list[str], tags=["Control"])
async def get_control_models_v2(
    unit_type: str = Query(default="controlnet", description="Unit type: controlnet, t2i, xs, lite, reference"),
):
    """List available models for a given control unit type."""
    from modules.api.endpoints import get_control_models
    return get_control_models(unit_type)


@router.get("/control-modes", response_model=dict[str, list[str]], tags=["Control"])
async def get_control_modes_v2():
    """List mode choices for control models that support modes."""
    from modules.api.endpoints import get_control_modes
    return get_control_modes()


@router.get("/ip-adapters", response_model=list[str], tags=["Control"])
async def get_ip_adapters_v2():
    """List available IP-Adapter models."""
    from modules.api.endpoints import get_ip_adapters
    return get_ip_adapters()


@router.get("/preprocessors", response_model=list[ItemPreprocessorV2], tags=["Control"])
async def get_preprocessors_v2():
    """List available image preprocessors with their configurable parameters."""
    from modules.control import processors
    return [
        ItemPreprocessorV2(name=k, group=v.get('group', 'Other'), params=v.get('params', {}))
        for k, v in processors.config.items()
    ]


@router.post("/preprocess", response_model=ResPreprocessV2, tags=["Control"])
async def post_preprocess_v2(req: ReqPreprocessV2):
    """Run an image preprocessor on the input image and return the processed result."""
    def _run():
        from modules.control import processors
        from modules.api.helpers import decode_base64_to_image, encode_pil_to_base64
        import modules.api.process as process_module
        processors_list = list(processors.config)
        if req.model not in processors_list:
            raise HTTPException(status_code=400, detail=f"Processor model not found: id={req.model}")
        image = decode_base64_to_image(req.image)
        proc = process_module.processor
        if proc is None or proc.processor_id != req.model:
            from modules.call_queue import queue_lock
            with queue_lock:
                proc = processors.Processor(req.model)
                process_module.processor = proc
        for k, v in req.params.items():
            if k not in processors.config[proc.processor_id]['params']:
                raise HTTPException(status_code=400, detail=f"Processor invalid parameter: id={req.model} {k}={v}")
        jobid = shared.state.begin('API-PRE', api=True)
        try:
            processed = proc(image, local_config=req.params)
            result_image = encode_pil_to_base64(processed)
        finally:
            shared.state.end(jobid)
        return ResPreprocessV2(ok=True, model=proc.processor_id, image=result_image)
    return await asyncio.to_thread(_run)


# --- SD Models ---

@router.get("/sd-models", response_model=ResModelsV2, tags=["Models"])
async def get_sd_models_v2(
    search: str | None = Query(default=None, description="Case-insensitive search on title/filename"),
    type: str | None = Query(default=None, description="Filter by model type", alias="type"),
    offset: int = Query(default=0, ge=0),
    limit: int | None = Query(default=None, ge=1, description="Max items to return (omit for all)"),
):
    """List registered checkpoint models with search and type filtering."""
    from modules import sd_checkpoint
    lower_search = search.lower() if search else None
    # build a name→item lookup from the extra_networks cache (already has size/mtime/version)
    model_items = {}
    for pg in shared.extra_networks:
        if pg.name == 'model':
            for item in pg.items:
                model_items[item.get('name', '')] = item
            break
    matched = []
    for v in sd_checkpoint.checkpoints_list.values():
        if lower_search:
            if lower_search not in (v.title or '').lower() and lower_search not in (v.filename or '').lower():
                continue
        if type and v.type != type:
            continue
        en_item = model_items.get(v.name)
        size = en_item.get('size') if en_item else None
        mtime = _format_mtime(en_item.get('mtime')) if en_item else None
        version = en_item.get('version') if en_item else None
        matched.append(ItemModelV2(
            title=v.title,
            model_name=v.name,
            filename=v.filename,
            type=v.type,
            hash=v.shorthash,
            sha256=v.sha256,
            size=size,
            mtime=mtime,
            version=version,
            subfolder=getattr(v, 'subfolder', None),
        ))
    matched.sort(key=lambda m: m.title.lower())
    total = len(matched)
    page = matched[offset:offset + limit] if limit is not None else matched[offset:]
    return ResModelsV2(items=page, total=total, offset=offset, limit=limit or total)


# --- Samplers ---

RES4LYF_KEYS = {'shift', 'base_shift', 'max_shift'}

@router.get("/samplers", response_model=list[ItemSamplerV2], tags=["Enumerators"])
async def get_samplers_v2(
    model_type: str | None = Query(default=None, description="Model type for compatibility filtering"),
    group: str | None = Query(default=None, description="Filter by sampler group"),
):
    """List available samplers with optional compatibility filtering."""
    from modules import sd_samplers_diffusers
    compat = {}
    if shared.sd_loaded and shared.sd_model is not None:
        compat = sd_samplers_diffusers.get_sampler_compatibility(shared.sd_model)
    elif model_type:
        caps = sd_samplers_diffusers._get_cls_caps()  # pylint: disable=protected-access
        flow_types = {'f1', 'f2', 'sd3', 'lumina', 'auraflow', 'sana', 'hidream', 'flux', 'stable diffusion 3', 'hunyuan'}
        lower_mt = model_type.lower()
        requires_flow = lower_mt in flow_types or 'flux' in lower_mt or 'flow' in lower_mt
        for name, cap in caps.items():
            if requires_flow:
                compat[name] = cap['flow']
            else:
                compat[name] = not cap['is_flow_only']
    result = []
    for k, v in sd_samplers_diffusers.config.items():
        if k in ('All', 'Default', 'Res4Lyf'):
            continue
        if 'FlowMatch' in k:
            sampler_group = 'FlowMatch'
        elif RES4LYF_KEYS.issubset(set(v.keys())):
            sampler_group = 'Res4Lyf'
        else:
            sampler_group = 'Standard'
        if group and sampler_group != group:
            continue
        compatible = compat.get(k) if compat else None
        if model_type and compatible is False:
            continue
        result.append(ItemSamplerV2(name=k, group=sampler_group, compatible=compatible, options=v))
    return result


# --- History ---

@router.get("/history", response_model=ResHistoryV2, tags=["Server"])
async def get_history_v2(
    since: float | None = Query(default=None, description="Return entries after this timestamp"),
    job: str | None = Query(default=None, description="Filter by job name"),
    op: str | None = Query(default=None, description="Filter by operation"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
):
    """List generation history entries with optional filtering."""
    items = list(shared.state.state_history)
    if since is not None:
        items = [h for h in items if (h.get('timestamp') or 0) > since]
    if job:
        lower_job = job.lower()
        items = [h for h in items if lower_job in (h.get('job', '') or '').lower()]
    if op:
        lower_op = op.lower()
        items = [h for h in items if lower_op in (h.get('op', '') or '').lower()]
    items.reverse()
    total = len(items)
    page = items[offset:offset + limit]
    return ResHistoryV2(
        items=[ItemHistoryV2(**h) for h in page],
        total=total,
        offset=offset,
        limit=limit,
    )


# --- Checkpoint ---

def _build_checkpoint_info() -> ResCheckpointV2:
    if not shared.sd_loaded or shared.sd_model is None:
        return ResCheckpointV2(loaded=False)
    info = ResCheckpointV2(
        loaded=True,
        type=shared.sd_model_type,
        class_name=shared.sd_model.__class__.__name__,
    )
    if hasattr(shared.sd_model, 'sd_model_checkpoint'):
        info.checkpoint = shared.sd_model.sd_model_checkpoint
    if hasattr(shared.sd_model, 'sd_checkpoint_info'):
        ci = shared.sd_model.sd_checkpoint_info
        info.title = ci.title
        info.name = ci.name
        info.filename = ci.filename
        info.hash = ci.shorthash
    return info


@router.get("/checkpoint", response_model=ResCheckpointV2, tags=["Models"])
async def get_checkpoint_v2():
    """Return information about the currently loaded checkpoint."""
    return _build_checkpoint_info()


@router.post("/checkpoint", response_model=ResSetCheckpointV2, tags=["Models"])
async def set_checkpoint_v2(req: ReqSetCheckpointV2):
    """Load a checkpoint by name with optional dtype and force options."""
    def _load():
        from modules import sd_models, devices
        if req.force:
            sd_models.unload_model_weights(op='model')
        if req.dtype is not None:
            shared.opts.cuda_dtype = req.dtype
            devices.set_dtype()
        shared.opts.sd_model_checkpoint = req.sd_model_checkpoint
        return sd_models.reload_model_weights()
    model = await asyncio.to_thread(_load)
    return ResSetCheckpointV2(ok=model is not None, checkpoint=_build_checkpoint_info())


# --- VAE ---

@router.get("/sd-vae", response_model=list[ItemVaeV2], tags=["Models"])
async def get_sd_vae_v2():
    """List available VAE models."""
    from modules.sd_vae import vae_dict
    return [ItemVaeV2(name=str(k), filename=str(vae_dict[k])) for k in vae_dict]


# --- Upscalers ---

@router.get("/upscalers", response_model=list[ItemUpscalerV2], tags=["Models"])
async def get_upscalers_v2():
    """List available upscaler models."""
    return [
        ItemUpscalerV2(
            name=str(u.name),
            group=str(u.scaler.name) if u.scaler and u.scaler.name else "Other",
            model_name=str(u.scaler.model_name) if u.scaler and u.scaler.model_name else None,
            model_path=str(u.data_path) if u.data_path else None,
            scale=u.scale,
        )
        for u in shared.sd_upscalers
    ]


# --- Checkpoint Mutations ---

@router.post("/checkpoint/refresh", response_model=ResSetCheckpointV2, tags=["Models"])
async def refresh_checkpoints_v2():
    """Rescan checkpoint directories and update the available models list."""
    await asyncio.to_thread(shared.refresh_checkpoints)
    return ResSetCheckpointV2(ok=True, checkpoint=_build_checkpoint_info())


@router.post("/checkpoint/reload", response_model=ResSetCheckpointV2, tags=["Models"])
async def reload_checkpoint_v2(force: bool = Query(default=False, description="Unload model first for a clean reload")):
    """Reload the selected checkpoint."""
    def _reload():
        from modules import sd_models
        if force:
            sd_models.unload_model_weights(op='model')
        sd_models.reload_model_weights()
    await asyncio.to_thread(_reload)
    return ResSetCheckpointV2(ok=True, checkpoint=_build_checkpoint_info())


@router.post("/checkpoint/unload", response_model=ResSetCheckpointV2, tags=["Models"])
async def unload_checkpoint_v2():
    """Unload the current model and refiner from memory."""
    def _unload():
        from modules import sd_models
        sd_models.unload_model_weights(op='model')
        sd_models.unload_model_weights(op='refiner')
        sd_models.unload_auxiliary_models()
    await asyncio.to_thread(_unload)
    return ResSetCheckpointV2(ok=True, checkpoint=_build_checkpoint_info())


# --- Scripts ---

@router.get("/scripts", response_model=ResScriptsV2, tags=["Scripts"])
async def get_scripts_v2(
    context: str | None = Query(default=None, description="Filter by context: txt2img, img2img, control"),
    alwayson: bool | None = Query(default=None, description="Filter by always-on status"),
):
    """List available scripts with optional context and always-on filtering."""
    from modules import scripts_manager
    runners = [
        (scripts_manager.scripts_txt2img, 'txt2img'),
        (scripts_manager.scripts_img2img, 'img2img'),
        (scripts_manager.scripts_control, 'control'),
    ]
    dedup: dict[str, ItemScriptV2] = {}
    for runner, ctx_label in runners:
        if runner is None:
            continue
        for script in runner.scripts:
            if script.api_info is None:
                continue
            info = script.api_info
            key = info.name
            if key in dedup:
                if ctx_label not in dedup[key].contexts:
                    dedup[key].contexts.append(ctx_label)
            else:
                args = [{'label': a.label, 'value': a.value, 'minimum': a.minimum, 'maximum': a.maximum, 'step': a.step, 'choices': a.choices} for a in info.args] if hasattr(info, 'args') else []
                dedup[key] = ItemScriptV2(
                    name=info.name,
                    is_alwayson=info.is_alwayson,
                    contexts=[ctx_label],
                    args=args,
                )
    scripts = list(dedup.values())
    if context:
        scripts = [s for s in scripts if context.lower() in s.contexts]
    if alwayson is not None:
        scripts = [s for s in scripts if s.is_alwayson == alwayson]
    return ResScriptsV2(scripts=scripts)


# --- Embeddings ---

@router.get("/embeddings", response_model=ResEmbeddingsV2, tags=["Enumerators"])
async def get_embeddings_v2():
    """List loaded and skipped textual-inversion embeddings for the current model."""
    db = getattr(shared.sd_model, 'embedding_db', None) if shared.sd_loaded else None
    if db is None:
        return ResEmbeddingsV2()

    def _embed_item(emb) -> ItemEmbeddingV2:
        return ItemEmbeddingV2(
            name=emb.name,
            filename=getattr(emb, 'filename', None),
            step=getattr(emb, 'step', None),
            shape=getattr(emb, 'shape', None),
            vectors=getattr(emb, 'vectors', None),
            sd_checkpoint=getattr(emb, 'sd_checkpoint', None),
            sd_checkpoint_name=getattr(emb, 'sd_checkpoint_name', None),
        )

    loaded = [_embed_item(e) for e in db.word_embeddings.values()]
    skipped = []
    for v in db.skipped_embeddings.values():
        if isinstance(v, list):
            skipped.extend(_embed_item(e) for e in v)
        else:
            skipped.append(_embed_item(v))
    return ResEmbeddingsV2(loaded=loaded, skipped=skipped)


# --- Prompt Styles ---

@router.get("/prompt-styles", response_model=list[ItemPromptStyleV2], tags=["Enumerators"])
async def get_prompt_styles_v2():
    """List all saved prompt styles with prompt templates and metadata."""
    return [
        ItemPromptStyleV2(
            name=v.name,
            prompt=v.prompt or None,
            negative_prompt=v.negative_prompt or None,
            extra=v.extra or None,
            description=getattr(v, 'description', None) or None,
            wildcards=getattr(v, 'wildcards', None) or None,
            filename=v.filename or None,
            preview=v.preview or None,
            mtime=_format_mtime(getattr(v, 'mtime', None)),
        )
        for v in shared.prompt_styles.styles.values()
    ]


# --- Extra Networks Refresh ---

@router.post("/extra-networks/refresh", response_model=ResRefreshNetworksV2, tags=["Enumerators"])
async def refresh_extra_networks_v2():
    """Rescan all extra-network directories (LoRA, embeddings, etc.) and rebuild caches."""
    def _refresh():
        from modules.lora import lora_load
        lora_load.list_available_networks()
        for page in shared.extra_networks:
            page.refresh_time = 0
            page.create_items('txt2img')
    await asyncio.to_thread(_refresh)
    total = sum(len(pg.items) for pg in shared.extra_networks)
    return ResRefreshNetworksV2(ok=True, total=total)


# --- Log ---

@router.get("/log", response_model=ResLogV2, tags=["Server"])
async def get_log_v2(lines: int = Query(default=100, ge=0)):
    """Return recent log lines from the in-memory buffer."""
    from modules.logger import log
    result = log.buffer[-lines:] if lines > 0 else log.buffer.copy()
    return ResLogV2(lines=result, total=len(result))


@router.delete("/log", response_model=ResLogClearV2, tags=["Server"])
async def delete_log_v2():
    """Clear the in-memory log buffer."""
    from modules.logger import log
    log.buffer.clear()
    return ResLogClearV2(ok=True)


# --- PNG Info ---

@router.post("/png-info", response_model=ResPngInfoV2, tags=["Server"])
async def post_png_info_v2(req: ReqPngInfoV2):
    """Extract generation parameters from a PNG image's metadata."""
    from modules import images, script_callbacks, infotext
    from modules.api.helpers import decode_base64_to_image
    if not req.image.strip():
        return ResPngInfoV2(ok=False, info="")
    image = decode_base64_to_image(req.image.strip())
    if image is None:
        return ResPngInfoV2(ok=False, info="")
    geninfo, items = images.read_info_from_image(image)
    if geninfo is None:
        geninfo = ""
    params = infotext.parse(geninfo)
    script_callbacks.infotext_pasted_callback(geninfo, params)
    return ResPngInfoV2(ok=True, info=geninfo, items=items, parameters=params)


# --- Extensions ---

@router.get("/extensions", response_model=list[ItemExtensionV2], tags=["Server"])
async def get_extensions_v2():
    """List installed extensions with their remote URLs, branches, versions, and enabled status."""
    def _list():
        from modules import extensions
        extensions.list_extensions()
        result = []
        for ext in extensions.extensions:
            ext.read_info()
            if ext.remote is not None:
                branch = ext.branch if ext.branch and ext.branch != "uknnown" else "unknown"  # "uknnown" is a known typo in extension metadata
                result.append(ItemExtensionV2(
                    name=ext.name,
                    remote=ext.remote,
                    branch=branch,
                    commit_hash=ext.commit_hash,
                    version=ext.version,
                    commit_date=str(ext.commit_date) if ext.commit_date is not None else None,
                    enabled=ext.enabled,
                ))
        return result
    return await asyncio.to_thread(_list)


# --- Detailers ---

@router.get("/detailers", response_model=list[ItemDetailerV2], tags=["Enumerators"])
async def get_detailers_v2():
    """List available detailer (YOLO) models for face/object detection and inpainting."""
    shared.yolo.enumerate()
    return [ItemDetailerV2(name=k, path=v) for k, v in shared.yolo.list.items()]
