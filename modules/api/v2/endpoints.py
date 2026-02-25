import os
from datetime import datetime
from typing import Any
from fastapi import APIRouter, Query
from modules import shared
from modules.api.v2.models import (
    ItemExtraNetworkV2, ResExtraNetworksV2,
    ItemModelV2, ResModelsV2,
    ItemSamplerV2,
    ItemHistoryV2, ResHistoryV2,
    ResCheckpointV2, ReqSetCheckpointV2, ResSetCheckpointV2,
    ItemScriptV2, ResScriptsV2,
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
def get_extra_networks_v2(
    page: str | None = Query(default=None, description="Filter by page name (lora, model, embedding, etc.)"),
    search: str | None = Query(default=None, description="Case-insensitive search across name, title, filename, tags"),
    subfolder: str | None = Query(default=None, description="Filter by subfolder component in filename"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
):
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
    return ResExtraNetworksV2(items=matched[offset:offset + limit], total=total, offset=offset, limit=limit)


# --- Options ---

@router.get("/options", tags=["Server"])
def get_options_v2(
    keys: str | None = Query(default=None, description="Comma-separated list of keys to return"),
) -> dict[str, Any]:
    from modules.api.server import get_config
    options = get_config()
    if keys:
        requested = [k.strip() for k in keys.split(',') if k.strip()]
        return {k: v for k, v in options.items() if k in requested}
    return options


# --- SD Models ---

@router.get("/sd-models", response_model=ResModelsV2, tags=["Models"])
def get_sd_models_v2(
    search: str | None = Query(default=None, description="Case-insensitive search on title/filename"),
    type: str | None = Query(default=None, description="Filter by model type", alias="type"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
):
    from modules import sd_checkpoint
    lower_search = search.lower() if search else None
    matched = []
    for v in sd_checkpoint.checkpoints_list.values():
        if lower_search:
            if lower_search not in (v.title or '').lower() and lower_search not in (v.filename or '').lower():
                continue
        if type and v.type != type:
            continue
        size = None
        mtime = None
        try:
            if v.filename and os.path.isfile(v.filename):
                size = os.path.getsize(v.filename)
                mtime = datetime.fromtimestamp(os.path.getmtime(v.filename)).isoformat()
        except OSError:
            pass
        version = None
        for pg in shared.extra_networks:
            if pg.name == 'model':
                for item in pg.items:
                    if item.get('name', '') == v.name:
                        version = item.get('version', None)
                        break
                break
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
    return ResModelsV2(items=matched[offset:offset + limit], total=total, offset=offset, limit=limit)


# --- Samplers ---

RES4LYF_KEYS = {'shift', 'base_shift', 'max_shift'}

@router.get("/samplers", response_model=list[ItemSamplerV2], tags=["Enumerators"])
def get_samplers_v2(
    model_type: str | None = Query(default=None, description="Model type for compatibility filtering"),
    group: str | None = Query(default=None, description="Filter by sampler group"),
):
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
def get_history_v2(
    since: float | None = Query(default=None, description="Return entries after this timestamp"),
    job: str | None = Query(default=None, description="Filter by job name"),
    op: str | None = Query(default=None, description="Filter by operation"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
):
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
def get_checkpoint_v2():
    return _build_checkpoint_info()


@router.post("/checkpoint", response_model=ResSetCheckpointV2, tags=["Models"])
def set_checkpoint_v2(req: ReqSetCheckpointV2):
    from modules import sd_models, devices
    if req.force:
        sd_models.unload_model_weights(op='model')
    if req.dtype is not None:
        shared.opts.cuda_dtype = req.dtype
        devices.set_dtype()
    shared.opts.sd_model_checkpoint = req.sd_model_checkpoint
    model = sd_models.reload_model_weights()
    return ResSetCheckpointV2(ok=model is not None, checkpoint=_build_checkpoint_info())


# --- Scripts ---

@router.get("/scripts", response_model=ResScriptsV2, tags=["Scripts"])
def get_scripts_v2(
    context: str | None = Query(default=None, description="Filter by context: txt2img, img2img, control"),
    alwayson: bool | None = Query(default=None, description="Filter by always-on status"),
):
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
