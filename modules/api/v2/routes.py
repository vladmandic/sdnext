import asyncio
import json
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from modules.api.v2.models import JobResponse, JobListResponse, JobResult, ImageRef, StatusResponse, VideoEngine, VideoModel, VideoLoadResponse, MessageResponse, FramePackLoadResponse


router = APIRouter(prefix="/sdapi/v2", tags=["v2"])


def _job_to_response(job: dict) -> JobResponse:
    result = job.get('result')
    job_result = None
    if result:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                result = None
        if isinstance(result, dict):
            images = [ImageRef(**img) for img in result.get('images', [])]
            job_result = JobResult(images=images, info=result.get('info', {}), params=result.get('params', {}))
    return JobResponse(
        id=job['id'],
        type=job['type'],
        status=job['status'],
        progress=job.get('progress', 0),
        step=job.get('step', 0),
        steps=job.get('steps', 0),
        created_at=job['created_at'],
        started_at=job.get('started_at'),
        completed_at=job.get('completed_at'),
        error=job.get('error'),
        result=job_result,
    )


@router.post("/jobs", response_model=JobResponse, status_code=202, tags=["Jobs"])
async def submit_job(request: dict):
    from modules.api.v2.job_queue import job_queue
    job_type = request.get('type')
    if not job_type:
        raise HTTPException(status_code=400, detail="Missing 'type' field")
    valid_types = {'generate', 'upscale', 'caption', 'enhance', 'detect', 'preprocess', 'video', 'framepack', 'ltx'}
    if job_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid job type: {job_type}. Must be one of: {', '.join(sorted(valid_types))}")
    priority = request.pop('priority', 0)
    job = job_queue.submit(job_type=job_type, params=request, priority=priority)
    return _job_to_response(job)


@router.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    status: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    from modules.api.v2.job_queue import job_queue
    items, total = job_queue.store.list(status=status, job_type=type, limit=limit, offset=offset)
    return JobListResponse(items=[_job_to_response(j) for j in items], total=total, offset=offset, limit=limit)


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(job_id: str):
    from modules.api.v2.job_queue import job_queue
    job = job_queue.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    resp = _job_to_response(job)
    # Optionally include inline base64 images
    try:
        from modules import shared
        if getattr(shared.opts, 'api_v2_base64', False) and resp.result and resp.result.images:
            _embed_base64(job, resp)
    except Exception:
        pass
    return resp


@router.delete("/jobs/{job_id}", response_model=StatusResponse, tags=["Jobs"])
async def cancel_job(job_id: str):
    from modules.api.v2.job_queue import job_queue
    job = job_queue.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    success = job_queue.cancel(job_id)
    if not success:
        raise HTTPException(status_code=409, detail="Job cannot be cancelled (already completed/failed/cancelled)")
    return {"id": job_id, "status": "cancelled"}


def _embed_base64(job: dict, resp: JobResponse) -> None:
    import base64
    for img_ref in resp.result.images:
        file_path = _get_ref_path(job, 'images', img_ref.index)
        if file_path and os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                img_ref.__dict__['data'] = base64.b64encode(f.read()).decode('ascii')


def _get_ref_path(job: dict, key: str, index: int) -> str | None:
    result = job.get('result')
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(result, dict):
        return None
    items = result.get(key, [])
    if index < 0 or index >= len(items):
        return None
    return items[index].get('path')


def _serve_job_file(job: dict, key: str, index: int):
    file_path = _get_ref_path(job, key, index)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"{key} index {index} out of range")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    ext = os.path.splitext(file_path)[1].lstrip('.').lower()
    media_types = {'png': 'image/png', 'jpeg': 'image/jpeg', 'jpg': 'image/jpeg', 'webp': 'image/webp', 'jxl': 'image/jxl', 'mp4': 'video/mp4', 'webm': 'video/webm', 'gif': 'image/gif'}
    return FileResponse(file_path, media_type=media_types.get(ext, 'application/octet-stream'))


def _get_completed_job(job_id: str):
    from modules.api.v2.job_queue import job_queue
    job = job_queue.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job['status'] != 'completed':
        raise HTTPException(status_code=409, detail="Job not completed")
    return job


@router.get("/jobs/{job_id}/images/{index}", tags=["Jobs"])
async def get_job_image(job_id: str, index: int):
    return _serve_job_file(_get_completed_job(job_id), 'images', index)


@router.get("/jobs/{job_id}/processed/{index}", tags=["Jobs"])
async def get_job_processed(job_id: str, index: int):
    return _serve_job_file(_get_completed_job(job_id), 'processed', index)


@router.get("/video/engines", response_model=list[VideoEngine], tags=["Video"])
async def list_video_engines():
    from modules.video_models import models_def
    result = []
    for engine_name, model_list in models_def.models.items():
        if engine_name == 'None':
            continue
        model_names = [m.name for m in model_list if m.name != 'None']
        result.append({'engine': engine_name, 'models': model_names})
    return result


@router.get("/video/engines/{engine}/models", response_model=list[VideoModel], tags=["Video"])
async def list_video_engine_models(engine: str):
    from modules.video_models import models_def
    if engine not in models_def.models:
        raise HTTPException(status_code=404, detail=f"Engine not found: {engine}")
    model_list = models_def.models[engine]
    return [{'name': m.name, 'repo': m.repo or '', 'url': m.url or ''} for m in model_list if m.name != 'None']


@router.post("/video/load", response_model=VideoLoadResponse, tags=["Video"])
async def load_video_model(request: dict):
    engine = request.get('engine', '')
    model = request.get('model', '')
    if not engine or not model:
        raise HTTPException(status_code=400, detail="Both 'engine' and 'model' fields are required")
    def _load():
        from modules.video_models import video_ui
        return list(video_ui.model_load(engine, model))
    messages = await asyncio.to_thread(_load)
    return {"engine": engine, "model": model, "messages": messages}


@router.get("/framepack/variants", response_model=list[str], tags=["Video"])
async def list_framepack_variants():
    from modules.framepack import framepack_load
    return list(framepack_load.models.keys())


@router.post("/framepack/load", response_model=FramePackLoadResponse, tags=["Video"])
async def load_framepack_model(request: dict):
    variant = request.get('variant', 'bi-directional')
    attention = request.get('attention', 'Default')
    def _load():
        from modules.framepack import framepack_wrappers
        messages = []
        for item in framepack_wrappers.load_model(variant, attention):
            if isinstance(item, tuple) and len(item) > 2 and isinstance(item[2], str):
                messages.append(item[2])
        return messages
    messages = await asyncio.to_thread(_load)
    return {"variant": variant, "messages": messages}


@router.post("/framepack/unload", response_model=MessageResponse, tags=["Video"])
async def unload_framepack_model():
    def _unload():
        from modules.framepack import framepack_wrappers
        list(framepack_wrappers.unload_model())
    await asyncio.to_thread(_unload)
    return {"messages": ["Model unloaded"]}
