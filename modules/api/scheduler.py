"""Scheduler task management API routes."""

import datetime as dt
import json
import os
from typing import Any, Literal
import mimetypes

from fastapi import Depends, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field

from modules import scheduler


TaskStatus = Literal['scheduled', 'running', 'paused', 'completed', 'failed']


class ParametersUpdate(BaseModel):
    parameters: dict[str, Any] = Field(description='Named generation parameters to update')


class RescheduleRequest(BaseModel):
    scheduled_at: dt.datetime


class MoveTaskRequest(BaseModel):
    target_id: int
    position: Literal['before', 'after'] = 'before'


class SchedulerSettingsRequest(BaseModel):
    state: Literal['Active', 'Paused', 'Auto'] | None = None
    start: str | None = None
    end: str | None = None


class PruneRequest(BaseModel):
    before: dt.datetime


def _require_scheduler_available():
    if scheduler._worker_start_failed: # pylint: disable=protected-access
        raise HTTPException(status_code=503, detail='Scheduler is disabled')


def _add_scheduler_route(api, path, endpoint, **kwargs):
    api.add_api_route(path, endpoint, dependencies=[Depends(_require_scheduler_available)], **kwargs)


def list_tasks(
    status: TaskStatus | None = None,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    parameters: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    criteria = None
    if parameters:
        try:
            criteria = json.loads(parameters)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f'Invalid parameter JSON: {exc}') from exc
        if not isinstance(criteria, dict):
            raise HTTPException(status_code=422, detail='Parameter search must be a JSON object')
    if start is not None or end is not None or criteria:
        return scheduler.search_tasks(start=start, end=end, status=status, parameters=criteria, limit=limit, offset=offset)
    return scheduler.list_tasks(status=status, limit=limit, offset=offset)


def get_task(task_id: int):
    task = scheduler.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    return task


def _has_image_reference(value, image_index):
    if isinstance(value, list):
        return any(_has_image_reference(item, image_index) for item in value)
    if isinstance(value, dict):
        if value.get('__sd_scheduler_type__') == 'image':
            return value.get('id') == image_index
        return any(_has_image_reference(item, image_index) for item in value.values())
    return False


def _has_media_reference(value, media_index):
    if isinstance(value, list):
        return any(_has_media_reference(item, media_index) for item in value)
    if isinstance(value, dict):
        if value.get('__sd_scheduler_type__') in ('image', 'file', 'blob'):
            return value.get('id') == media_index
        return any(_has_media_reference(item, media_index) for item in value.values())
    return False


def _media_type_for_reference(value, image_index):
    if isinstance(value, list):
        for item in value:
            media_type = _media_type_for_reference(item, image_index)
            if media_type:
                return media_type
    elif isinstance(value, dict):
        marker = value.get('__sd_scheduler_type__')
        if marker in ('image', 'file', 'blob') and value.get('id') == image_index:
            return value.get('media_type') or ('image/png' if marker == 'image' else 'application/octet-stream')
        for item in value.values():
            media_type = _media_type_for_reference(item, image_index)
            if media_type:
                return media_type
    return None


def get_task_image(task_id: int, image_index: int):
    task = scheduler.get_task(task_id)
    if task is None or image_index < 0 or not _has_image_reference(task['parameters'], image_index):
        raise HTTPException(status_code=404, detail='Scheduler input image not found')
    image = scheduler.get_task_image(task_id, image_index)
    if image is None:
        raise HTTPException(status_code=404, detail='Scheduler input image not found')
    return Response(content=image, media_type='image/png')


def get_task_media(task_id: int, media_index: int):
    task = scheduler.get_task(task_id)
    if task is None or media_index < 0 or not _has_media_reference(task['parameters'], media_index):
        raise HTTPException(status_code=404, detail='Scheduler input media not found')
    media = scheduler.get_task_image(task_id, media_index)
    if media is None:
        raise HTTPException(status_code=404, detail='Scheduler input media not found')
    media_type = _media_type_for_reference(task['parameters'], media_index)
    return Response(content=media, media_type=media_type or mimetypes.guess_type('media')[0] or 'application/octet-stream')


def get_scheduler_status():
    return scheduler.get_status()


def execute_task(task_id: int):
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    if not scheduler.queue_existing_task(task_id, immediate=True):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} cannot be queued')
    return {'task_id': task_id, 'status': 'scheduled'}


def update_task_parameters(task_id: int, request: ParametersUpdate):
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    if not scheduler.update_task_parameters(task_id, request.parameters):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} parameters cannot be updated')
    return scheduler.get_task(task_id)


async def update_task_parameters_media(
    task_id: int,
    parameters: str = Form(...),
    media_ids: str = Form(...),
    media: list[UploadFile] = File(...),
):
    try:
        parameter_updates = json.loads(parameters)
        media_indexes = json.loads(media_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f'Invalid scheduler update JSON: {exc}') from exc
    if not isinstance(parameter_updates, dict) or not isinstance(media_indexes, list) or len(media_indexes) != len(media):
        raise HTTPException(status_code=422, detail='Parameters and media IDs must be valid matching collections')
    if any(type(index) is not int or index < 0 for index in media_indexes) or len(set(media_indexes)) != len(media_indexes):
        raise HTTPException(status_code=422, detail='Media IDs must be unique non-negative integers')
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    replacements = {}
    metadata = {}
    for index, upload in zip(media_indexes, media):
        filename = os.path.basename((upload.filename or f'media-{index}').replace('\\', '/')) or f'media-{index}'
        media_type = (upload.content_type or '').split(';', maxsplit=1)[0].strip().lower()
        if not media_type or media_type == 'application/octet-stream':
            media_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        replacements[index] = await upload.read()
        metadata[index] = {'filename': filename, 'media_type': media_type}
    if not scheduler.update_task_parameters_with_media(
            task_id,
            parameter_updates,
            media_replacements=replacements,
            media_metadata=metadata,
    ):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} parameters cannot be updated')
    return scheduler.get_task(task_id)


def delete_task(task_id: int):
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    if not scheduler.delete_task(task_id):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} cannot be deleted')
    return {'task_id': task_id, 'deleted': True}


def pause_task(task_id: int):
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    if not scheduler.pause_task(task_id):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} cannot be paused')
    return {'task_id': task_id, 'status': 'paused'}


def resume_task(task_id: int):
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    if not scheduler.resume_task(task_id):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} cannot be resumed')
    return {'task_id': task_id, 'status': 'scheduled'}


def move_task(task_id: int, request: MoveTaskRequest):
    task = scheduler.get_task(task_id)
    target = scheduler.get_task(request.target_id)
    if task is None or target is None:
        raise HTTPException(status_code=404, detail='Scheduler task not found')
    if task['status'] not in ('scheduled', 'paused') or target['status'] not in ('scheduled', 'paused'):
        raise HTTPException(status_code=409, detail='Only queued tasks can be reordered')
    if not scheduler.move_task(task_id, request.target_id, request.position):
        raise HTTPException(status_code=409, detail='Scheduler tasks cannot be reordered')
    return {'task_id': task_id, 'target_id': request.target_id, 'position': request.position}


def reschedule_task(task_id: int, request: RescheduleRequest):
    if scheduler.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail=f'Scheduler task {task_id} not found')
    if not scheduler.reschedule_task(task_id, request.scheduled_at):
        raise HTTPException(status_code=409, detail=f'Scheduler task {task_id} cannot be rescheduled')
    return {'task_id': task_id, 'status': 'scheduled', 'scheduled_at': request.scheduled_at}


def pause_scheduler():
    scheduler.pause_all()
    return {'state': 'Paused', 'paused': True}


def resume_scheduler():
    scheduler.resume_all()
    return {'state': 'Active', 'paused': False}


def update_scheduler_settings(request: SchedulerSettingsRequest):
    if request.state is None and request.start is None and request.end is None:
        raise HTTPException(status_code=422, detail='At least one scheduler setting is required')
    try:
        return scheduler.set_scheduler_state(request.state, request.start, request.end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def prune_tasks(request: PruneRequest):
    count = scheduler.prune_tasks(request.before)
    return {'deleted': count, 'before': request.before}


def register_api(api):
    _add_scheduler_route(api, '/sdapi/v1/scheduler', get_scheduler_status, methods=['GET'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks', list_tasks, methods=['GET'], response_model=list[dict], tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}', get_task, methods=['GET'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/images/{image_index}', get_task_image, methods=['GET'], tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/media/{media_index}', get_task_media, methods=['GET'], tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}', delete_task, methods=['DELETE'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/execute', execute_task, methods=['POST'], status_code=202, response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/parameters', update_task_parameters, methods=['PATCH'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/parameters/media', update_task_parameters_media, methods=['PATCH'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/pause', pause_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/resume', resume_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/reschedule', reschedule_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/tasks/{task_id}/move', move_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/pause', pause_scheduler, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/resume', resume_scheduler, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/state', update_scheduler_settings, methods=['POST'], response_model=dict, tags=['Scheduler'])
    _add_scheduler_route(api, '/sdapi/v1/scheduler/prune', prune_tasks, methods=['POST'], response_model=dict, tags=['Scheduler'])
