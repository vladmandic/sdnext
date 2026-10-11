"""Scheduler task management API routes."""

import datetime
import json
from typing import Any, Literal

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field

from modules import scheduler


TaskStatus = Literal['scheduled', 'running', 'paused', 'completed', 'failed']


class ParametersUpdate(BaseModel):
    parameters: dict[str, Any] = Field(description='Named Control generation parameters to update')


class RescheduleRequest(BaseModel):
    scheduled_at: datetime.datetime


class MoveTaskRequest(BaseModel):
    target_id: int
    position: Literal['before', 'after'] = 'before'


class SchedulerSettingsRequest(BaseModel):
    state: Literal['Active', 'Paused', 'Auto'] | None = None
    start: str | None = None
    end: str | None = None


class PruneRequest(BaseModel):
    before: datetime.datetime


def list_tasks(
    status: TaskStatus | None = None,
    start: datetime.datetime | None = None,
    end: datetime.datetime | None = None,
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


def get_task_image(task_id: int, image_index: int):
    task = scheduler.get_task(task_id)
    if task is None or image_index < 0 or not _has_image_reference(task['parameters'], image_index):
        raise HTTPException(status_code=404, detail='Scheduler input image not found')
    image = scheduler.get_task_image(task_id, image_index)
    if image is None:
        raise HTTPException(status_code=404, detail='Scheduler input image not found')
    return Response(content=image, media_type='image/png')


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
    api.add_api_route('/sdapi/v1/scheduler', get_scheduler_status, methods=['GET'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks', list_tasks, methods=['GET'], response_model=list[dict], tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}', get_task, methods=['GET'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/images/{image_index}', get_task_image, methods=['GET'], tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}', delete_task, methods=['DELETE'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/execute', execute_task, methods=['POST'], status_code=202, response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/parameters', update_task_parameters, methods=['PATCH'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/pause', pause_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/resume', resume_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/reschedule', reschedule_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/tasks/{task_id}/move', move_task, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/pause', pause_scheduler, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/resume', resume_scheduler, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/state', update_scheduler_settings, methods=['POST'], response_model=dict, tags=['Scheduler'])
    api.add_api_route('/sdapi/v1/scheduler/prune', prune_tasks, methods=['POST'], response_model=dict, tags=['Scheduler'])
