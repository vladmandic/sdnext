"""Typed API contracts and session routes for the optional chat feature."""

import sqlite3
from typing import Any, Literal

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field

from modules import shared
from modules.api import control as api_control
from modules.chat import commands, db
from modules.logger import log


ContextId = Literal['image', 'video', 'process', 'caption']
ChatCommand = Literal[
    'context', 'set', 'get', 'list', 'params', 'help', 'opts', 'prompt', 'negative',
    'generate', 'run', 'start', 'again', 'reprocess', 'schedule', 'cancel', 'abort', 'stop', 'interrupt', 'pause',
    'resume', 'skip', 'history', 'reset', 'enhance', 'dlss', 'upscale', 'model',
]
ServerEndpointAlias = Literal[
    'info', 'ver', 'version', 'gpu', 'smi', 'mem', 'memory', 'status', 'storage',
]
OperationStatus = Literal['running', 'completed', 'failed', 'cancelled']


class ReqSessionCreate(BaseModel):
    title: str | None = Field(default=None, title='Title')
    context: ContextId = Field(default='image', title='Context')


class ReqSessionUpdate(BaseModel):
    title: str | None = Field(default=None, title='Title')
    context: ContextId | None = Field(default=None, title='Context')
    drafts: dict[str, dict[str, Any]] | None = Field(default=None, title='Context drafts')
    last_generation_prompt: str | None = Field(default=None, title='Last generation prompt')
    last_generation_snapshot: dict[str, Any] | None = Field(default=None, title='Last generation snapshot')


class ResSession(BaseModel):
    id: str
    title: str
    context: ContextId
    drafts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    last_generation_prompt: str | None = None
    last_generation_snapshot: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class ResMessage(BaseModel):
    id: str
    role: Literal['system', 'user', 'assistant']
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ResOperation(BaseModel):
    id: str
    message_id: str | None = None
    command: str
    context: ContextId
    status: OperationStatus
    snapshot: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    started_at: str
    completed_at: str | None = None


class ResSessionDetail(BaseModel):
    session: ResSession
    messages: list[ResMessage]
    operations: list[ResOperation]
    restore_errors: list[str] = Field(default_factory=list)


class ReqChatAction(BaseModel):
    command: ChatCommand | ServerEndpointAlias
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReqChatMessage(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    image: str | None = None


class ReqChatMessageUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=16000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResChatAction(BaseModel):
    kind: Literal['text', 'operation', 'schedule', 'endpoint', 'error']
    text: str = ''
    data: dict[str, Any] = Field(default_factory=dict)
    sections: list[dict[str, str]] = Field(default_factory=list)
    operation_id: str | None = None


class ResChatSubmission(BaseModel):
    session: ResSession
    user_message: ResMessage
    assistant_message: ResMessage
    result: ResChatAction


def get_sessions(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    return db.list_sessions(limit=limit, offset=offset)


def create_session(req: ReqSessionCreate):
    try:
        return db.create_session(title=req.title, context=req.context)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_session(session_id: str):
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Chat session not found')
    return ResSessionDetail(
        session=session,
        messages=db.list_messages(session_id),
        operations=db.list_operations(session_id),
    )


def restore_session(session_id: str):
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Chat session not found')
    restore_errors = commands.restore_session_state(session_id)
    return ResSessionDetail(
        session=db.get_session(session_id),
        messages=db.list_messages(session_id),
        operations=db.list_operations(session_id),
        restore_errors=restore_errors,
    )


def _execute_chat_operation(session_id: str, user_message: dict[str, Any], result: dict[str, Any], api_instance):
    data = result.get('data') or {}
    descriptors = data.get('endpoints')
    if data.get('alias') != 'generate' or not isinstance(descriptors, list) or len(descriptors) != 1:
        raise ValueError('Invalid chat generation descriptor')
    operation_command = data.get('command', 'generate')
    if operation_command not in {'generate', 'again'}:
        raise ValueError('Invalid chat operation command')
    descriptor = descriptors[0]
    if not isinstance(descriptor, dict) or descriptor.get('method') != 'POST':
        raise ValueError('Invalid chat generation endpoint')
    handlers = {
        '/sdapi/v1/control': 'image',
        '/sdapi/v1/video': 'video',
        '/sdapi/v1/process': 'process',
        '/sdapi/v1/vqa': 'caption',
    }
    context = handlers.get(descriptor.get('path'))
    session = db.get_session(session_id)
    if context is None or session is None or session['context'] != context:
        raise ValueError('Chat generation endpoint does not match the session context')
    arguments = descriptor.get('arguments')
    if not isinstance(arguments, dict):
        raise ValueError('Invalid chat generation arguments')

    operation = db.create_operation(
        session_id,
        operation_command,
        context,
        {name: value for name, value in arguments.items() if name != 'image'},
        message_id=user_message['id'],
    )
    operation_id = operation['id']
    try:
        if context == 'image':
            control_api_instance = getattr(api_instance, 'control', None)
            if control_api_instance is None:
                raise RuntimeError('The control generation API is unavailable')
            request = api_control.ReqControl.model_validate(arguments)
            response = control_api_instance.post_control(request)
            images = response.images or []
            info = response.info or ''
            parameters = response.params or {}
            completion_text = f'Generated {len(images)} image(s)'
            if info:
                completion_text += f'\n{info}'
            output_data = {'images': images, 'info': info, 'parameters': parameters}
            saved_result = {'image_count': len(images), 'info': info, 'parameters': parameters}
            message_metadata = {'image_count': len(images), 'format': 'html'}
        elif context == 'video':
            from modules.api.video import ReqVideo
            video_api_instance = getattr(api_instance, 'video', None)
            if video_api_instance is None:
                raise RuntimeError('The video generation API is unavailable')
            request = ReqVideo.model_validate(arguments)
            response = video_api_instance.post_video(request)
            info = response.info or ''
            completion_text = 'Generated video'
            if info:
                completion_text += f'\n{info}'
            output_data = {
                'video': response.video,
                'video_path': response.video_path,
                'thumbnail': response.thumbnail,
                'frames': response.frames,
                'info': info,
                'parameters': response.params,
            }
            saved_result = {'frames_count': response.frames_count, 'info': info, 'parameters': response.params}
            message_metadata = {'format': 'html'}
        elif context == 'process':
            from modules.api import models
            process_api_instance = getattr(api_instance, 'process', None)
            if process_api_instance is None:
                raise RuntimeError('The image processing API is unavailable')
            request = models.ReqProcessImage.model_validate(arguments)
            response = process_api_instance.extras_single_image_api(request)
            info = response.html_info or ''
            completion_text = 'Image processed'
            if info:
                completion_text += f'\n{info}'
            output_data = {'images': [response.image], 'info': info}
            saved_result = {'info': info}
            message_metadata = {'format': 'html', 'image_count': 1}
        else:
            from modules.api.caption import ReqVQA, post_vqa
            request = ReqVQA.model_validate(arguments)
            response = post_vqa(request)
            completion_text = response.answer or 'Caption complete'
            output_data = {'caption': response.answer, 'annotated_image': response.annotated_image}
            saved_result = {'caption': response.answer}
            message_metadata = {}
        db.update_operation(
            operation_id,
            'completed',
            result=saved_result,
        )
        if context in {'image', 'video'}:
            db.update_session(
                session_id,
                last_generation_prompt=str(arguments.get('prompt') or ''),
                last_generation_snapshot=arguments,
            )
        result.update({
            'text': completion_text,
            'operation_id': operation_id,
            'data': output_data,
            'message_metadata': {
                'kind': 'operation',
                'operation_id': operation_id,
                'status': 'completed',
                **message_metadata,
            },
        })
    except Exception as exc:
        log.error(f'Chat: {context} operation failed: {type(exc).__name__}: {exc}')
        db.update_operation(operation_id, 'failed', error=str(exc))
        result.update({
            'text': f'{context.capitalize()} operation failed: {exc}',
            'operation_id': operation_id,
            'data': {},
            'message_metadata': {
                'kind': 'operation',
                'operation_id': operation_id,
                'status': 'failed',
            },
        })
    return result


def _schedule_chat_operation(result: dict[str, Any]):
    from modules import scheduler
    descriptors = (result.get('data') or {}).get('endpoints')
    if not isinstance(descriptors, list) or len(descriptors) != 1 or not isinstance(descriptors[0], dict):
        raise ValueError('Invalid chat schedule descriptor')
    descriptor = descriptors[0]
    arguments = descriptor.get('arguments')
    if not isinstance(arguments, dict):
        raise ValueError('Invalid chat schedule arguments')
    if descriptor.get('path') == '/sdapi/v1/control':
        response = next(scheduler.enqueue_control_request(**arguments))
        text = response[2]
    elif descriptor.get('path') == '/sdapi/v1/video':
        response = scheduler.enqueue_video_request('video', **arguments)
        text = response[4]
    else:
        raise ValueError('Scheduling is not available for this chat context')
    return {'kind': 'schedule', 'text': text, 'data': {}}


def post_message(session_id: str, req: ReqChatMessage, api_instance=None):
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Chat session not found')
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail='Chat message cannot be empty')
    if session['title'] == 'New chat' and not db.list_messages(session_id, limit=1):
        title = content.splitlines()[0][:64]
        db.update_session(session_id, title=title)
    user_message = db.create_message(session_id, 'user', content)
    try:
        result = commands.execute_command(session_id, content, image=req.image)
        if result.get('kind') == 'operation':
            result = _execute_chat_operation(session_id, user_message, result, api_instance)
        elif result.get('kind') == 'schedule':
            result = _schedule_chat_operation(result)
    except Exception as exc:  # Keep a command failure in the persisted conversation.
        log.error(f'Chat: command failed: {type(exc).__name__}')
        result = {'kind': 'error', 'text': 'Command failed. Check the server log.', 'data': {}}
    metadata = result.get('message_metadata', {
        'kind': result.get('kind', 'text'),
        'data': result.get('data', {}),
    })
    if result.get('sections'):
        metadata['sections'] = result['sections']
    assistant_message = db.create_message(
        session_id,
        'assistant',
        result.get('text', ''),
        metadata=metadata,
    )
    return ResChatSubmission(
        session=db.get_session(session_id),
        user_message=user_message,
        assistant_message=assistant_message,
        result=result,
    )


def patch_message(session_id: str, message_id: str, req: ReqChatMessageUpdate):
    if req.metadata.get('kind') != 'endpoint':
        raise HTTPException(status_code=400, detail='Only endpoint results can update an assistant message')
    try:
        message = db.update_message(session_id, message_id, req.content, req.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if message is None:
        raise HTTPException(status_code=404, detail='Chat endpoint response message not found')
    return message


def update_session(session_id: str, req: ReqSessionUpdate):
    try:
        session = db.update_session(session_id, **req.dict(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=404, detail='Chat session not found')
    return session


def delete_session(session_id: str):
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail='Chat session not found')
    return Response(status_code=204)


def register_api(api):
    try:
        db.initialize()
    except (sqlite3.Error, OSError) as exc:
        shared.opts.data['chat_enabled'] = False
        log.error(f'Chat: disabled for this session because initialization failed: {exc}')
        return False
    api.add_api_route('/sdapi/v1/chat/sessions', get_sessions, methods=['GET'], response_model=list[ResSession], tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions', create_session, methods=['POST'], response_model=ResSession, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}', get_session, methods=['GET'], response_model=ResSessionDetail, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}', update_session, methods=['PATCH'], response_model=ResSession, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}', delete_session, methods=['DELETE'], status_code=204, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/restore', restore_session, methods=['POST'], response_model=ResSessionDetail, tags=['Chat'])
    control_api_instance = api if hasattr(api, 'control') else None

    def post_message_route(session_id: str, req: ReqChatMessage):
        return post_message(session_id, req, control_api_instance)

    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/messages', post_message_route, methods=['POST'], response_model=ResChatSubmission, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/messages/{message_id}', patch_message, methods=['PATCH'], response_model=ResMessage, tags=['Chat'])
    return True
