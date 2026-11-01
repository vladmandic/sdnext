"""Typed API contracts and session routes for the optional chat feature."""

import json
import sqlite3
from html import escape
from typing import Any, Literal

from fastapi import File, HTTPException, Query, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from modules import shared
from modules.api import control as api_control
from modules.chat import artifacts, commands, db
from modules.logger import log


ContextId = Literal['image', 'video', 'process', 'caption']
ChatCommand = Literal[
    'context', 'set', 'get', 'list', 'params', 'help', 'opts', 'prompt', 'negative',
    'generate', 'run', 'start', 'again', 'reprocess', 'schedule', 'cancel', 'abort', 'stop', 'interrupt', 'pause',
    'resume', 'skip', 'history', 'last', 'details', 'title', 'reset', 'model',
    'use', 'file', 'image', 'meta', 'exif', 'metadata',
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
    selected_image_id: str | None = Field(default=None, title='Selected image artifact')


class ResSession(BaseModel):
    id: str
    title: str
    context: ContextId
    drafts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    last_generation_prompt: str | None = None
    last_generation_snapshot: dict[str, Any] | None = None
    selected_image_id: str | None = None
    created_at: str
    updated_at: str


class ResMessage(BaseModel):
    id: str
    role: Literal['system', 'user', 'assistant']
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ResChatArtifact(BaseModel):
    id: str
    message_id: str | None = None
    media_type: str
    display_name: str
    role: Literal['input', 'result']
    source_artifact_id: str | None = None
    source_deleted: bool = False
    created_at: str
    available: bool


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
    artifacts: list[ResChatArtifact] = Field(default_factory=list)
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
    artifacts: list[ResChatArtifact] = Field(default_factory=list)


def _public_artifacts(session_id: str) -> list[dict[str, Any]]:
    return [artifacts.public_artifact(item) for item in db.list_artifacts(session_id)]


def _log_chat_error(operation: str, exc: Exception | None = None) -> None:
    detail = f': {type(exc).__name__}: {exc}' if exc is not None else ''
    log.error(f'Chat: {operation}{detail}')


def _chat_error_result(result: dict[str, Any]) -> str:
    if (result.get('message_metadata') or {}).get('format') == 'html':
        return 'Imported file command failed'
    return result.get('text') or 'Unknown error'


_IMAGE_PAYLOAD_FIELDS = {'image', 'images', 'inputs', 'inits', 'mask', 'init_images', 'init_control', 'control', 'ip_adapter', 'face'}


def _safe_operation_values(values: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in values.items() if name not in _IMAGE_PAYLOAD_FIELDS}


def _store_image_results(session_id: str, message_id: str, images: list, saved_images: list, source_artifact_id: str | None) -> list[dict[str, Any]]:
    stored = []
    for index, image_data in enumerate(images):
        if not isinstance(image_data, str):
            continue
        saved_path = saved_images[index] if index < len(saved_images) and isinstance(saved_images[index], str) else None
        display_name = artifacts.safe_display_name(saved_path) if saved_path else f'generated-{index + 1}.png'
        try:
            artifact = artifacts.create_generated_artifact(
                session_id,
                message_id,
                image_data,
                display_name,
                saved_path=saved_path,
                source_artifact_id=source_artifact_id,
            )
            stored.append(artifact)
        except Exception as exc:
            _log_chat_error('image artifact could not be stored', exc)
    return stored


def get_sessions(limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    try:
        return db.list_sessions(limit=limit, offset=offset)
    except Exception as exc:
        _log_chat_error('session listing failed', exc)
        raise


def create_session(req: ReqSessionCreate):
    try:
        return db.create_session(title=req.title, context=req.context)
    except ValueError as exc:
        _log_chat_error('session creation failed', exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log_chat_error('session creation failed', exc)
        raise


def get_session(session_id: str):
    try:
        session = db.get_session(session_id)
    except Exception as exc:
        _log_chat_error('session lookup failed', exc)
        raise
    if session is None:
        _log_chat_error('session lookup failed: session not found')
        raise HTTPException(status_code=404, detail='Chat session not found')
    try:
        return ResSessionDetail(
            session=session,
            messages=db.list_messages(session_id),
            operations=db.list_operations(session_id),
            artifacts=_public_artifacts(session_id),
        )
    except Exception as exc:
        _log_chat_error('session detail retrieval failed', exc)
        raise


def restore_session(session_id: str):
    try:
        session = db.get_session(session_id)
    except Exception as exc:
        _log_chat_error('session restore failed', exc)
        raise
    if session is None:
        _log_chat_error('session restore failed: session not found')
        raise HTTPException(status_code=404, detail='Chat session not found')
    restore_errors = commands.restore_session_state(session_id)
    for item in _public_artifacts(session_id):
        if not item['available']:
            error = f'Image unavailable: {item["display_name"]}'
            restore_errors.append(error)
            _log_chat_error(f'session restore failed: {error}')
    return ResSessionDetail(
        session=db.get_session(session_id),
        messages=db.list_messages(session_id),
        operations=db.list_operations(session_id),
        artifacts=_public_artifacts(session_id),
        restore_errors=restore_errors,
    )


def _execute_chat_operation(session_id: str, user_message: dict[str, Any], result: dict[str, Any], api_instance):
    data = result.get('data') or {}
    descriptors = data.get('endpoints')
    if data.get('alias') != 'generate' or not isinstance(descriptors, list) or len(descriptors) != 1:
        raise ValueError('Invalid chat generation descriptor')
    operation_command = data.get('command', 'generate')
    if operation_command not in {'generate', 'again', 'run'}:
        raise ValueError('Invalid chat operation command')
    run_count = data.get('run_count', 1)
    if not isinstance(run_count, int) or not 1 <= run_count <= 100:
        raise ValueError('Invalid chat run count')
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
    if run_count > 1 and context != 'image':
        raise ValueError('Repeated runs are supported only in image context')
    arguments = descriptor.get('arguments')
    if not isinstance(arguments, dict):
        raise ValueError('Invalid chat generation arguments')

    operation = db.create_operation(
        session_id,
        operation_command,
        context,
        _safe_operation_values(arguments),
        message_id=user_message['id'],
    )
    operation_id = operation['id']
    try:
        if context == 'image':
            control_api_instance = getattr(api_instance, 'control', None)
            if control_api_instance is None:
                raise RuntimeError('The control generation API is unavailable')
            responses = [
                control_api_instance.post_control(api_control.ReqControl.model_validate(arguments))
                for _ in range(run_count)
            ]
            images = [image for response in responses for image in (response.images or [])]
            info = '\n'.join(response.info for response in responses if response.info)
            stats = '\n'.join(response.stats for response in responses if getattr(response, 'stats', ''))
            saved_images = [path for response in responses for path in (getattr(response, 'saved_images', []) or [])]
            parameters = _safe_operation_values(responses[-1].params or {})
            stored_artifacts = _store_image_results(
                session_id, user_message['id'], images, saved_images, session.get('selected_image_id'),
            )
            completion_text = f'Generated {len(images)} image(s)'
            if info:
                completion_text += f'\n{info}'
            if stats:
                completion_text += stats
            output_data = {
                'images': images,
                'info': info,
                'stats': stats,
                'parameters': parameters,
                'artifacts': [artifacts.public_artifact(item) for item in stored_artifacts],
            }
            saved_result = {
                'image_count': len(images),
                'run_count': run_count,
                'info': info,
                'stats': stats,
                'parameters': parameters,
            }
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
            source_artifact_id = session.get('selected_image_id')
            stored_artifacts = _store_image_results(
                session_id, user_message['id'], [response.image], [], source_artifact_id,
            )
            output_data = {
                'images': [response.image],
                'info': info,
                'artifacts': [artifacts.public_artifact(item) for item in stored_artifacts],
            }
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
                last_generation_snapshot=_safe_operation_values(arguments),
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
        _log_chat_error(f'{context} operation failed', exc)
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
        scheduler_arguments = {
            name: value for name, value in arguments.items()
            if name not in {'send_images', 'save_images'}
        }
        response = next(scheduler.enqueue_control_request(**scheduler_arguments))
        text = response[2]
    elif descriptor.get('path') == '/sdapi/v1/video':
        response = scheduler.enqueue_video_request('video', **arguments)
        text = response[4]
    else:
        raise ValueError('Scheduling is not available for this chat context')
    return {'kind': 'schedule', 'text': text, 'data': {}}


def _execute_import_endpoint(result: dict[str, Any]) -> dict[str, Any]:
    descriptors = (result.get('data') or {}).get('endpoints')
    if not isinstance(descriptors, list) or not descriptors:
        raise ValueError('Invalid imported endpoint command')
    from modules.api import gpu, models, server
    handlers = {
        '/sdapi/v1/version': (server.get_version, None),
        '/sdapi/v1/platform': (server.get_platform, None),
        '/sdapi/v1/torch': (server.get_torch, None),
        '/sdapi/v1/gpu': (gpu.get_gpu, None),
        '/sdapi/v1/gpu-smi': (gpu.get_gpu_smi, None),
        '/sdapi/v1/memory': (server.get_memory, None),
        '/sdapi/v1/status': (server.get_status, None),
        '/sdapi/v1/progress': (server.get_progress, models.ReqProgress),
        '/sdapi/v1/storage': (server.get_storage, models.ReqStorage),
    }
    output = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or descriptor.get('method') != 'GET':
            raise ValueError('Invalid imported endpoint descriptor')
        handler_spec = handlers.get(descriptor.get('path'))
        arguments = descriptor.get('arguments', {})
        if handler_spec is None or not isinstance(arguments, dict):
            raise ValueError('Unsupported imported endpoint')
        handler, request_model = handler_spec
        response = handler(req=request_model.model_validate(arguments)) if request_model else handler()
        content = json.dumps(jsonable_encoder(response), indent=2, ensure_ascii=False)
        if len(content) > 12000:
            content = f'{content[:12000]}\n[Output truncated]'
        output.append(f'{descriptor.get("alias", "endpoint")}:\n{content}')
    return {'kind': 'text', 'text': '\n\n'.join(output), 'data': {}}


def _import_staged_file(session_id: str, user_message: dict[str, Any], imported: dict[str, Any], api_instance, depth: int = 0):
    if imported['kind'] == 'image':
        try:
            artifact = artifacts.create_image_artifact(
                session_id, user_message['id'], imported['path'], imported['display_name'], role='input',
            )
            db.set_selected_image(session_id, artifact['id'])
        except Exception as exc:
            _log_chat_error('image import failed', exc)
            from pathlib import Path
            Path(imported['path']).unlink(missing_ok=True)
            raise ValueError('Image file could not be imported') from None
        public = artifacts.public_artifact(artifact)
        return {
            'kind': 'text',
            'text': f'Imported and selected image: {public["display_name"]}',
            'data': {'artifacts': [public]},
            'message_metadata': {'kind': 'text', 'data': {'artifacts': [public]}},
        }

    from pathlib import Path
    staged_path = Path(imported['path'])
    results = []
    failed = False
    progress_job = f'Chat file import: {imported["display_name"]}'
    owns_progress = not shared.state.job
    previous_textinfo = shared.state.textinfo
    progress_updates = []
    if owns_progress:
        shared.state.job = progress_job
    try:
        lines = imported['content'].splitlines()
        line_count = len(lines)
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            if owns_progress:
                if not shared.state.job:
                    shared.state.job = progress_job
                if shared.state.job == progress_job:
                    update = f'line {line_number}/{line_count}: {line}'
                    progress_updates.append(f'Processing... {update}' if not progress_updates else update)
                    shared.state.textinfo = '\n'.join(progress_updates)
            line_result = _execute_chat_command(
                session_id, user_message, line, None, api_instance, import_depth=depth + 1,
            )
            if line_result.get('kind') == 'endpoint':
                try:
                    line_result = _execute_import_endpoint(line_result)
                except Exception as exc:
                    line_result = {'kind': 'error', 'text': f'Endpoint command failed: {exc}', 'data': {}}
            line_text = str(line_result.get('text') or '')
            if line_result.get('kind') == 'error':
                failed = True
            line_metadata = line_result.get('message_metadata') or {}
            if line_metadata.get('format') == 'html':
                rendered_result = line_text
            else:
                rendered_result = escape(line_text)
            label = escape(f'{imported["display_name"]}:{line_number}')
            results.append(f'<div><span class="chat-file-line">{label}</span> {rendered_result}</div>')
            if line_result.get('kind') == 'error':
                break
    finally:
        staged_path.unlink(missing_ok=True)
        if owns_progress and shared.state.job == progress_job:
            shared.state.job = ''
            shared.state.textinfo = previous_textinfo
    if not results:
        label = escape(imported['display_name'])
        results.append(f'<div><span class="chat-file-line">{label}</span> No commands found</div>')
    public_artifacts = _public_artifacts(session_id)
    kind = 'error' if failed else 'text'
    return {
        'kind': kind,
        'text': ''.join(results),
        'data': {'artifacts': public_artifacts},
        'message_metadata': {'kind': kind, 'format': 'html', 'data': {'artifacts': public_artifacts}},
    }


def _execute_chat_command(
    session_id: str,
    user_message: dict[str, Any],
    content: str,
    image: str | None,
    api_instance,
    import_depth: int = 0,
):
    session = db.get_session(session_id)
    if session is not None:
        try:
            image = _selected_image_for_command(session, content, image)
        except ValueError as exc:
            return {'kind': 'error', 'text': str(exc), 'data': {}}
    result = commands.execute_command(session_id, content, image=image)
    if result.get('kind') == 'import':
        if import_depth >= 8:
            return {'kind': 'error', 'text': 'Nested file imports are too deep', 'data': {}}
        try:
            imported = artifacts.copy_allowed_file(result['data']['path'])
            return _import_staged_file(session_id, user_message, imported, api_instance, depth=import_depth)
        except ValueError as exc:
            return {'kind': 'error', 'text': str(exc), 'data': {}}
    if result.get('kind') == 'operation':
        return _execute_chat_operation(session_id, user_message, result, api_instance)
    if result.get('kind') == 'schedule':
        try:
            return _schedule_chat_operation(result)
        except Exception as exc:
            return {'kind': 'error', 'text': f'Scheduler error: {exc}', 'data': {}}
    return result


def _selected_image_for_command(session: dict[str, Any], content: str, fallback: str | None) -> str | None:
    try:
        command = commands.parse_command(content)
    except commands.CommandError:
        return fallback
    if command.name not in {'generate', 'again', 'schedule'}:
        return fallback
    artifact_id = session.get('selected_image_id')
    if not artifact_id:
        return fallback
    artifact = db.get_artifact(session['id'], artifact_id)
    if artifact is None:
        return None
    return artifacts.read_image_base64(artifact)


def _finish_submission(session_id: str, user_message: dict[str, Any], result: dict[str, Any]):
    try:
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
            artifacts=_public_artifacts(session_id),
        )
    except Exception as exc:
        _log_chat_error('message persistence failed', exc)
        raise


async def upload_chat_file(session_id: str, upload: UploadFile, api_instance=None):
    try:
        session = db.get_session(session_id)
    except Exception as exc:
        _log_chat_error('file upload session lookup failed', exc)
        raise
    if session is None:
        _log_chat_error('file upload failed: session not found')
        raise HTTPException(status_code=404, detail='Chat session not found')
    try:
        imported = await artifacts.stage_upload(upload)
    except ValueError as exc:
        _log_chat_error('file upload rejected', exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log_chat_error('file upload failed', exc)
        raise HTTPException(status_code=400, detail='File could not be imported') from None
    finally:
        await upload.close()
    try:
        user_message = db.create_message(
            session_id,
            'user',
            f'Imported file: {imported["display_name"]}',
            metadata={'source': 'file-upload'},
        )
    except Exception as exc:
        _log_chat_error('file upload message creation failed', exc)
        raise
    try:
        result = _import_staged_file(session_id, user_message, imported, api_instance)
    except Exception as exc:
        _log_chat_error('imported file processing failed', exc)
        result = {'kind': 'error', 'text': 'File could not be imported', 'data': {}}
        staged_path = imported.get('path')
        if imported['kind'] == 'image' and staged_path:
            from pathlib import Path
            Path(staged_path).unlink(missing_ok=True)
    else:
        if result.get('kind') == 'error':
            _log_chat_error(f'file processing failed: {_chat_error_result(result)}')
    return _finish_submission(session_id, user_message, result)


def post_message(session_id: str, req: ReqChatMessage, api_instance=None):
    try:
        session = db.get_session(session_id)
    except Exception as exc:
        _log_chat_error('message session lookup failed', exc)
        raise
    if session is None:
        _log_chat_error('message submission failed: session not found')
        raise HTTPException(status_code=404, detail='Chat session not found')
    content = req.content.strip()
    if not content:
        _log_chat_error('message submission failed: message is empty')
        raise HTTPException(status_code=422, detail='Chat message cannot be empty')
    command_word = content.split(maxsplit=1)[0].casefold()
    is_private_path_command = command_word in {'use', 'file'}
    try:
        if session['title'] == 'New chat' and command_word != 'title' and not is_private_path_command and not db.list_messages(session_id, limit=1):
            title = content.splitlines()[0][:64]
            db.update_session(session_id, title=title)
        transcript_content = 'Server file import' if is_private_path_command else content
        user_message = db.create_message(session_id, 'user', transcript_content)
    except Exception as exc:
        _log_chat_error('message persistence failed', exc)
        raise
    try:
        image = _selected_image_for_command(session, content, req.image)
        result = _execute_chat_command(session_id, user_message, content, image, api_instance)
    except Exception as exc:  # Keep a command failure in the persisted conversation.
        _log_chat_error('command failed', exc)
        result = {'kind': 'error', 'text': 'Command failed. Check the server log.', 'data': {}}
    else:
        if result.get('kind') == 'error':
            _log_chat_error(f'command failed: {_chat_error_result(result)}')
    return _finish_submission(session_id, user_message, result)


def get_chat_artifact_content(session_id: str, artifact_id: str):
    try:
        artifact = db.get_artifact(session_id, artifact_id)
    except Exception as exc:
        _log_chat_error('image content lookup failed', exc)
        raise
    if artifact is None or not artifacts.public_artifact(artifact)['available']:
        _log_chat_error('image content request failed: artifact not found or unavailable')
        raise HTTPException(status_code=404, detail='Chat image artifact not found or unavailable')
    try:
        return FileResponse(
            artifact['storage_ref'],
            media_type=artifact['media_type'],
            filename=artifact['display_name'],
        )
    except Exception as exc:
        _log_chat_error('image content request failed', exc)
        raise


def _get_available_chat_image(session_id: str, artifact_id: str):
    try:
        artifact = db.get_artifact(session_id, artifact_id)
    except Exception as exc:
        _log_chat_error('image operation lookup failed', exc)
        raise
    if artifact is None or not artifact['media_type'].startswith('image/') \
            or not artifacts.public_artifact(artifact)['available']:
        _log_chat_error('image operation failed: artifact not found or unavailable')
        raise HTTPException(status_code=404, detail='Chat image artifact not found or unavailable')
    return artifact


def open_chat_artifact_folder(session_id: str, artifact_id: str):
    artifact = _get_available_chat_image(session_id, artifact_id)
    import os
    import platform
    import shutil
    import subprocess
    from pathlib import Path
    if shared.cmd_opts.hide_ui_dir_config:
        return {'opened': False}
    folder = Path(artifact['storage_ref']).parent
    if not folder.is_dir():
        _log_chat_error(f'image folder open failed: folder unavailable: {folder}')
        raise HTTPException(status_code=404, detail='Chat image folder is unavailable')
    system = platform.system()
    try:
        if system == 'Windows':
            os.startfile(str(folder))  # pylint: disable=no-member
        elif system == 'Darwin':
            subprocess.Popen(['open', str(folder)])  # pylint: disable=consider-using-with
        elif 'microsoft-standard-WSL2' in platform.uname().release:
            opener = 'wslview' if shutil.which('wslview') is not None else 'wsl-open'
            subprocess.Popen([opener, str(folder)])  # pylint: disable=consider-using-with
        else:
            subprocess.Popen(['xdg-open', str(folder)])  # pylint: disable=consider-using-with
    except Exception as exc:
        _log_chat_error('image folder open failed', exc)
        raise HTTPException(status_code=500, detail='Chat image folder could not be opened') from exc
    return {'opened': True}


def save_chat_artifact(session_id: str, artifact_id: str):
    artifact = _get_available_chat_image(session_id, artifact_id)
    from types import SimpleNamespace
    from PIL import Image
    from modules import images, infotext, paths
    try:
        with Image.open(artifact['storage_ref']) as image:
            image.load()
            info, _items = images.read_info_from_image(image)
            params = infotext.parse(info) if info else {}
            prompt = params.get('Prompt') or ''
            negative_prompt = params.get('Negative prompt') or ''
            sampler = params.get('Sampler') or ''
            try:
                seed = int(params.get('Seed') or 0)
            except (TypeError, ValueError):
                seed = 0
            try:
                steps = int(params.get('Steps') or 0)
            except (TypeError, ValueError):
                steps = 0
            styles = params.get('Styles') or []
            if isinstance(styles, str):
                styles = [style.strip() for style in styles.split(',')]
            save_params = SimpleNamespace(
                prompt=prompt,
                negative_prompt=negative_prompt,
                sampler=sampler,
                sampler_name=sampler,
                seed=seed,
                steps=steps,
                cfg_scale=params.get('CFG scale') or 0,
                clip_skip=params.get('Clip skip') or 1,
                denoising_strength=params.get('Denoising strength') or 0,
                width=image.width,
                height=image.height,
                styles=styles,
                n_iter=1,
                batch_size=1,
                iteration=0,
                batch_index=0,
                all_seeds=[seed],
                all_prompts=[prompt],
                all_negative_prompts=[negative_prompt],
                index_of_first_image=0,
            )
            saved_path, _text_path, _exif = images.save_image(
                image,
                path=paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save),
                seed=seed,
                prompt=prompt,
                info=info,
                extension=shared.opts.samples_format,
                p=save_params,
            )
    except Exception as exc:
        _log_chat_error('image save failed', exc)
        raise HTTPException(status_code=500, detail='Chat image could not be saved') from exc
    if not saved_path:
        _log_chat_error('image save failed: no output file was created')
        raise HTTPException(status_code=500, detail='Chat image could not be saved')
    return {'saved': True}


def delete_chat_artifact(session_id: str, artifact_id: str):
    try:
        artifact = db.get_artifact(session_id, artifact_id)
    except Exception as exc:
        _log_chat_error('image deletion lookup failed', exc)
        raise
    if artifact is None or not artifact['media_type'].startswith('image/'):
        _log_chat_error('image deletion failed: artifact not found')
        raise HTTPException(status_code=404, detail='Chat image artifact not found')
    try:
        artifacts.delete_storage_file(artifact)
        db.delete_artifact(session_id, artifact_id)
        log.warning(f'Chat: Delete: image="{artifact["storage_ref"]}"')
    except ValueError as exc:
        _log_chat_error(f'Delete: image="{artifact["storage_ref"]}"', exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log_chat_error(f'image deletion failed: {artifact["display_name"]}', exc)
        raise
    return {'deleted': True, 'id': artifact_id}


def patch_message(session_id: str, message_id: str, req: ReqChatMessageUpdate):
    if req.metadata.get('kind') != 'endpoint':
        _log_chat_error('message update rejected: only endpoint results can be updated')
        raise HTTPException(status_code=400, detail='Only endpoint results can update an assistant message')
    try:
        message = db.update_message(session_id, message_id, req.content, req.metadata)
    except ValueError as exc:
        _log_chat_error('message update failed', exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log_chat_error('message update failed', exc)
        raise
    if message is None:
        _log_chat_error('message update failed: endpoint response not found')
        raise HTTPException(status_code=404, detail='Chat endpoint response message not found')
    return message


def update_session(session_id: str, req: ReqSessionUpdate):
    updates = req.model_dump(exclude_unset=True)
    if 'selected_image_id' in updates:
        selected_image_id = updates['selected_image_id']
        try:
            session = db.get_session(session_id)
        except Exception as exc:
            _log_chat_error('session update lookup failed', exc)
            raise
        if session is None:
            _log_chat_error('session update failed: session not found')
            raise HTTPException(status_code=404, detail='Chat session not found')
        if selected_image_id is not None:
            try:
                artifact = db.get_artifact(session_id, selected_image_id)
            except Exception as exc:
                _log_chat_error('session image selection lookup failed', exc)
                raise
            if artifact is None or not artifact['media_type'].startswith('image/') \
                    or not artifacts.public_artifact(artifact)['available']:
                _log_chat_error('session update failed: selected image is unavailable')
                raise HTTPException(status_code=400, detail='Selected image is not available in this session')
    try:
        session = db.update_session(session_id, **updates)
    except ValueError as exc:
        _log_chat_error('session update failed', exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log_chat_error('session update failed', exc)
        raise
    if session is None:
        _log_chat_error('session update failed: session not found')
        raise HTTPException(status_code=404, detail='Chat session not found')
    return session


def delete_session(session_id: str):
    try:
        deleted = db.delete_session(session_id)
    except Exception as exc:
        _log_chat_error('session deletion failed', exc)
        raise
    if not deleted:
        _log_chat_error('session deletion failed: session not found')
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

    async def upload_chat_file_route(session_id: str, file: UploadFile = File(...)):
        return await upload_chat_file(session_id, file, control_api_instance)

    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/messages', post_message_route, methods=['POST'], response_model=ResChatSubmission, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/files', upload_chat_file_route, methods=['POST'], response_model=ResChatSubmission, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/messages/{message_id}', patch_message, methods=['PATCH'], response_model=ResMessage, tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}/content', get_chat_artifact_content, methods=['GET'], tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}/open-folder', open_chat_artifact_folder, methods=['POST'], tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}/save', save_chat_artifact, methods=['POST'], tags=['Chat'])
    api.add_api_route('/sdapi/v1/chat/sessions/{session_id}/artifacts/{artifact_id}', delete_chat_artifact, methods=['DELETE'], tags=['Chat'])
    return True
