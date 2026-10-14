"""Task serialization, lifecycle management, and background execution."""

import atexit
import datetime as dt
import inspect
import io
import mimetypes
import os
import tempfile
import threading
from contextlib import contextmanager, nullcontext

from modules import scheduler_db
from modules.logger import log


_worker = None
_worker_start_failed = False
_worker_stop = threading.Event()
_worker_lock = threading.Lock()
_worker_poll_interval = 1.0
_worker_idle_interval = 5.0
_task_max_attempts = 3
_task_retry_delay = 3
_model_option_keys = ('sd_model_checkpoint', 'sd_model_refiner', 'sd_unet', 'sd_unet_secondary', 'sd_vae', 'sd_text_encoder')
_file_media_fields = {'condition_video', 'condition_files', 'reference_media', 'inputs', 'inits'}
debug = os.environ.get('SD_SCHEDULER_DEBUG', None) is not None
debug_log = log.trace if debug else lambda *args, **kwargs: None


def _file_details(value):
    if isinstance(value, dict):
        path = value.get('path') or value.get('name')
        filename = value.get('orig_name') or value.get('filename') or value.get('name')
        media_type = value.get('mime_type') or value.get('media_type')
    elif isinstance(value, (str, os.PathLike)):
        path = value
        filename = value
        media_type = None
    else:
        path = getattr(value, 'path', None) or getattr(value, 'name', None)
        filename = getattr(value, 'orig_name', None) or getattr(value, 'filename', None) or getattr(value, 'name', None)
        media_type = getattr(value, 'mime_type', None) or getattr(value, 'media_type', None)
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
        return None
    filename = os.path.basename(os.fspath(filename or path))
    media_type = media_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    return os.fspath(path), filename, media_type


def _encode_file(value, image_blobs):
    details = _file_details(value)
    if details is None:
        return None
    path, filename, media_type = details
    with open(path, 'rb') as source:
        data = source.read()
    media_id = len(image_blobs)
    image_blobs.append(data)
    return {'__sd_scheduler_type__': 'file', 'id': media_id, 'filename': filename, 'media_type': media_type}


def _encode_value(value, image_blobs, field=None):
    from PIL import Image

    if isinstance(value, Image.Image):
        stream = io.BytesIO()
        value.save(stream, format='PNG')
        image_id = len(image_blobs)
        image_blobs.append(stream.getvalue())
        return {'__sd_scheduler_type__': 'image', 'id': image_id}
    if field in _file_media_fields:
        encoded_file = _encode_file(value, image_blobs)
        if encoded_file is not None:
            return encoded_file
    if isinstance(value, (bytes, bytearray, memoryview)):
        image_id = len(image_blobs)
        image_blobs.append(bytes(value))
        return {'__sd_scheduler_type__': 'blob', 'id': image_id}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _encode_value(item, image_blobs, field=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item, image_blobs, field=field) for item in value]
    if value.__class__.__module__ == 'modules.control.unit' and value.__class__.__name__ == 'Unit':
        names = ('index', 'enabled', 'strength', 'type', 'start', 'end', 'mode', 'model_name', 'process_name', 'factor', 'guess', 'attention', 'fidelity', 'query_weight', 'adain_weight', 'process_params', 'tile')
        fields = {name: _encode_value(getattr(value, name, None), image_blobs) for name in names}
        fields['override'] = _encode_value(getattr(value, 'override', None), image_blobs)
        return {'__sd_scheduler_type__': 'unit', 'fields': fields}
    raise TypeError(f'Unsupported scheduler parameter type: {type(value).__name__}')


def _decode_value(value, image_blobs, media_dir=None):
    if isinstance(value, list):
        return [_decode_value(item, image_blobs, media_dir=media_dir) for item in value]
    if isinstance(value, dict):
        marker = value.get('__sd_scheduler_type__')
        if marker == 'image':
            from PIL import Image
            return Image.open(io.BytesIO(image_blobs[value['id']])).copy()
        if marker == 'blob':
            return image_blobs[value['id']]
        if marker == 'file':
            data = image_blobs[value['id']]
            if media_dir is None:
                return data
            filename = os.path.basename(value.get('filename') or f'media-{value["id"]}')
            path = os.path.join(media_dir, f'{value["id"]}-{filename}')
            with open(path, 'wb') as target:
                target.write(data)
            return path
        if marker == 'unit':
            from modules.control import unit
            fields = {key: _decode_value(item, image_blobs, media_dir=media_dir) for key, item in value['fields'].items()}
            instance = unit.Unit(
                index=fields.get('index'),
                enabled=fields.get('enabled'),
                strength=fields.get('strength'),
                unit_type=fields.get('type'),
                start=fields.get('start', 0),
                end=fields.get('end', 1),
            )
            for key, item in fields.items():
                if key not in ('type', 'override'):
                    setattr(instance, key, item)
            instance.override = fields.get('override')
            if instance.process is not None:
                instance.process.override = instance.override
            return instance
        return {key: _decode_value(item, image_blobs, media_dir=media_dir) for key, item in value.items()}
    return value


def _video_handler(surface):
    if surface == 'video':
        from modules.video_models.video_run import generate
        return generate
    if surface == 'minimax':
        from modules.minimax.minimax_video import generate
        return generate
    if surface == 'ltx':
        from modules.ltx.ltx_process import run_ltx
        return run_ltx
    if surface == 'framepack':
        from modules.framepack.framepack_wrappers import run_framepack
        return run_framepack
    raise ValueError(f'Unsupported scheduler Video surface: {surface}')


def _bound_arguments(function, args, kwargs):
    bound = inspect.signature(function).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _capture_request(args, kwargs):
    from modules.control.run import control_run

    named = _bound_arguments(control_run, args, kwargs)
    blobs = []
    parameters = {'control': _encode_value(named, blobs)}
    return parameters, blobs


def _contains_image(value):
    if isinstance(value, list):
        return any(_contains_image(item) for item in value)
    if isinstance(value, dict):
        marker = value.get('__sd_scheduler_type__')
        if marker == 'image' or (marker == 'file' and str(value.get('media_type', '')).startswith('image/')):
            return True
        return any(_contains_image(item) for item in value.values())
    return False


def _has_control_unit(named):
    unit_type = named.get('unit_type')
    units = named.get('units')
    if not unit_type or not isinstance(units, list):
        return False
    for unit_value in units:
        if not isinstance(unit_value, dict) or unit_value.get('__sd_scheduler_type__') != 'unit':
            continue
        fields = unit_value.get('fields', {})
        if not fields.get('enabled') or fields.get('type') != unit_type:
            continue
        if unit_type == 'reference':
            return True
        if str(fields.get('model_name') or '').strip().lower() not in ('', 'none'):
            return True
    return False


def _task_mode(entry_point, named):
    if entry_point == 'video':
        return 'video'
    if _has_control_unit(named):
        return 'control'
    return 'img2img' if _contains_image(named) else 'txt2img'


def _capture_model_options(parameters):
    from modules import shared
    if shared.sd_loaded:
        model = shared.sd_model
        checkpoint_info = getattr(model, 'sd_checkpoint_info', None)
        parameters['model'] = (getattr(checkpoint_info, 'name', None) or getattr(model, 'sd_model_checkpoint', None)) if model is not None else None
    else:
        parameters['model'] = shared.opts.sd_model_checkpoint
    for key in _model_option_keys:
        parameters[key] = getattr(shared.opts, key)


def enqueue_control_request(*args, **kwargs):
    if scheduler_stopped():
        return _queued_response(None, 'Scheduler is stopped')
    try:
        parameters, image_blobs = _capture_request(args, kwargs)
        _capture_model_options(parameters)
        mode = _task_mode('control', parameters['control'])
        task_id = scheduler_db.create_task(parameters, image_blobs=image_blobs, max_attempts=_task_max_attempts, mode=mode)
    except Exception as exc:
        log.error(f'Scheduler: unable to record Control task: {exc}')
        return _queued_response(None, str(exc))
    log.info(f'Scheduler: queued task=Control id={task_id}')
    debug_log(f'Scheduler: captured task id={task_id} fields={len(parameters["control"])} images={len(image_blobs)}')
    return _queued_response(task_id)


def enqueue_video_request(surface, *args, **kwargs):
    if scheduler_stopped():
        return _video_queued_response(None, surface, 'Scheduler is stopped')
    try:
        handler = _video_handler(surface)
        named = _bound_arguments(handler, args, kwargs)
        image_blobs = []
        encoded = _encode_value(named, image_blobs)
        parameters = {'video': {'surface': surface, 'arguments': encoded}}
        model_name = named.get('model') or named.get('variant')
        if model_name:
            parameters['model'] = str(model_name)
        _capture_model_options(parameters)
        parameters['model'] = str(model_name or parameters['model'])
        task_id = scheduler_db.create_task(parameters, image_blobs=image_blobs, max_attempts=_task_max_attempts, mode=_task_mode('video', named))
        log.info(f'Scheduler: queued task=Video surface={surface} id={task_id}')
        debug_log(f'Scheduler: captured video task id={task_id} surface={surface} fields={len(named)} media={len(image_blobs)}')
        return _video_queued_response(task_id, surface)
    except Exception as exc:
        log.error(f'Scheduler: unable to record Video task surface={surface}: {exc}')
        return _video_queued_response(None, surface, str(exc))


def _queued_response(task_id, error=None):
    message = f'Queued scheduler task {task_id}' if task_id is not None else f'Scheduler error: {error}'

    def response():
        yield (None, None, message, None)

    return response()


def _video_queued_response(task_id, surface, error=None):
    output_count, message_index = {'video': (5, 4), 'minimax': (2, 1), 'ltx': (2, 1), 'framepack': (3, 2)}.get(surface, (5, 4))
    values = [None] * output_count
    values[message_index] = f'Queued scheduler task {task_id}' if task_id is not None else f'Scheduler error: {error}'
    return values


def _result_paths(result):
    if not isinstance(result, tuple) or len(result) == 0:
        return []
    paths = []
    images = result[0] if isinstance(result[0], list) else [result[0]]
    for image in images:
        filename = getattr(image, 'already_saved_as', None)
        if filename:
            paths.append(os.fspath(filename))
    if len(result) > 3 and isinstance(result[3], str) and result[3]:
        paths.append(result[3])
    return list(dict.fromkeys(paths))


def _call_arguments(function, named):
    args = []
    kwargs = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if parameter.name in named:
                args.append(named[parameter.name])
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            args.extend(named.get(parameter.name, ()))
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.name in named:
            kwargs[parameter.name] = named[parameter.name]
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs.update(named.get(parameter.name, {}))
    return args, kwargs


def _consume_generator(result, on_yield=None):
    if not inspect.isgenerator(result):
        return result
    latest = None
    while True:
        try:
            latest = next(result)
        except StopIteration as stop:
            return stop.value if stop.value is not None else latest
        if on_yield is not None:
            on_yield(latest)


def _video_result_paths(surface, result):
    paths = []
    if surface == 'video' and isinstance(result, tuple):
        images = result[0] if result and isinstance(result[0], list) else []
        for image in images:
            filename = getattr(image, 'already_saved_as', None)
            if filename:
                paths.append(os.fspath(filename))
        video_path = result[1] if len(result) > 1 else None
        if isinstance(video_path, str) and video_path:
            paths.append(video_path)
    elif isinstance(result, tuple) and result and isinstance(result[0], str) and result[0]:
        paths.append(result[0])
    return list(dict.fromkeys(paths))


def _video_result_error(surface, result):
    if result is None:
        return 'Video handler returned no result'
    if isinstance(result, (list, tuple)):
        status_index = {'video': 4, 'minimax': 1, 'ltx': 1, 'framepack': 2}.get(surface, len(result) - 1)
        values = [result[status_index]] if status_index < len(result) else []
        if surface in ('video', 'minimax') and result and result[0] is None and len(result) > 1:
            values.append(result[1])
    else:
        values = [result]
    for value in values:
        if isinstance(value, str) and any(marker in value.lower() for marker in ('error:', 'not loaded', 'not selected', 'not found', 'no frames', 'no output', 'not installed', 'load failed')):
            return value
    return None


@contextmanager
def _temporary_model_options(parameters):
    from modules import shared
    values = {key: shared.opts.get(key) for key in _model_option_keys}
    _samples_save = shared.opts.samples_save
    try:
        if isinstance(parameters, dict):
            for key in _model_option_keys:
                if key in parameters:
                    shared.opts.set(key, parameters[key])
        yield
    finally:
        shared.opts.samples_save = _samples_save
        for key, value in values.items():
            shared.opts.set(key, value)


def _run_task(task, lock):
    from modules import call_queue
    with _temporary_model_options(task['parameters']):
        ctx = call_queue.get_lock() if lock else nullcontext()
        with ctx:
            _execute_task(task)


def _execute_task(task):
    if task.get('mode', 'control') == 'video':
        _execute_video_task(task)
        return
    _execute_control_task(task)


def _execute_control_task(task):
    blobs = task['image_blobs']
    parameters = task['parameters']
    from modules.control.run import control_run
    from modules import timer

    with tempfile.TemporaryDirectory(prefix=f'sdnext-scheduler-{task["id"]}-') as media_dir:
        signature = inspect.signature(control_run)
        named = _decode_value(parameters['control'], blobs, media_dir=media_dir)
        args = []
        kwargs = {}
        for parameter in signature.parameters.values():
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                if parameter.name in named:
                    args.append(named[parameter.name])
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                args.extend(named.get(parameter.name, ()))
            elif parameter.kind == inspect.Parameter.KEYWORD_ONLY and parameter.name in named:
                kwargs[parameter.name] = named[parameter.name]
        iterator = iter(control_run(*args, **kwargs))
        result = None
        while True:
            try:
                result = next(iterator)
            except StopIteration as stop:
                if stop.value is not None:
                    result = stop.value
                break
        if isinstance(result, str):
            raise RuntimeError(result)
        if isinstance(result, tuple):
            message = result[2] if len(result) > 2 and isinstance(result[2], str) else ''
            video_path = result[3] if len(result) > 3 and isinstance(result[3], str) else ''
            if 'Error:' in message or 'Exception:' in message or video_path.startswith('Error:') or not result[0]:
                raise RuntimeError(message or video_path or 'generation returned no output images')
        infotexts = result[4] if isinstance(result, tuple) and len(result) > 4 and isinstance(result[4], list) else []
        if not scheduler_db.update_results(task['id'], _result_paths(result), process_stats=timer.process.dct(), infotexts=infotexts):
            raise RuntimeError(f'Scheduler task {task["id"]} result could not be stored')
    log.info(f'Scheduler: completed task id={task["id"]}')


def _execute_video_task(task):
    from modules import timer

    video_request = task['parameters']['video']
    surface = video_request['surface']
    handler = _video_handler(surface)
    with tempfile.TemporaryDirectory(prefix=f'sdnext-scheduler-{task["id"]}-') as media_dir:
        named = _decode_value(video_request['arguments'], task['image_blobs'], media_dir=media_dir)
        named['task_id'] = f'task(scheduler-{task["id"]})'
        args, kwargs = _call_arguments(handler, named)
        timer.process.reset()
        result = handler(*args, **kwargs)
        streamed_paths = []
        if surface == 'framepack':
            def record_output_path(output):
                streamed_paths.extend(_video_result_paths(surface, output))
        else:
            record_output_path = None
        result = _consume_generator(result, record_output_path)
        error = _video_result_error(surface, result)
        if error:
            raise RuntimeError(error)
        paths = _video_result_paths(surface, result)
        if surface == 'framepack':
            paths.extend(streamed_paths)
            paths = list(dict.fromkeys(paths))
        summary = []
        if isinstance(result, tuple):
            summary = [value for value in result if isinstance(value, str) and value and value not in paths]
        if not scheduler_db.update_results(task['id'], paths, process_stats=timer.process.dct(), infotexts=summary[-1:]):
            raise RuntimeError(f'Scheduler task {task["id"]} result could not be stored')
    log.info(f'Scheduler: completed Video task id={task["id"]} surface={surface}')


def run_once():
    from modules import call_queue
    if call_queue.is_locked():
        return False
    task = scheduler_db.claim_next_task(force_only=not scheduler_enabled())
    if task is None:
        return False
    log.debug(f'Scheduler: claimed task id={task["id"]} attempt={task["attempts"]}')
    try:
        task['image_blobs'] = scheduler_db.get_task(task['id'], include_images=True)['image_blobs']
        if task.get('mode', 'control') == 'video':
            surface = task.get('parameters', {}).get('video', {}).get('surface')
            _run_task(task, lock=surface == 'video') # ltx/framepack/minimax acquire their own lock
        else:
            _run_task(task, lock=True)
    except Exception as exc:
        scheduler_db.retry_or_fail(task['id'], exc, retry_delay=_task_retry_delay)
        log.error(f'Scheduler: task failed id={task["id"]} attempt={task["attempts"]}: {exc}')
    return True


def _worker_wait_interval():
    from modules import call_queue
    return _worker_poll_interval if call_queue.is_locked() else _worker_idle_interval


def _worker_loop():
    if hasattr(os, 'setpriority') and hasattr(os, 'PRIO_PROCESS') and hasattr(threading, 'get_native_id'):
        try:
            thread_id = threading.get_native_id()
            priority = os.getpriority(os.PRIO_PROCESS, thread_id)
            os.setpriority(os.PRIO_PROCESS, thread_id, min(19, priority + 5))
            priority = os.getpriority(os.PRIO_PROCESS, thread_id)
            debug_log(f'Scheduler: thread={thread_id} priority={priority}')
        except OSError as exc:
            debug_log(f'Scheduler: thread priority adjustment unavailable: {exc}')
    while not _worker_stop.is_set():
        scheduler_db.change_event.clear()
        try:
            did_work = run_once()
        except Exception as exc:
            log.error(f'Scheduler: worker error: {exc}')
            _worker_stop.wait(_worker_poll_interval)
            continue
        if did_work:
            _worker_stop.wait(0.1)
        else:
            scheduler_db.change_event.wait(_worker_wait_interval())


def start_worker(diag: bool = False):
    log.debug('Initializing: scheduler')
    global _worker, _worker_start_failed # pylint: disable=global-statement
    from modules import shared
    with _worker_lock:
        if scheduler_stopped():
            log.debug('Scheduler stopped: worker initialization skipped')
            return None
        if _worker_start_failed:
            return None
        if _worker is not None and _worker.is_alive():
            return _worker
        scheduler_db.configure(shared.opts.scheduler_dir)
        try:
            scheduler_db.initialize()
        except Exception as e:
            _worker_start_failed = True
            log.error('Scheduler disabled: database initialization failed')
            from modules.errors import display
            display(e, 'Scheduler')
            return None
        if diag:
            try:
                stats, opts = scheduler_db.diagnostics()
                log.info(f'Scheduler compile options: {opts}')
                log.info(f'Scheduler database diagnostics: {stats}')
                if not stats['ok']:
                    log.error('Scheduler: database diagnostics found integrity or schema issues')
            except Exception as exc:
                log.error(f'Scheduler: database diagnostics failed: {exc}')
        _worker_stop.clear()
        _worker = threading.Thread(target=_worker_loop, name='SDNextScheduler', daemon=True)
        _worker.start()
        atexit.register(stop_worker)
        return _worker


def stop_worker(timeout=5):
    _worker_stop.set()
    scheduler_db.change_event.set()
    if _worker is not None and _worker.is_alive() and threading.current_thread() is not _worker:
        _worker.join(timeout=max(0, timeout))


def queue_existing_task(task_id, immediate=False):
    queued = scheduler_db.queue_task(task_id, immediate=immediate)
    if queued:
        action = 'execution=immediate' if immediate else 'execution=queued'
        log.info(f'Scheduler: {action} id={task_id}')
    return queued


def move_task(task_id, target_id, position='before'):
    moved = scheduler_db.move_task(task_id, target_id, position)
    if moved:
        log.info(f'Scheduler move: task id={task_id} position={position} target={target_id}')
    return moved


def pause_all():
    set_scheduler_state('Paused')
    log.info('Scheduler: pause')


def resume_all():
    set_scheduler_state('Active')
    log.info('Scheduler: resume')


def scheduler_enabled(now=None):
    from modules import shared
    state = shared.opts.scheduler_state
    if state == 'Active':
        return True
    if state == 'Paused':
        return False
    if state != 'Auto':
        return False
    local_time = (now or dt.datetime.now().astimezone()).strftime('%H:%M')
    start = shared.opts.scheduler_start
    end = shared.opts.scheduler_end
    if start == end:
        return True
    if start < end:
        return start <= local_time < end
    return local_time >= start or local_time < end


def scheduler_stopped():
    from modules import shared
    return getattr(shared.opts, 'scheduler_state', None) == 'Stopped'


def set_scheduler_state(state=None, start=None, end=None):
    from modules import shared
    updates = {}
    if state is not None:
        if state not in ('Active', 'Paused', 'Auto', 'Stopped'):
            raise ValueError(f'Invalid scheduler state: {state}')
        updates['scheduler_state'] = state
    for key, value in (('scheduler_start', start), ('scheduler_end', end)):
        if value is not None:
            try:
                value = dt.time.fromisoformat(value).strftime('%H:%M')
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid scheduler time for {key}: {value}') from exc
            updates[key] = value
    for key, value in updates.items():
        shared.opts.set(key, value)
    if updates:
        shared.opts.save()
        log.info(f'Scheduler state: {updates}')
        if updates.get('scheduler_state') == 'Stopped':
            stop_worker()
    return get_status()


def list_tasks(**kwargs):
    tasks = scheduler_db.list_tasks(**kwargs)
    debug_log(f'Scheduler list: count={len(tasks)} status={kwargs.get("status")}')
    return tasks


def search_tasks(**kwargs):
    tasks = scheduler_db.search_tasks(**kwargs)
    debug_log(f'Scheduler search: count={len(tasks)} filters={list(kwargs.keys())}')
    return tasks


def get_task(task_id, **kwargs):
    task = scheduler_db.get_task(task_id, **kwargs)
    debug_log(f'Scheduler fetch: id={task_id} found={task is not None}')
    return task


def get_task_image(task_id, image_index):
    image = scheduler_db.get_task_image(task_id, image_index)
    debug_log(f'Scheduler fetch image: task_id={task_id} index={image_index} found={image is not None}')
    return image


def get_status():
    from modules import shared
    state = shared.opts.scheduler_state
    status = {
        'state': state,
        'start': shared.opts.scheduler_start,
        'end': shared.opts.scheduler_end,
        'window_open': scheduler_enabled(),
        'paused': state == 'Paused',
        'worker_running': _worker is not None and _worker.is_alive(),
    }
    debug_log(f'Scheduler status: state={state} window_open={status["window_open"]} worker_running={status["worker_running"]}')
    return status


def update_task_parameters(task_id, parameters, db_path=None):
    return update_task_parameters_with_media(task_id, parameters, media_replacements=None, db_path=db_path)


def _contains_media_marker(value):
    if isinstance(value, list):
        return any(_contains_media_marker(item) for item in value)
    if isinstance(value, dict):
        if value.get('__sd_scheduler_type__') in ('image', 'file', 'blob'):
            return True
        return any(_contains_media_marker(item) for item in value.values())
    return False


def update_task_parameters_with_media(task_id, parameters, media_replacements=None, db_path=None, media_metadata=None):
    task = scheduler_db.get_task(task_id, include_images=media_replacements is not None, db_path=db_path)
    if task is None or not isinstance(parameters, dict):
        return False
    named = task['parameters'].get('control')
    if isinstance(named, dict):
        immutable = {'state', 'scheduler', 'units', 'inputs', 'inits', 'mask', 'input_script_args', 'override_script_args'}
        scoped = named
        model_updates = {key: value for key, value in parameters.items() if key in _model_option_keys}
        updates = {key: value for key, value in parameters.items() if key in named}
        if len(updates) + len(model_updates) != len(parameters) or any(key in immutable for key in updates):
            return False
        if any(not isinstance(value, str) for value in model_updates.values()):
            return False
        string_fields = {'prompt', 'negative_prompt'}
        integer_fields = {'steps', 'width_before', 'height_before'}
        if any(not isinstance(value, str) for key, value in updates.items() if key in string_fields):
            return False
        if any(type(value) is not int for key, value in updates.items() if key in integer_fields):
            return False
    else:
        video = task['parameters'].get('video')
        if task.get('mode') != 'video' or not isinstance(video, dict) or not isinstance(video.get('arguments'), dict):
            return False
        scoped = video['arguments']
        model_updates = {key: value for key, value in parameters.items() if key in _model_option_keys}
        updates = {key: value for key, value in parameters.items() if key in scoped}
        if len(updates) + len(model_updates) != len(parameters):
            return False
        if any(key in ('task_id', '_ui_state') or _contains_media_marker(scoped[key]) for key in updates):
            return False
        if any(type(value) is not type(scoped[key]) for key, value in updates.items() if scoped[key] is not None):
            return False
        if any(not isinstance(value, str) for value in model_updates.values()):
            return False

    replacements = media_replacements or {}
    if replacements:
        if not isinstance(replacements, dict) or any(type(index) is not int or not isinstance(blob, bytes) for index, blob in replacements.items()):
            return False
        if media_metadata is not None and (
                not isinstance(media_metadata, dict)
                or set(media_metadata) != set(replacements)
                or any(
                    not isinstance(metadata, dict)
                    or not isinstance(metadata.get('filename'), str)
                    or not isinstance(metadata.get('media_type'), str)
                    for metadata in media_metadata.values()
                )
        ):
            return False
        media_ids = set()

        def collect_media_ids(value):
            if isinstance(value, list):
                for item in value:
                    collect_media_ids(item)
            elif isinstance(value, dict):
                marker = value.get('__sd_scheduler_type__')
                if marker in ('image', 'file', 'blob'):
                    media_ids.add(value.get('id'))
                else:
                    for item in value.values():
                        collect_media_ids(item)

        collect_media_ids(task['parameters'])
        if not set(replacements).issubset(media_ids):
            return False
        for index, blob in replacements.items():
            if index < 0 or index >= len(task['image_blobs']):
                return False
            task['image_blobs'][index] = blob
        if media_metadata:
            def update_media_references(value):
                if isinstance(value, list):
                    for item in value:
                        update_media_references(item)
                elif isinstance(value, dict):
                    marker = value.get('__sd_scheduler_type__')
                    media_id = value.get('id')
                    if marker in ('image', 'file', 'blob') and media_id in media_metadata:
                        value.update(media_metadata[media_id])
                    else:
                        for item in value.values():
                            update_media_references(item)

            update_media_references(task['parameters'])

    scoped.update(updates)
    task['parameters'].update(model_updates)
    if task.get('mode') == 'video' and 'model' in updates:
        task['parameters']['model'] = str(updates['model'])
    update_kwargs = {'db_path': db_path}
    if media_replacements is not None:
        update_kwargs['image_blobs'] = task['image_blobs']
    updated = scheduler_db.update_parameters(task_id, task['parameters'], **update_kwargs)
    if updated:
        log.info(f'Scheduler update: id={task_id} fields={list(parameters.keys())} media={len(replacements)}')
    return updated


def delete_task(task_id):
    deleted = scheduler_db.delete_task(task_id)
    if deleted:
        log.info(f'Scheduler delete: id={task_id}')
    return deleted


def prune_tasks(before):
    count = scheduler_db.prune_tasks(before)
    log.info(f'Scheduler prune: count={count} before={before}')
    return count


def pause_task(task_id):
    paused = scheduler_db.pause_task(task_id)
    if paused:
        log.info(f'Scheduler pause: id={task_id}')
    return paused


def resume_task(task_id, scheduled_at=None):
    resumed = scheduler_db.resume_task(task_id, scheduled_at=scheduled_at)
    if resumed:
        log.info(f'Scheduler resume: id={task_id} scheduled_at={scheduled_at}')
    return resumed


def reschedule_task(task_id, scheduled_at):
    rescheduled = scheduler_db.reschedule_task(task_id, scheduled_at)
    if rescheduled:
        log.info(f'Scheduler reschedule: id={task_id} scheduled_at={scheduled_at}')
    return rescheduled
