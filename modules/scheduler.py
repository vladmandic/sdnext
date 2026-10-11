"""Task serialization, lifecycle management, and background execution."""

import atexit
import datetime
import inspect
import io
import os
import threading
from contextlib import contextmanager

from modules import scheduler_db
from modules.logger import log


_worker = None
_worker_start_failed = False
_worker_stop = threading.Event()
_worker_lock = threading.Lock()
_worker_poll_interval = 1.0
_model_option_keys = ('sd_model_checkpoint', 'sd_model_refiner', 'sd_unet', 'sd_unet_secondary', 'sd_vae', 'sd_text_encoder')
debug = os.environ.get('SD_SCHEDULER_DEBUG', None) is not None
debug_log = log.trace if debug else lambda *args, **kwargs: None


def _encode_value(value, image_blobs):
    from PIL import Image

    if isinstance(value, Image.Image):
        stream = io.BytesIO()
        value.save(stream, format='PNG')
        image_id = len(image_blobs)
        image_blobs.append(stream.getvalue())
        return {'__sd_scheduler_type__': 'image', 'id': image_id}
    if isinstance(value, (bytes, bytearray, memoryview)):
        image_id = len(image_blobs)
        image_blobs.append(bytes(value))
        return {'__sd_scheduler_type__': 'blob', 'id': image_id}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _encode_value(item, image_blobs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item, image_blobs) for item in value]
    if value.__class__.__module__ == 'modules.control.unit' and value.__class__.__name__ == 'Unit':
        names = ('index', 'enabled', 'strength', 'type', 'start', 'end', 'mode', 'model_name', 'process_name', 'factor', 'guess', 'attention', 'fidelity', 'query_weight', 'adain_weight', 'process_params', 'tile')
        fields = {name: _encode_value(getattr(value, name, None), image_blobs) for name in names}
        fields['override'] = _encode_value(getattr(value, 'override', None), image_blobs)
        return {'__sd_scheduler_type__': 'unit', 'fields': fields}
    raise TypeError(f'Unsupported scheduler parameter type: {type(value).__name__}')


def _decode_value(value, image_blobs):
    if isinstance(value, list):
        return [_decode_value(item, image_blobs) for item in value]
    if isinstance(value, dict):
        marker = value.get('__sd_scheduler_type__')
        if marker == 'image':
            from PIL import Image
            return Image.open(io.BytesIO(image_blobs[value['id']])).copy()
        if marker == 'blob':
            return image_blobs[value['id']]
        if marker == 'unit':
            from modules.control import unit
            fields = {key: _decode_value(item, image_blobs) for key, item in value['fields'].items()}
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
        return {key: _decode_value(item, image_blobs) for key, item in value.items()}
    return value


def _capture_request(args, kwargs):
    from modules.control.run import control_run

    bound = inspect.signature(control_run).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    blobs = []
    parameters = {'control': _encode_value(dict(bound.arguments), blobs)}
    return parameters, blobs


def enqueue_control_request(*args, **kwargs):
    try:
        parameters, image_blobs = _capture_request(args, kwargs)
        from modules import shared
        if shared.sd_loaded:
            model = shared.sd_model
            checkpoint_info = getattr(model, 'sd_checkpoint_info', None)
            parameters['model'] = (getattr(checkpoint_info, 'name', None) or getattr(model, 'sd_model_checkpoint', None)) if model is not None else None
        else:
            parameters['model'] = shared.opts.sd_model_checkpoint
        for key in _model_option_keys:
            parameters[key] = getattr(shared.opts, key)
        task_id = scheduler_db.create_task(parameters, image_blobs=image_blobs)
    except Exception as exc:
        log.error(f'Scheduler: unable to record Control task: {exc}')
        return _queued_response(None, str(exc))
    log.info(f'Scheduler: queued task=Control id={task_id}')
    debug_log(f'Scheduler: captured task id={task_id} fields={len(parameters["control"])} images={len(image_blobs)}')
    return _queued_response(task_id)


def _queued_response(task_id, error=None):
    message = f'Queued scheduler task {task_id}' if task_id is not None else f'Scheduler error: {error}'

    def response():
        yield (None, None, message, None)

    return response()


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


@contextmanager
def _temporary_model_options(parameters):
    from modules import shared
    values = {key: getattr(shared.opts, key) for key in _model_option_keys}
    _samples_save = shared.opts.samples_save
    try:
        if isinstance(parameters, dict):
            for key in _model_option_keys:
                if key in parameters:
                    setattr(shared.opts, key, parameters[key])
        yield
    finally:
        shared.opts.samples_save = _samples_save
        for key, value in values.items():
            setattr(shared.opts, key, value)


def _run_task(task):
    with _temporary_model_options(task['parameters']):
        _execute_task(task)


def _execute_task(task):
    blobs = task['image_blobs']
    parameters = task['parameters']
    from modules.control.run import control_run
    from modules import timer

    signature = inspect.signature(control_run)
    named = _decode_value(parameters['control'], blobs)
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
        with call_queue.get_lock():
            _run_task(task)
    except Exception as exc:
        delay = min(3600, 60 * (2 ** max(0, task['attempts'] - 1)))
        scheduler_db.retry_or_fail(task['id'], exc, retry_delay=delay)
        log.error(f'Scheduler: task failed id={task["id"]} attempt={task["attempts"]}: {exc}')
    return True


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
        try:
            did_work = run_once()
        except Exception as exc:
            log.error(f'Scheduler: worker error: {exc}')
            did_work = False
        _worker_stop.wait(0.1 if did_work else _worker_poll_interval)


def start_worker(diag: bool = False):
    log.debug('Initializing: scheduler')
    global _worker, _worker_start_failed # pylint: disable=global-statement
    from modules import shared
    with _worker_lock:
        if _worker_start_failed:
            return None
        if _worker is not None and _worker.is_alive():
            return _worker
        scheduler_db.configure(shared.opts.scheduler_dir)
        try:
            scheduler_db.initialize()
        except Exception as exc:
            _worker_start_failed = True
            log.error(f'Scheduler: database initialization failed; scheduler disabled until restart: {exc}')
            return None
        if diag:
            try:
                stats = scheduler_db.diagnostics()
                log.info(f'Scheduler: database diagnostics {stats}')
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
    if _worker is not None and _worker.is_alive() and threading.current_thread() is not _worker:
        _worker.join(timeout=max(0, timeout))


def queue_existing_task(task_id, immediate=False):
    queued = scheduler_db.queue_task(task_id, immediate=immediate)
    if queued:
        action = 'requested immediate execution for' if immediate else 'queued existing task'
        log.info(f'Scheduler: {action} id={task_id}')
    return queued


def move_task(task_id, target_id, position='before'):
    moved = scheduler_db.move_task(task_id, target_id, position)
    if moved:
        log.info(f'Scheduler: moved task id={task_id} {position} task_id={target_id}')
    return moved


def pause_all():
    set_scheduler_state('Paused')
    log.info('Scheduler: paused globally')


def resume_all():
    set_scheduler_state('Active')
    log.info('Scheduler: resumed globally')


def scheduler_enabled(now=None):
    from modules import shared
    state = shared.opts.scheduler_state
    if state == 'Active':
        return True
    if state == 'Paused':
        return False
    if state != 'Auto':
        return False
    local_time = (now or datetime.datetime.now().astimezone()).strftime('%H:%M')
    start = shared.opts.scheduler_start
    end = shared.opts.scheduler_end
    if start == end:
        return True
    if start < end:
        return start <= local_time < end
    return local_time >= start or local_time < end


def set_scheduler_state(state=None, start=None, end=None):
    from modules import shared
    updates = {}
    if state is not None:
        if state not in ('Active', 'Paused', 'Auto'):
            raise ValueError(f'Invalid scheduler state: {state}')
        updates['scheduler_state'] = state
    for key, value in (('scheduler_start', start), ('scheduler_end', end)):
        if value is not None:
            try:
                value = datetime.time.fromisoformat(value).strftime('%H:%M')
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid scheduler time for {key}: {value}') from exc
            updates[key] = value
    for key, value in updates.items():
        shared.opts.set(key, value)
    if updates:
        shared.opts.save()
        log.info(f'Scheduler: updated settings {updates}')
    return get_status()


def list_tasks(**kwargs):
    tasks = scheduler_db.list_tasks(**kwargs)
    debug_log(f'Scheduler: listed tasks count={len(tasks)} status={kwargs.get("status")}')
    return tasks


def search_tasks(**kwargs):
    tasks = scheduler_db.search_tasks(**kwargs)
    debug_log(f'Scheduler: searched tasks count={len(tasks)} filters={list(kwargs.keys())}')
    return tasks


def get_task(task_id, **kwargs):
    task = scheduler_db.get_task(task_id, **kwargs)
    debug_log(f'Scheduler: fetched task id={task_id} found={task is not None}')
    return task


def get_task_image(task_id, image_index):
    image = scheduler_db.get_task_image(task_id, image_index)
    debug_log(f'Scheduler: fetched input image task_id={task_id} index={image_index} found={image is not None}')
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
    debug_log(f'Scheduler: status state={state} window_open={status["window_open"]} worker_running={status["worker_running"]}')
    return status


def update_task_parameters(task_id, parameters, db_path=None):
    task = scheduler_db.get_task(task_id, db_path=db_path)
    if task is None or not isinstance(parameters, dict):
        return False
    named = task['parameters'].get('control')
    if not isinstance(named, dict):
        return False
    immutable = {'state', 'scheduler', 'units', 'inputs', 'inits', 'mask', 'input_script_args', 'override_script_args'}
    control_updates = {key: value for key, value in parameters.items() if key in named}
    model_updates = {key: value for key, value in parameters.items() if key in _model_option_keys}
    if len(control_updates) + len(model_updates) != len(parameters):
        return False
    if any(key in immutable for key in control_updates):
        return False
    if any(not isinstance(value, str) for value in model_updates.values()):
        return False
    string_fields = {'prompt', 'negative_prompt'}
    integer_fields = {'steps', 'width_before', 'height_before'}
    if any(not isinstance(value, str) for key, value in control_updates.items() if key in string_fields):
        return False
    if any(type(value) is not int for key, value in control_updates.items() if key in integer_fields):
        return False
    named.update(control_updates)
    task['parameters'].update(model_updates)
    updated = scheduler_db.update_parameters(task_id, task['parameters'], db_path=db_path)
    if updated:
        log.info(f'Scheduler: updated task parameters id={task_id} fields={list(parameters.keys())}')
    return updated


def delete_task(task_id):
    deleted = scheduler_db.delete_task(task_id)
    if deleted:
        log.info(f'Scheduler: deleted task id={task_id}')
    return deleted


def prune_tasks(before):
    count = scheduler_db.prune_tasks(before)
    log.info(f'Scheduler: pruned tasks count={count} before={before}')
    return count


def pause_task(task_id):
    paused = scheduler_db.pause_task(task_id)
    if paused:
        log.info(f'Scheduler: paused task id={task_id}')
    return paused


def resume_task(task_id, scheduled_at=None):
    resumed = scheduler_db.resume_task(task_id, scheduled_at=scheduled_at)
    if resumed:
        log.info(f'Scheduler: resumed task id={task_id} scheduled_at={scheduled_at}')
    return resumed


def reschedule_task(task_id, scheduled_at):
    rescheduled = scheduler_db.reschedule_task(task_id, scheduled_at)
    if rescheduled:
        log.info(f'Scheduler: rescheduled task id={task_id} scheduled_at={scheduled_at}')
    return rescheduled
