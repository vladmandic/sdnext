import os
import sys
import html
import threading
import time
import cProfile
from modules import shared, progress, errors, timer
from modules.logger import log


queue_lock = threading.Lock()
debug = os.environ.get('SD_QUEUE_DEBUG', None) is not None


def get_lock():
    if debug:
        fn = f'{sys._getframe(3).f_code.co_name}:{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Queue: fn={fn} lock={queue_lock.locked()}')
    return queue_lock


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with get_lock():
            res = func(*args, **kwargs)
        return res
    return f


def wrap_gradio_gpu_call(func, extra_outputs=None, name=None):
    name = name or func.__name__
    def f(*args, **kwargs):
        # if the first argument is a string that says "task(...)", it is treated as a job id
        if len(args) > 0 and type(args[0]) == str and args[0][0:5] == "task(" and args[0][-1] == ")":
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None
        with get_lock():
            progress.start_task(id_task)
            try:
                res = func(*args, **kwargs)
                progress.record_results(id_task, res)
            except Exception as e:
                log.error(f"Exception: {e}")
                log.error(f"Arguments: args={str(args)[:10240]} kwargs={str(kwargs)[:10240]}")
                errors.display(e, 'gradio call')
                res = extra_outputs or []
                res.append(f"<div class='error'>{html.escape(str(e))}</div>")
            finally:
                progress.finish_task(id_task)
        return res
    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True, name=name)


def wrap_gradio_call(func, extra_outputs=None, add_stats=False, name=None):
    job_name = name if name is not None else func.__name__
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        t = time.perf_counter()
        shared.mem_mon.reset()
        if len(args) > 0 and type(args[0]) == str and args[0][0:5] == "task(" and args[0][-1] == ")":
            task_id = args[0]
        else:
            task_id = 0
        jobid = shared.state.begin(job_name, task_id=task_id)
        try:
            if shared.cmd_opts.profile:
                pr = cProfile.Profile()
                pr.enable()
            res = func(*args, **kwargs)
            if res is None:
                msg = "No result returned from function"
                log.warning(msg)
                res = extra_outputs_array or []
                res.append(f"<div class='error'>{html.escape(msg)}</div>")
            else:
                res = list(res)
            if shared.cmd_opts.profile:
                pr.disable()
                errors.profile(pr, 'Wrap')
        except Exception as e:
            errors.display(e, 'gradio call')
            res = extra_outputs_array or []
            res.append(f"<div class='error'>{html.escape(type(e).__name__+': '+str(e))}</div>")
        shared.state.end(jobid)
        if not add_stats:
            return tuple(res)
        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_m}m {elapsed_s:.2f}s" if elapsed_m > 0 else f"{elapsed_s:.2f}s"
        summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
        memory = shared.mem_mon.summary()
        if isinstance(res, list) and isinstance(res[-1], str):
            res[-1] += f"<div class='performance'><p>Time: {elapsed_text} | {summary} {memory}</p></div>"
        return tuple(res)
    return f
