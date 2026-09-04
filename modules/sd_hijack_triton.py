import os
import time
import math
from typing import Any
from modules.logger import log, get_console
from modules.timer import autotune


installed = False
status: dict[str, Any] = {'session': None, 'pending': None, 'reported': set(), 'run_id': 0}
slow_compile_seconds = 1.0


def config_key(config):
    """Compact form of a tuned config: the block shape is what separates a good candidate from one
    that cannot fit its accumulator in registers."""
    if config is None:
        return 'unknown'
    kwargs = getattr(config, 'kwargs', None) or {}
    parts = [f'{k.replace("BLOCK_SIZE_", "B").replace("GROUP_SIZE_", "G")}={v}' for k, v in kwargs.items()]
    parts.append(f'warps={getattr(config, "num_warps", "?")}')
    parts.append(f'stages={getattr(config, "num_stages", "?")}')
    return ','.join(parts)


def timing_spread(configs_timings):
    """(ratio, unusable) over a sweep: how many times slower the worst candidate was than the best,
    and how many could not run at all. A large ratio means the config list holds candidates the
    hardware cannot execute well, which costs the sweep far more than it costs the chosen kernel."""
    values = []
    unusable = 0
    for timing in (configs_timings or {}).values():
        value = timing[0] if isinstance(timing, (list, tuple)) and len(timing) > 0 else timing
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            values.append(value)
        else:
            unusable += 1
    if len(values) < 2:
        return None, unusable
    return max(values) / min(values), unusable


def kernel_name(fn):
    while fn is not None and not hasattr(fn, '__name__'):
        fn = getattr(fn, 'fn', None)
    return getattr(fn, '__name__', 'unknown')


def shape_key(key):
    """Compact form of the autotune cache key, which is what separates one sweep from the next:
    successive sweeps of the same kernel are different shapes, not a repeating loop."""
    if key is None:
        return 'unknown'
    if isinstance(key, (tuple, list)):
        parts = []
        for k in key:
            if isinstance(k, str) and k.startswith('torch.'):
                continue
            parts.append(str(k))
        if len(parts) > 6:
            text = '...' + ','.join(parts[-6:])
        else:
            text = ','.join(parts)
    else:
        text = str(key)
    return text if len(text) <= 64 else text[:61] + '...'


def start_progress(name: str, total: int, shape: str | None = None):
    """Console bar for the sweep, matching how model and file loading report elsewhere. Returns
    (progress, task) or (None, None) when there is no console to draw on."""
    console = get_console()
    if console is None:
        return None, None
    import rich.progress as rp
    progress = rp.Progress(
        rp.TextColumn('[cyan]Autotune'),
        rp.BarColumn(),
        rp.TaskProgressColumn(),
        rp.TextColumn('[green]{task.completed}/{task.total}'),
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
        rp.TextColumn('[yellow]{task.description}'),
        rp.TextColumn('[blue]{task.fields[shape]}'),
        console=console,
        transient=True,
    )
    task = progress.add_task(description=f'kernel={name}', total=total, shape=(f'shape={shape}' if shape is not None else ''))
    progress.ts = time.time()
    progress.kernel = name
    progress.start()
    return progress, task


def stop_progress(session):
    progress = session.get('progress', None) if session is not None else None
    task = session.get('task', None) if session is not None else None
    if progress is not None:
        try:
            progress.stop()
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        try:
            if task is not None:
                autotune.add(progress.kernel, time.time() - progress.ts)
                autotune.add('_shapes', 1)
                progress.remove_task(task)
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        session['progress'] = None


def bench_hook(orig):
    def wrapped(self, *args, config, **meta):
        try:
            session: dict[str, Any] | None = status.get('session', None)
            current_run_id = status.get('run_id')
            if session is not None and session.get('owner') is self and session.get('run_id') == current_run_id:
                session['count'] += 1
                if session['count'] > session['total']:
                    session['total'] = session['count']
                if session['progress'] is not None and session['task'] is not None:
                    session['progress'].update(session['task'], completed=session['count'], description=f'kernel={session["name"]}')
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        return orig(self, *args, config=config, **meta)
    return wrapped


def make_autotune_listener(prior):
    def listener(*, fn=None, key=None, best_config=None, configs_timings=None, duration=None, cache_hit=False, **kwargs):
        try:
            session = status['session']
            if session is not None:
                stop_progress(session)
                status['session'] = None
            name = kernel_name(fn)
            shape = shape_key(key)
            if cache_hit:
                log.debug(f'Kernel autotune: kernel={name} shape={shape} cached')
            else:
                compile_s = (session['compile_us'] / 1e6) if session is not None else 0
                ratio, unusable = timing_spread(configs_timings)
                spread = f' spread={ratio:.0f}x' if ratio is not None else ''
                rejected = f' unusable={unusable}' if unusable else ''
                log.debug(f'Kernel autotune: kernel={name} shape={shape} time={duration or 0:.2f} compile={compile_s:.2f} best="{config_key(best_config)}"{spread}{rejected}')
                status['pending'] = (name, config_key(best_config)) # the chosen kernel is only loaded once run() returns, so its register use is reported there
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        if prior is not None:
            prior(fn=fn, key=key, best_config=best_config, configs_timings=configs_timings, duration=duration, cache_hit=cache_hit, **kwargs)
    return listener


def run_hook(orig):
    """Report the register use of the config a sweep just chose. n_regs and n_spills are filled in
    when the driver loads the binary, so they exist only once the kernel has run, not at compile."""
    def wrapped(self, *args, **kwargs):
        session = status['session']
        if session is not None:
            stop_progress(session)
            status['session'] = None
        status['run_id'] = status.get('run_id', 0) + 1
        run_id = status['run_id']

        try:
            self.nargs = dict(zip(self.arg_names, args))
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = tuple(_args[k] for k in self.keys if k in _args)
            for arg in _args.values():
                if hasattr(arg, 'dtype'):
                    key += (str(arg.dtype),)
            needs_benchmark = len(self.configs) > 1 and key not in self.cache
            skip_autotune = os.environ.get('SD_SKIP_AUTOTUNE', None) is not None
            if needs_benchmark and skip_autotune:
                self.cache[key] = self.configs[0] # pre-seed the cache so orig() takes its cache-hit path and skips the sweep
                needs_benchmark = False
            if needs_benchmark:
                try:
                    total = len(self.prune_configs(kwargs))
                except Exception:
                    total = len(self.configs)
                shape = shape_key(key)
                progress, task = start_progress(kernel_name(getattr(self, 'base_fn', None)), total, shape=shape)
                session = {
                    'owner': self,
                    'run_id': run_id,
                    'count': 0,
                    'total': total,
                    'name': kernel_name(getattr(self, 'base_fn', None)),
                    'shape': shape,
                    'compile_us': 0,
                    'progress': progress,
                    'task': task,
                }
                status['session'] = session
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')

        res = None
        try:
            res = orig(self, *args, **kwargs)
            return res
        finally:
            pending = status.get('pending', None)
            if pending is not None and res is not None:
                status['pending'] = None
                name, config = pending
                try:
                    regs, spills = getattr(res, 'n_regs', None), getattr(res, 'n_spills', None)
                    if spills:
                        seen = f'{name}:{config}:{spills}'
                        if seen not in status['reported']:
                            status['reported'].add(seen)
                            log.warning(f'Kernel autotune: kernel={name} register spill config="{config}" regs={regs} spills={spills}')
                    elif regs is not None:
                        log.debug(f'Kernel autotune: kernel={name} regs={regs} spills=0')
                except Exception as e:
                    log.debug(f'Kernel autotune: report error: {e}')
            session = status.get('session', None)
            if session is not None and session.get('owner') is self and session.get('run_id') == run_id:
                stop_progress(session)
                status['session'] = None
    return wrapped


def make_compile_listener(prior):
    def listener(*, src=None, metadata=None, metadata_group=None, times=None, cache_hit=False, **kwargs):
        try:
            if not cache_hit and times is not None:
                total_s = getattr(times, 'total', 0) / 1e6
                session = status['session']
                if session is not None:
                    session['compile_us'] += getattr(times, 'total', 0)
                elif total_s >= slow_compile_seconds:
                    name = kernel_name(getattr(src, 'fn', None))
                    if name == 'unknown' and isinstance(metadata, dict):
                        name = str(metadata.get('name', 'unknown'))
                    log.debug(f'Kernel compile: kernel={name} time={total_s:.2f}')
        except Exception as e:
            log.debug(f'Kernel compile: report error: {e}')
        if prior is not None:
            prior(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit, **kwargs)
    return listener


def install():
    """Report triton kernel autotuning and slow compiles in the log and in live progress text.

    Autotune sweeps and kernel compiles run inside the first forward pass at a
    new shape and can take minutes; without reporting they are indistinguishable
    from slow inference. Uses the triton knobs listeners for completion events
    and wraps the per-candidate benchmark for the live signal.
    """
    global installed # pylint: disable=global-statement
    if installed:
        return
    installed = True
    try:
        from triton import knobs
        from triton.runtime.autotuner import Autotuner
    except Exception as e:
        log.debug(f'Kernel autotune: {e}')
        return
    try:
        knobs.autotuning.listener = make_autotune_listener(getattr(knobs.autotuning, 'listener', None)) # this is not actually invoked by triton
        knobs.compilation.listener = make_compile_listener(getattr(knobs.compilation, 'listener', None))
        Autotuner._bench = bench_hook(Autotuner._bench) # pylint: disable=protected-access
        Autotuner.run = run_hook(Autotuner.run)
        # log.debug('Kernel autotune: reporting installed')
    except Exception as e:
        log.warning(f'Kernel autotune: reporting install failed: {e}')
