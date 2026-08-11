from modules.logger import log, get_console


installed = False
status = {'session': None}
slow_compile_seconds = 1.0


def kernel_name(fn):
    while fn is not None and not hasattr(fn, '__name__'):
        fn = getattr(fn, 'fn', None)
    return getattr(fn, '__name__', 'unknown')


def shape_key(key):
    """Compact form of the autotune cache key, which is what separates one sweep from the next:
    successive sweeps of the same kernel are different shapes, not a repeating loop."""
    if key is None:
        return 'unknown'
    text = ','.join(str(k) for k in key) if isinstance(key, (tuple, list)) else str(key)
    return text if len(text) <= 64 else text[:61] + '...'


def start_progress(name: str, total: int):
    """Console bar for the sweep, matching how model and file loading report elsewhere. Returns
    (progress, task) or (None, None) when there is no console to draw on."""
    console = get_console()
    if console is None:
        return None, None
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    progress = Progress(TextColumn('[cyan]{task.description}'), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), TimeElapsedColumn(), console=console, transient=True)
    task = progress.add_task(description=f'Autotune: kernel={name}', total=total)
    progress.start()
    return progress, task


def stop_progress(session):
    progress = session.get('progress', None) if session is not None else None
    if progress is not None:
        try:
            progress.stop()
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        session['progress'] = None


def bench_hook(orig):
    def wrapped(self, *args, config, **meta):
        try:
            from modules import shared
            session = status['session']
            if session is None or session['owner'] is not self:
                stop_progress(session) # a sweep that never reported completion must not leave its bar drawing
                prev = session['prev'] if session is not None else shared.state.textinfo # chained sweeps inherit the pre-tuning text, so an abandoned one cannot leave its own label behind
                progress, task = start_progress(kernel_name(getattr(self, 'base_fn', None)), len(self.configs))
                session = {'owner': self, 'count': 0, 'total': len(self.configs), 'name': kernel_name(getattr(self, 'base_fn', None)), 'prev': prev, 'compile_us': 0, 'progress': progress, 'task': task}
                status['session'] = session
            session['count'] += 1
            if session['progress'] is not None:
                session['progress'].update(session['task'], completed=session['count'])
            shared.state.textinfo = f"Tuning kernel {session['name']} {session['count']}/{session['total']}"
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        return orig(self, *args, config=config, **meta)
    return wrapped


def make_autotune_listener(prior):
    def listener(*, fn=None, key=None, best_config=None, configs_timings=None, duration=None, cache_hit=False, **kwargs):
        try:
            from modules import shared
            session = status['session']
            if session is not None:
                stop_progress(session)
                shared.state.textinfo = session['prev']
                status['session'] = None
            name = kernel_name(fn)
            shape = shape_key(key)
            if cache_hit:
                log.debug(f'Kernel autotune: kernel={name} shape={shape} cached')
            else:
                compile_s = (session['compile_us'] / 1e6) if session is not None else 0
                log.debug(f'Kernel autotune: kernel={name} shape={shape} time={duration or 0:.2f} compile={compile_s:.2f}')
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        if prior is not None:
            prior(fn=fn, key=key, best_config=best_config, configs_timings=configs_timings, duration=duration, cache_hit=cache_hit, **kwargs)
    return listener


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
                    log.info(f'Kernel compile: kernel={name} time={total_s:.2f}')
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
        log.debug(f'Kernel autotune: reporting unavailable: {e}')
        return
    try:
        knobs.autotuning.listener = make_autotune_listener(getattr(knobs.autotuning, 'listener', None))
        knobs.compilation.listener = make_compile_listener(getattr(knobs.compilation, 'listener', None))
        Autotuner._bench = bench_hook(Autotuner._bench) # pylint: disable=protected-access
        # log.debug('Kernel autotune: reporting installed')
    except Exception as e:
        log.warning(f'Kernel autotune: reporting install failed: {e}')
