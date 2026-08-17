import math
from modules.logger import log, get_console


installed = False
status = {'session': None, 'pending': None, 'reported': set()}
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
    text = ','.join(str(k) for k in key) if isinstance(key, (tuple, list)) else str(key)
    return text if len(text) <= 64 else text[:61] + '...'


def start_progress(name: str, total: int):
    """Console bar for the sweep, matching how model and file loading report elsewhere. Returns
    (progress, task) or (None, None) when there is no console to draw on."""
    console = get_console()
    if console is None:
        return None, None
    import rich.progress as rp
    progress = rp.Progress(rp.TextColumn('[cyan]Autotune'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[yellow]{task.description}'), console=console, transient=True)
    task = progress.add_task(description=f'kernel={name}', total=total)
    progress.start()
    return progress, task


def stop_progress(session):
    progress = session.get('progress', None) if session is not None else None
    task = session.get('task', None) if session is not None else None
    if progress is not None:
        try:
            progress.remove_task(task)
            progress.stop()
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        session['progress'] = None


def bench_hook(orig):
    def wrapped(self, *args, config, **meta):
        from modules import shared
        """
        if shared.state.textinfo != 'Autotune kernel':
            _textinfo = shared.state.textinfo
            shared.state.textinfo = 'Autotune kernel'
        else:
            _textinfo = None
        """
        try:
            session = status['session']
            if session is None or session['owner'] is not self:
                stop_progress(session) # a sweep that never reported completion must not leave its bar drawing
                prev = session['prev'] if session is not None else shared.state.textinfo # chained sweeps inherit the pre-tuning text, so an abandoned one cannot leave its own label behind
                progress, task = start_progress(kernel_name(getattr(self, 'base_fn', None)), len(self.configs))
                session = {'owner': self, 'count': 0, 'total': len(self.configs), 'name': kernel_name(getattr(self, 'base_fn', None)), 'prev': prev, 'compile_us': 0, 'progress': progress, 'task': task}
                status['session'] = session
            session['count'] += 1
            if session['progress'] is not None and session['task'] is not None:
                session['progress'].update(session['task'], completed=session['count'], description=f'kernel={session["name"]} {session["count"]}/{session["total"]}')
                if session["count"] >= session["total"]:
                    stop_progress(session)
        except Exception as e:
            log.debug(f'Kernel autotune: report error: {e}')
        res = orig(self, *args, config=config, **meta)
        """
        if _textinfo is not None:
            shared.state.textinfo = _textinfo
        """
        return res
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
        res = orig(self, *args, **kwargs)
        pending = status.get('pending', None)
        if pending is not None:
            status['pending'] = None
            name, config = pending
            try:
                regs, spills = getattr(res, 'n_regs', None), getattr(res, 'n_spills', None)
                if spills:
                    seen = f'{name}:{config}:{spills}'
                    if seen not in status['reported']: # the same kernel tunes once per shape, so warn on each distinct config only
                        status['reported'].add(seen)
                        # the sweep line carries the config too, but it is debug and filtered out at default level
                        log.warning(f'Kernel autotune: kernel={name} register spill config="{config}" regs={regs} spills={spills}')
                elif regs is not None:
                    log.debug(f'Kernel autotune: kernel={name} regs={regs} spills=0')
            except Exception as e:
                log.debug(f'Kernel autotune: report error: {e}')
        return res
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
        Autotuner.run = run_hook(Autotuner.run)
        # log.debug('Kernel autotune: reporting installed')
    except Exception as e:
        log.warning(f'Kernel autotune: reporting install failed: {e}')
