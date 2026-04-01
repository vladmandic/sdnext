import logging
import warnings
from modules.logger import get_log, get_console, setup_logging, install_traceback
from modules.errorlimiter import ErrorLimiterAbort


log = get_log()
setup_logging()
install_traceback()
already_displayed = {}


def install(suppress=None):
    if suppress is None:
        suppress = []
    warnings.filterwarnings("ignore", category=UserWarning)
    install_traceback(suppress=suppress)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(levelname)s | %(pathname)s | %(message)s')


def display(e: Exception, task: str, suppress=None):
    if suppress is None:
        suppress = []
    if isinstance(e, ErrorLimiterAbort):
        return
    log.critical(f"{task or 'error'}: {type(e).__name__}")
    """
    trace = traceback.format_exc()
    log.error(trace)
    for line in traceback.format_tb(e.__traceback__):
        log.error(repr(line))
    console = get_console()
    console.print_exception(show_locals=False, max_frames=16, extra_lines=1, suppress=suppress, theme="ansi_dark", word_wrap=False, width=console.width)
    """
    log.traceback(e, suppress=suppress)


def display_once(e: Exception, task):
    if task in already_displayed:
        return
    display(e, task)
    already_displayed[task] = 1


def run(code, task: str):
    try:
        code()
    except Exception as e:
        display(e, task)


def exception(suppress=None):
    if suppress is None:
        suppress = []
    console = get_console()
    console.print_exception(show_locals=False, max_frames=16, extra_lines=2, suppress=suppress, theme="ansi_dark", word_wrap=False, width=min([console.width, 200]))


def profile(profiler, msg: str, n: int = 16):
    profiler.disable()
    import io
    import pstats
    stream = io.StringIO() # pylint: disable=abstract-class-instantiated
    p = pstats.Stats(profiler, stream=stream)
    p.sort_stats(pstats.SortKey.CUMULATIVE)
    p.print_stats(200)
    # p.print_title()
    # p.print_call_heading(10, 'time')
    # p.print_callees(10)
    # p.print_callers(10)
    profiler = None
    lines = stream.getvalue().split('\n')
    lines = [x for x in lines if '<frozen' not in x
             and '{built-in' not in x
             and '/logging' not in x
             and 'Ordered by' not in x
             and 'List reduced' not in x
             and '_lsprof' not in x
             and '/profiler' not in x
             and 'rich' not in x
             and 'profile_torch' not in x
             and x.strip() != ''
            ]
    txt = '\n'.join(lines[:min(n, len(lines))])
    log.debug(f'Profile {msg}: {txt}')


def profile_torch(profiler, msg: str):
    profiler.stop()
    lines = profiler.key_averages().table(sort_by="cpu_time_total", row_limit=12)
    lines = lines.split('\n')
    lines = [x for x in lines if '/profiler' not in x and '---' not in x]
    txt = '\n'.join(lines)
    log.debug(f'Torch profile CPU-total {msg}: \n{txt}')
    lines = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
    lines = lines.split('\n')
    lines = [x for x in lines if '/profiler' not in x and '---' not in x]
    txt = '\n'.join(lines)
    log.debug(f'Torch profile CPU-self {msg}: \n{txt}')
    lines = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    lines = lines.split('\n')
    lines = [x for x in lines if '/profiler' not in x and '---' not in x]
    txt = '\n'.join(lines)
    log.debug(f'Torch profile CUDA {msg}: \n{txt}')
