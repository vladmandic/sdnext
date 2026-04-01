import os
import sys
import logging
import socket
from functools import partial, partialmethod
from logging.handlers import RotatingFileHandler

# rich imports
from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console
from rich.padding import Padding
from rich.segment import Segment
from rich import box
from rich import print as rprint
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

# Global logger and console instances
log = logging.getLogger("sd")
console = None
log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sdnext.log') # adjusted path relative to modules/
hostname = socket.gethostname()
log_rolled = False


def str_to_bool(val: str | bool | None) -> bool | None:
    if isinstance(val, str):
        if val.strip() and val.strip().lower() in ("1", "true"):
            return True
        return False
    return val


def get_console():
    return console


def get_log():
    return log


def install_traceback(suppress: list | None = None):
    if suppress is None:
        suppress = []
    width = os.environ.get("SD_TRACEWIDTH", console.width if console else None)
    if width is not None:
        width = int(width)
    log.excepthook = traceback_install(
        console=console,
        extra_lines=int(os.environ.get("SD_TRACELINES", 1)),
        max_frames=int(os.environ.get("SD_TRACEFRAMES", 16)),
        width=width,
        word_wrap=str_to_bool(os.environ.get("SD_TRACEWRAP", False)),
        indent_guides=str_to_bool(os.environ.get("SD_TRACEINDENT", False)),
        show_locals=str_to_bool(os.environ.get("SD_TRACELOCALS", False)),
        locals_hide_dunder=str_to_bool(os.environ.get("SD_TRACEDUNDER", True)),
        locals_hide_sunder=str_to_bool(os.environ.get("SD_TRACESUNDER", None)),
        suppress=suppress,
    )
    pretty_install(console=console)



_log_config = {'debug': False, 'trace': False, 'log_filename': None}

def setup_logging(debug=None, trace=None, filename=None):
    global log_file, console, log_rolled # pylint: disable=global-statement

    if debug is not None:
        _log_config['debug'] = debug
    if trace is not None:
        _log_config['trace'] = trace
    if filename is not None:
        _log_config['log_filename'] = filename

    debug = _log_config['debug']
    trace = _log_config['trace']
    log_filename = _log_config['log_filename']

    class RingBuffer(logging.StreamHandler):
        def __init__(self, capacity):
            super().__init__()
            self.capacity = capacity
            self.buffer = []
            self.formatter = logging.Formatter('{ "asctime":"%(asctime)s", "created":%(created)f, "facility":"%(name)s", "pid":%(process)d, "tid":%(thread)d, "level":"%(levelname)s", "module":"%(module)s", "func":"%(funcName)s", "msg":"%(message)s" }')

        def emit(self, record):
            if record.msg is not None and not isinstance(record.msg, str):
                record.msg = str(record.msg)
            try:
                record.msg = record.msg.replace('"', "'")
            except Exception:
                pass
            msg = self.format(record)
            self.buffer.append(msg)
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)

        def get(self):
            return self.buffer

    class LogFilter(logging.Filter):
        def __init__(self):
            super().__init__()

        def filter(self, record):
            return len(record.getMessage()) > 2

    def override_padding(self, console, options): # pylint: disable=redefined-outer-name
        style = console.get_style(self.style)
        width = options.max_width
        self.left = 0
        render_options = options.update_width(width - self.left - self.right)
        if render_options.height is not None:
            render_options = render_options.update_height(height=render_options.height - self.top - self.bottom)
        lines = console.render_lines(self.renderable, render_options, style=style, pad=False)
        _Segment = Segment
        left = _Segment(" " * self.left, style) if self.left else None
        right = [_Segment.line()]
        blank_line: list[Segment] | None = None
        if self.top:
            blank_line = [_Segment(f'{" " * width}\\n', style)]
            yield from blank_line * self.top
        if left:
            for line in lines:
                yield left
                yield from line
                yield from right
        else:
            for line in lines:
                yield from line
                yield from right
        if self.bottom:
            blank_line = blank_line or [_Segment(f'{" " * width}\\n', style)]
            yield from blank_line * self.bottom

    if log_filename:
        log_file = log_filename

    logging.TRACE = 25
    logging.addLevelName(logging.TRACE, 'TRACE')
    logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = partial(logging.log, logging.TRACE)

    def exception_hook(e: Exception, suppress: list | None = None):
        from rich.traceback import Traceback
        if suppress is None:
            suppress = []
        tb = Traceback.from_exception(type(e), e, e.__traceback__, show_locals=False, max_frames=16, extra_lines=1, suppress=suppress, theme="ansi_dark", word_wrap=False, width=console.width)
        # print-to-console, does not get printed-to-file
        exc_type, exc_value, exc_traceback = sys.exc_info()
        log.excepthook(exc_type, exc_value, exc_traceback)
        # print-to-file, temporarily disable-console-handler
        for handler in log.handlers.copy():
            if isinstance(handler, RichHandler):
                log.removeHandler(handler)
        with console.capture() as capture:
            console.print(tb)
        log.critical(capture.get())
        log.addHandler(rh)

    log.traceback = exception_hook

    level = logging.DEBUG if (debug or trace) else logging.INFO
    log.setLevel(logging.DEBUG) # log to file is always at level debug for facility `sd`
    log.print = rprint

    theme = Theme({
        "traceback.border": "black",
        "inspect.value.border": "black",
        "traceback.border.syntax_error": "dark_red",
        "logging.level.info": "blue_violet",
        "logging.level.debug": "purple4",
        "logging.level.trace": "dark_blue",
    })

    Padding.__rich_console__ = override_padding
    box.ROUNDED = box.SIMPLE
    console = Console(
        log_time=True,
        log_time_format='%H:%M:%S-%f',
        tab_size=4,
        soft_wrap=True,
        safe_box=True,
        theme=theme,
    )

    logging.basicConfig(level=logging.ERROR, format='%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s', handlers=[logging.NullHandler()]) # redirect default logger to null

    pretty_install(console=console)
    install_traceback()

    while log.hasHandlers() and len(log.handlers) > 0:
        log.removeHandler(log.handlers[0])

    log_filter = LogFilter()
    # handlers
    rh = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=level, console=console)
    if trace:
        rh.formatter = logging.Formatter('[%(module)s][%(pathname)s:%(lineno)d]  %(message)s')
    rh.addFilter(log_filter)
    rh.setLevel(level)
    log.addHandler(rh)

    fh = RotatingFileHandler(log_file, maxBytes=32*1024*1024, backupCount=9, encoding='utf-8', delay=True) # 10MB default for log rotation
    if trace:
        fh.formatter = logging.Formatter(f'%(asctime)s | {hostname} | %(name)s | %(levelname)s | %(module)s | | %(pathname)s:%(lineno)d | %(message)s')
    else:
        fh.formatter = logging.Formatter(f'%(asctime)s | {hostname} | %(name)s | %(levelname)s | %(module)s | %(message)s')
    fh.addFilter(log_filter)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    if not log_rolled and debug and not log_filename:
        try:
            fh.doRollover()
        except Exception:
            pass
        log_rolled = True

    rb = RingBuffer(100) # 100 entries default in log ring buffer
    rb.addFilter(log_filter)
    rb.setLevel(level)
    log.addHandler(rb)
    log.buffer = rb.buffer

    def quiet_log(quiet: bool=False, *args, **kwargs): # pylint: disable=redefined-outer-name,keyword-arg-before-vararg
        if not quiet:
            log.debug(*args, **kwargs)
    log.quiet = quiet_log

    # overrides
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("ControlNet").handlers = log.handlers
    logging.getLogger("lycoris").handlers = log.handlers
