from typing import overload, List, Optional
import os
import sys
import json
import time
import shutil
import locale
import socket
import logging
import platform
import subprocess
import cProfile
import importlib
import importlib.util


class Dot(dict): # dot notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

version = {
    'app': 'sd.next',
    'updated': 'unknown',
    'commit': 'unknown',
    'branch': 'unknown',
    'url': 'unknown',
    'kanvas': 'unknown',
}
pkg_resources, setuptools, distutils = None, None, None # defined via ensure_base_requirements
current_branch = None
log = logging.getLogger("sd")
console = None
debug = log.debug if os.environ.get('SD_INSTALL_DEBUG', None) is not None else lambda *args, **kwargs: None
pip_log = '--log pip.log ' if os.environ.get('SD_PIP_DEBUG', None) is not None else ''
log_file = os.path.join(os.path.dirname(__file__), 'sdnext.log')
hostname = socket.gethostname()
log_rolled = False
first_call = True
quick_allowed = True
errors = []
opts = {}
args = Dot({
    'debug': False,
    'reset': False,
    'profile': False,
    'upgrade': False,
    'skip_extensions': False,
    'skip_requirements': False,
    'skip_git': False,
    'skip_torch': False,
    'use_directml': False,
    'use_ipex': False,
    'use_cuda': False,
    'use_rocm': False,
    'experimental': False,
    'test': False,
    'tls_selfsign': False,
    'reinstall': False,
    'version': False,
    'ignore': False,
    'uv': False,
})
git_commit = "unknown"
diffusers_commit = "unknown"
restart_required = False
extensions_commit = { # force specific commit for extensions
    'sd-webui-controlnet': 'ecd33eb',
    'adetailer': 'a89c01d'
    # 'stable-diffusion-webui-images-browser': '27fe4a7',
}
control_extensions = [ # 3rd party extensions marked as safe for control ui
    'NudeNet',
    'IP Adapters',
    'Remove background',
]


try:
    from modules.timer import init
    ts = init.ts
    elapsed = init.elapsed
except Exception:
    ts = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment
    elapsed = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment


def get_console():
    return console


def get_log():
    return log


@overload
def str_to_bool(val: str | bool) -> bool: ...
@overload
def str_to_bool(val: None) -> None: ...
def str_to_bool(val: str | bool | None) -> bool | None:
    if isinstance(val, str):
        if val.strip() and val.strip().lower() in ("1", "true"):
            return True
        return False
    return val


def install_traceback(suppress: list = []):
    from rich.traceback import install as traceback_install
    from rich.pretty import install as pretty_install

    width = os.environ.get("SD_TRACEWIDTH", console.width if console else None)
    if width is not None:
        width = int(width)
    traceback_install(
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


# setup console and file logging
def setup_logging():
    from functools import partial, partialmethod
    from logging.handlers import RotatingFileHandler
    try:
        import rich # pylint: disable=unused-import
    except Exception:
        log.error('Please restart SD.Next so changes take effect')
        sys.exit(1)
    from rich.theme import Theme
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.padding import Padding
    from rich.segment import Segment
    from rich import box
    from rich import print as rprint
    from rich.pretty import install as pretty_install

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
            # self.buffer.append(json.loads(msg))
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
        blank_line: Optional[List[Segment]] = None
        if self.top:
            blank_line = [_Segment(f'{" " * width}\n', style)]
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
            blank_line = blank_line or [_Segment(f'{" " * width}\n', style)]
            yield from blank_line * self.bottom

    t_start = time.time()

    if args.log:
        global log_file # pylint: disable=global-statement
        log_file = args.log

    logging.TRACE = 25
    logging.addLevelName(logging.TRACE, 'TRACE')
    logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = partial(logging.log, logging.TRACE)

    level = logging.DEBUG if (args.debug or args.trace) else logging.INFO
    log.setLevel(logging.DEBUG) # log to file is always at level debug for facility `sd`
    log.print = rprint
    global console # pylint: disable=global-statement
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
    if args.trace:
        rh.formatter = logging.Formatter('[%(module)s][%(pathname)s:%(lineno)d]  %(message)s')
    rh.addFilter(log_filter)
    rh.setLevel(level)
    log.addHandler(rh)

    fh = RotatingFileHandler(log_file, maxBytes=32*1024*1024, backupCount=9, encoding='utf-8', delay=True) # 10MB default for log rotation
    if args.trace:
        fh.formatter = logging.Formatter(f'%(asctime)s | {hostname} | %(name)s | %(levelname)s | %(module)s | | %(pathname)s:%(lineno)d | %(message)s')
    else:
        fh.formatter = logging.Formatter(f'%(asctime)s | {hostname} | %(name)s | %(levelname)s | %(module)s | %(message)s')
    fh.addFilter(log_filter)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    global log_rolled # pylint: disable=global-statement
    if not log_rolled and args.debug and not args.log:
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
    # logging.getLogger("DeepSpeed").handlers = log.handlers
    ts('log', t_start)


def get_logfile():
    log_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0
    log.info(f'Logger: file="{os.path.abspath(log_file)}" level={logging.getLevelName(logging.DEBUG if args.debug else logging.INFO)} host="{hostname}" size={log_size} mode={"append" if not log_rolled else "create"}')
    return log_file


def custom_excepthook(exc_type, exc_value, exc_traceback):
    import traceback
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error(f"Uncaught exception occurred: type={exc_type} value={exc_value}")
    if exc_traceback:
        format_exception = traceback.format_tb(exc_traceback)
        for line in format_exception:
            log.error(repr(line))


def print_dict(d):
    if d is None:
        return ''
    return ' '.join([f'{k}={v}' for k, v in d.items()])


def print_profile(profiler: cProfile.Profile, msg: str):
    profiler.disable()
    from modules.errors import profile
    profile(profiler, msg)


def package_version(package):
    global pkg_resources # pylint: disable=global-statement
    if pkg_resources is None:
        import pkg_resources # pylint: disable=redefined-outer-name
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def package_spec(package):
    global pkg_resources # pylint: disable=global-statement
    if pkg_resources is None:
        import pkg_resources # pylint: disable=redefined-outer-name
    spec = pkg_resources.working_set.by_key.get(package, None) # more reliable than importlib
    if spec is None:
        spec = pkg_resources.working_set.by_key.get(package.lower(), None) # check name variations
    if spec is None:
        spec = pkg_resources.working_set.by_key.get(package.replace('_', '-'), None) # check name variations
    return spec


# check if package is installed
def installed(package, friendly: str = None, reload = False, quiet = False): # pylint: disable=redefined-outer-name
    t_start = time.time()
    ok = True
    try:
        if reload:
            try:
                importlib.reload(pkg_resources)
            except Exception:
                pass
        if friendly:
            pkgs = friendly.split()
        else:
            pkgs = [p for p in package.split() if not p.startswith('-') and not p.startswith('=') and not p.startswith('git+')]
            pkgs = [p.split('/')[-1] for p in pkgs] # get only package name if installing from url
        for pkg in pkgs:
            if '!=' in pkg:
                p = pkg.split('!=')
                return True # check for not equal always return true
            elif '>=' in pkg:
                p = pkg.split('>=')
            else:
                p = pkg.split('==')
            spec = package_spec(p[0])
            ok = ok and spec is not None
            if ok:
                pkg_version = package_version(p[0])
                if len(p) > 1:
                    exact = pkg_version == p[1]
                    if not exact and not quiet:
                        if args.experimental:
                            log.warning(f'Install: package="{p[0]}" installed={pkg_version} required={p[1]} allowing experimental')
                        else:
                            log.warning(f'Install: package="{p[0]}" installed={pkg_version} required={p[1]} version mismatch')
                            global restart_required # pylint: disable=global-statement
                            restart_required = True
                    ok = ok and (exact or args.experimental)
            else:
                if not quiet:
                    log.debug(f'Install: package="{p[0]}" install required')
        ts('installed', t_start)
        return ok
    except Exception as e:
        log.error(f'Install: package="{pkgs}" {e}')
        ts('installed', t_start)
        return False

def uninstall(package, quiet = False):
    t_start = time.time()
    packages = package if isinstance(package, list) else [package]
    res = ''
    for p in packages:
        if installed(p, p, quiet=True):
            if not quiet:
                log.warning(f'Package: {p} uninstall')
            res += pip(f"uninstall {p} --yes --quiet", ignore=True, quiet=True, uv=False)
    ts('uninstall', t_start)
    return res


def run(cmd: str, arg: str):
    result = subprocess.run(f'"{cmd}" {arg}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    debug(f'Exec {cmd}: {txt}')
    return txt


def pip(arg: str, ignore: bool = False, quiet: bool = True, uv = True):
    t_start = time.time()
    originalArg = arg
    arg = arg.replace('>=', '==')
    if opts.get('offline_mode', False):
        log.warning('Offline mode enabled')
        return 'offline'
    package = arg.replace("install", "").replace("--upgrade", "").replace("--no-deps", "").replace("--force-reinstall", "").replace(" ", " ").strip()
    uv = uv and args.uv and not package.startswith('git+')
    pipCmd = "uv pip" if uv else "pip"
    if not quiet and '-r ' not in arg:
        log.info(f'Install: package="{package}" mode={"uv" if uv else "pip"}')
    env_args = os.environ.get("PIP_EXTRA_ARGS", "")
    all_args = f'{pip_log}{arg} {env_args}'.strip()
    if not quiet:
        log.debug(f'Running: {pipCmd}="{all_args}"')
    result = subprocess.run(f'"{sys.executable}" -m {pipCmd} {all_args}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        if uv and result.returncode != 0:
            err = result.stderr.decode(encoding="utf8", errors="ignore")
            log.warning(f'Install: cmd="{pipCmd}" args="{all_args}" cannot use uv, fallback to pip')
            debug(f'Install: uv pip error: {err}')
            return pip(originalArg, ignore, quiet, uv=False)
        else:
            txt += ('\n' if len(txt) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    txt = txt.strip()
    debug(f'Install {pipCmd}: {txt}')
    if result.returncode != 0 and not ignore:
        errors.append(f'pip: {package}')
        log.error(f'Install: {pipCmd}: {arg}')
        log.debug(f'Install: pip output {txt}')
    ts('pip', t_start)
    return txt


# install package using pip if not already installed
def install(package, friendly: str = None, ignore: bool = False, reinstall: bool = False, no_deps: bool = False, quiet: bool = False, force: bool = False):
    t_start = time.time()
    res = ''
    if args.reinstall or args.upgrade:
        global quick_allowed # pylint: disable=global-statement
        quick_allowed = False
    if (args.reinstall) or (reinstall) or (not installed(package, friendly, quiet=quiet)):
        deps = '' if not no_deps else '--no-deps '
        cmd = f"install{' --upgrade' if not args.uv else ''}{' --force-reinstall' if force else ''} {deps}{package}"
        res = pip(cmd, ignore=ignore, uv=package != "uv" and not package.startswith('git+'))
        try:
            importlib.reload(pkg_resources)
        except Exception:
            pass
    ts('install', t_start)
    return res


# execute git command
def git(arg: str, folder: str = None, ignore: bool = False, optional: bool = False): # pylint: disable=unused-argument
    t_start = time.time()
    if args.skip_git:
        return ''
    if 'google.colab' in sys.modules:
        return ''
    git_cmd = os.environ.get('GIT', "git")
    if git_cmd != "git":
        git_cmd = os.path.abspath(git_cmd)
    result = subprocess.run(f'"{git_cmd}" {arg}', check=False, shell=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder or '.')
    stdout = result.stdout.decode(encoding="utf8", errors="ignore")
    if len(result.stderr) > 0:
        stdout += ('\n' if len(stdout) > 0 else '') + result.stderr.decode(encoding="utf8", errors="ignore")
    stdout = stdout.strip()
    if result.returncode != 0 and not ignore:
        if folder is None:
            folder = 'root'
        if "couldn't find remote ref" in stdout: # not a git repo
            log.error(f'Git: folder="{folder}" could not identify repository')
        elif "no submodule mapping found" in stdout:
            log.warning(f'Git: folder="{folder}" submodules changed')
        elif 'or stash them' in stdout:
            log.error(f'Git: folder="{folder}" local changes detected')
        else:
            log.error(f'Git: folder="{folder}" arg="{arg}" output={stdout}')
        errors.append(f'git: {folder}')
    ts('git', t_start)
    return stdout


# reattach as needed as head can get detached
def branch(folder=None):
    # if args.experimental:
    #    return None
    t_start = time.time()
    if not os.path.exists(os.path.join(folder or os.curdir, '.git')):
        return None
    branches = []
    try:
        b = git('branch --show-current', folder, optional=True)
        if b == '':
            branches = git('branch', folder).split('\n')
        if len(branches) > 0:
            b = [x for x in branches if x.startswith('*')][0]
            if 'detached' in b and len(branches) > 1:
                b = branches[1].strip()
                log.debug(f'Git detached head detected: folder="{folder}" reattach={b}')
    except Exception:
        b = git('git rev-parse --abbrev-ref HEAD', folder, optional=True)
    if 'main' in b:
        b = 'main'
    elif 'master' in b:
        b = 'master'
    else:
        b = b.split('\n')[0].replace('*', '').strip()
    log.debug(f'Git submodule: {folder} / {b}')
    git(f'checkout {b}', folder, ignore=True, optional=True)
    ts('branch', t_start)
    return b


# restart process
def restart():
    log.critical('Restarting process...')
    os.execv(sys.executable, ['python'] + sys.argv)


# update git repository
def update(folder, keep_branch = False, rebase = True):
    t_start = time.time()
    try:
        git('config rebase.Autostash true')
    except Exception:
        pass
    arg = '--rebase --force' if rebase else ''
    if keep_branch:
        res = git(f'pull {arg}', folder)
        debug(f'Install update: folder={folder} args={arg} {res}')
    else:
        b = branch(folder)
        if branch is None:
            res = git(f'pull {arg}', folder)
            debug(f'Install update: folder={folder} branch={b} args={arg} {res}')
        else:
            res = git(f'pull origin {b} {arg}', folder)
            debug(f'Install update: folder={folder} branch={b} args={arg} {res}')
        if not args.experimental:
            commit = extensions_commit.get(os.path.basename(folder), None)
            if commit is not None:
                res = git(f'checkout {commit}', folder)
                debug(f'Install update: folder={folder} branch={b} args={arg} commit={commit} {res}')
    ts('update', t_start)
    return res


# clone git repository
def clone(url, folder, commithash=None):
    t_start = time.time()
    if os.path.exists(folder):
        if commithash is None:
            update(folder)
        else:
            current_hash = git('rev-parse HEAD', folder).strip()
            if current_hash != commithash:
                res = git('fetch', folder)
                debug(f'Install clone: {res}')
                git(f'checkout {commithash}', folder)
                return
    else:
        log.info(f'Cloning repository: {url}')
        git(f'clone "{url}" "{folder}"')
        if commithash is not None:
            git(f'-C "{folder}" checkout {commithash}')
    ts('clone', t_start)


def get_platform():
    try:
        if platform.system() == 'Windows':
            release = platform.platform(aliased = True, terse = False)
        else:
            release = platform.release()
        return {
            'arch': platform.machine(),
            'cpu': platform.processor(),
            'system': platform.system(),
            'release': release,
            'python': platform.python_version(),
            'locale': locale.getlocale(),
            'docker': os.environ.get('SD_DOCKER', None) is not None,
            # 'host': platform.node(),
            # 'version': platform.version(),
        }
    except Exception as e:
        return { 'error': e }


# check python version
def check_python(supported_minors=[], experimental_minors=[], reason=None):
    if supported_minors is None or len(supported_minors) == 0:
        supported_minors = [10, 11, 12]
        experimental_minors = [13]
    t_start = time.time()
    if args.quick:
        return
    log.info(f'Python: version={platform.python_version()} platform={platform.system()} bin="{sys.executable}" venv="{sys.prefix}"')
    if int(sys.version_info.minor) == 12:
        os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'local') # hack for python 3.11 setuptools
    if int(sys.version_info.minor) == 10:
        log.warning(f"Python: version={platform.python_version()} is not actively supported")
    if int(sys.version_info.minor) == 9:
        log.error(f"Python: version={platform.python_version()} is end-of-life")
    if not (int(sys.version_info.major) == 3 and int(sys.version_info.minor) in supported_minors):
        if (int(sys.version_info.major) == 3 and int(sys.version_info.minor) in experimental_minors):
            log.warning(f"Python experimental: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        else:
            log.error(f"Python incompatible: current {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} required 3.{supported_minors}")
            if reason is not None:
                log.error(reason)
            if not args.ignore and not args.experimental:
                sys.exit(1)
    if not args.skip_git:
        git_cmd = os.environ.get('GIT', "git")
        if shutil.which(git_cmd) is None:
            log.error('Git not found')
            if not args.ignore:
                sys.exit(1)
    else:
        git_version = git('--version', folder=None, ignore=False)
        log.debug(f'Git: version={git_version.replace("git version", "").strip()}')
    ts('python', t_start)


# check diffusers version
def check_diffusers():
    t_start = time.time()
    if args.skip_all:
        return
    sha = '430c557b6a66a3c2b5740fb186324cb8a9f0f2e9' # diffusers commit hash
    # if args.use_rocm or args.use_zluda or args.use_directml:
    #     sha = '043ab2520f6a19fce78e6e060a68dbc947edb9f9' # lock diffusers versions for now
    pkg = pkg_resources.working_set.by_key.get('diffusers', None)
    minor = int(pkg.version.split('.')[1] if pkg is not None else -1)
    cur = opts.get('diffusers_version', '') if minor > -1 else ''
    if (minor == -1) or ((cur != sha) and (not args.experimental)):
        if minor == -1:
            log.info(f'Diffusers install: commit={sha}')
        else:
            log.info(f'Diffusers update: current={pkg.version} hash={cur} target={sha}')
            pip('uninstall --yes diffusers', ignore=True, quiet=True, uv=False)
        if args.skip_git:
            log.warning('Git: marked as not available but required for diffusers installation')
        pip(f'install --upgrade git+https://github.com/huggingface/diffusers@{sha}', ignore=False, quiet=True, uv=False)
        global diffusers_commit # pylint: disable=global-statement
        diffusers_commit = sha
    ts('diffusers', t_start)


# check transformers version
def check_transformers():
    t_start = time.time()
    if args.skip_all or args.skip_git or args.experimental:
        return
    pkg_transformers = pkg_resources.working_set.by_key.get('transformers', None)
    pkg_tokenizers = pkg_resources.working_set.by_key.get('tokenizers', None)
    if args.use_directml:
        target_transformers = '4.52.4'
        target_tokenizers = '0.21.4'
    elif args.new:
        target_transformers = '5.0.0rc2'
        target_tokenizers = '0.22.2'
    else:
        target_transformers = '4.57.5'
        target_tokenizers = '0.22.2'
    if (pkg_transformers is None) or ((pkg_transformers.version != target_transformers) or (pkg_tokenizers is None) or ((pkg_tokenizers.version != target_tokenizers) and (not args.experimental))):
        if pkg_transformers is None:
            log.info(f'Transformers install: version={target_transformers}')
        else:
            log.info(f'Transformers update: current={pkg_transformers.version} target={target_transformers}')
        pip('uninstall --yes transformers', ignore=True, quiet=True, uv=False)
        pip(f'install --upgrade tokenizers=={target_tokenizers}', ignore=False, quiet=True, uv=False)
        pip(f'install --upgrade transformers=={target_transformers}', ignore=False, quiet=True, uv=False)
    ts('transformers', t_start)


# check onnx version
def check_onnx():
    t_start = time.time()
    if args.skip_all or args.skip_requirements:
        return
    if not installed('onnx', quiet=True):
        install('onnx', 'onnx', ignore=True)
    if not installed('onnxruntime', quiet=True) and not (installed('onnxruntime-gpu', quiet=True) or installed('onnxruntime-openvino', quiet=True) or installed('onnxruntime-training', quiet=True)): # allow either
        install(os.environ.get('ONNXRUNTIME_COMMAND', 'onnxruntime'), ignore=True)
    ts('onnx', t_start)


def install_cuda():
    t_start = time.time()
    log.info('CUDA: nVidia toolkit detected')
    ts('cuda', t_start)
    if args.use_nightly:
        cmd = os.environ.get('TORCH_COMMAND', '--upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --extra-index-url https://download.pytorch.org/whl/nightly/cu126')
    else:
        cmd = os.environ.get('TORCH_COMMAND', 'torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128')
    return cmd


def install_rocm_zluda():
    torch_command = ''
    t_start = time.time()
    if args.skip_all or args.skip_requirements:
        return torch_command
    from modules import rocm

    amd_gpus = []
    try:
        amd_gpus = rocm.get_agents()
    except Exception as e:
        log.warning(f'ROCm agent enumerator failed: {e}')

    #os.environ.setdefault('TENSORFLOW_PACKAGE', 'tensorflow')

    device = None
    if len(amd_gpus) == 0:
        log.warning('ROCm: no agent was found')
    else:
        log.info(f'ROCm: agents={[gpu.name for gpu in amd_gpus]}')
        if args.device_id is None:
            index = 0
            for idx, gpu in enumerate(amd_gpus):
                index = idx
                if not gpu.is_apu:
                    # although apu was found, there can be a dedicated card. do not break loop.
                    # if no dedicated card was found, apu will be used.
                    break
            os.environ.setdefault('HIP_VISIBLE_DEVICES', str(index))
            device = amd_gpus[index]
        else:
            device_id = int(args.device_id)
            if device_id < len(amd_gpus):
                device = amd_gpus[device_id]

    if sys.platform == "win32" and not args.use_zluda and device is not None and device.therock is not None and not installed("rocm"):
        check_python(supported_minors=[11, 12, 13], reason='ROCm backend requires a Python version between 3.11 and 3.13')
        install(f"rocm[devel,libraries] --index-url https://rocm.nightlies.amd.com/{device.therock}")
        rocm.refresh()

    msg = f'ROCm: version={rocm.version}'
    if device is not None:
        msg += f', using agent {device}'
    log.info(msg)

    if sys.platform == "win32":
        if args.use_zluda:
            #check_python(supported_minors=[10, 11, 12, 13], reason='ZLUDA backend requires a Python version between 3.10 and 3.13')
            torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118')

            if args.device_id is not None:
                if os.environ.get('HIP_VISIBLE_DEVICES', None) is not None:
                    log.warning('Setting HIP_VISIBLE_DEVICES and --device-id at the same time may be mistake.')
                os.environ['HIP_VISIBLE_DEVICES'] = args.device_id
                del args.device_id

            from modules import zluda_installer
            try:
                if args.reinstall or zluda_installer.is_reinstall_needed():
                    zluda_installer.uninstall()
                zluda_installer.install()
                zluda_installer.set_default_agent(device)
            except Exception as e:
                log.error(f'Install ZLUDA: {e}')

            try:
                zluda_installer.load()
            except Exception as e:
                log.error(f'Load ZLUDA: {e}')
        else: # TODO rocm: switch to pytorch source when it becomes available
            if device is None:
                log.error('ROCm: no agent found - make sure that graphics driver is installed and up to date')
            if isinstance(rocm.environment, rocm.PythonPackageEnvironment):
                check_python(supported_minors=[11, 12, 13], reason='ROCm: python==3.11/3.12/3.13 required')
                torch_command = os.environ.get('TORCH_COMMAND', f'torch torchvision --index-url https://rocm.nightlies.amd.com/{device.therock}')
            else:
                check_python(supported_minors=[12], reason='ROCm: Windows preview python==3.12 required')
                torch_command = os.environ.get('TORCH_COMMAND', '--no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torch-2.8.0a0%2Bgitfc14c65-cp312-cp312-win_amd64.whl https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/torchvision-0.24.0a0%2Bc85f008-cp312-cp312-win_amd64.whl')
    else:
        #check_python(supported_minors=[10, 11, 12, 13, 14], reason='ROCm backend requires a Python version between 3.10 and 3.13')
        if args.use_nightly:
            if rocm.version is None or float(rocm.version) >= 7.1: # assume the latest if version check fails
                torch_command = os.environ.get('TORCH_COMMAND', '--upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.1')
            else: # oldest rocm version on nightly is 7.0
                torch_command = os.environ.get('TORCH_COMMAND', '--upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0')
        else:
            if rocm.version is None or float(rocm.version) >= 6.4: # assume the latest if version check fails
                torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.9.1+rocm6.4 torchvision==0.24.1+rocm6.4 --index-url https://download.pytorch.org/whl/rocm6.4')
            elif rocm.version == "6.3":
                torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.9.1+rocm6.3 torchvision==0.24.1+rocm6.3 --index-url https://download.pytorch.org/whl/rocm6.3')
            elif rocm.version == "6.2":
                # use rocm 6.2.4 instead of 6.2 as torch==2.7.1+rocm6.2 doesn't exists
                torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.7.1+rocm6.2.4 torchvision==0.22.1+rocm6.2.4 --index-url https://download.pytorch.org/whl/rocm6.2.4')
            elif rocm.version == "6.1":
                torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.6.0+rocm6.1 torchvision==0.21.0+rocm6.1 --index-url https://download.pytorch.org/whl/rocm6.1')
            else:
                # lock to 2.4.1 instead of 2.5.1 for performance reasons there are no support for torch 2.6 for rocm 6.0
                torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.4.1+rocm6.0 torchvision==0.19.1+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0')
                if float(rocm.version) < 6.0:
                    log.warning(f"ROCm: unsupported version={rocm.version}")
                    log.warning("ROCm: minimum supported version=6.0")

    if device is None or os.environ.get("HSA_OVERRIDE_GFX_VERSION", None) is not None:
        log.info(f'ROCm: HSA_OVERRIDE_GFX_VERSION auto config skipped: device={device} version={os.environ.get("HSA_OVERRIDE_GFX_VERSION", None)}')
    else:
        gfx_ver = device.get_gfx_version()
        if gfx_ver is not None and device.name.removeprefix("gfx") != gfx_ver.replace(".", ""):
            os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', gfx_ver)
            log.info(f'ROCm: HSA_OVERRIDE_GFX_VERSION config overridden: device={device} version={os.environ.get("HSA_OVERRIDE_GFX_VERSION", None)}')

    ts('amd', t_start)
    return torch_command


def install_ipex():
    t_start = time.time()
    #check_python(supported_minors=[10, 11, 12, 13, 14], reason='IPEX backend requires a Python version between 3.10 and 3.13')
    args.use_ipex = True # pylint: disable=attribute-defined-outside-init
    log.info('IPEX: Intel OneAPI toolkit detected')

    if args.use_nightly:
        torch_command = os.environ.get('TORCH_COMMAND', '--upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu')
    else:
        torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.9.1+xpu torchvision==0.24.1+xpu --index-url https://download.pytorch.org/whl/xpu')

    ts('ipex', t_start)
    return torch_command


def install_openvino():
    t_start = time.time()
    log.info('OpenVINO: selected')
    os.environ.setdefault('PYTORCH_TRACING_MODE', 'TORCHFX')

    #check_python(supported_minors=[10, 11, 12, 13], reason='OpenVINO backend requires a Python version between 3.10 and 3.13')
    if sys.platform == 'darwin':
        torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.9.1 torchvision==0.24.1')
    else:
        torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.9.1+cpu torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu')

    if not (args.skip_all or args.skip_requirements):
        install(os.environ.get('OPENVINO_COMMAND', 'openvino==2025.3.0'), 'openvino')
        install(os.environ.get('NNCF_COMMAND', 'nncf==2.18.0'), 'nncf')
    ts('openvino', t_start)
    return torch_command


def install_torch_addons():
    t_start = time.time()
    triton_command = os.environ.get('TRITON_COMMAND', None)
    if triton_command is not None and triton_command != 'skip':
        install(triton_command, 'triton', quiet=True)
    xformers_package = os.environ.get('XFORMERS_PACKAGE', '--pre xformers') if opts.get('cross_attention_optimization', '') == 'xFormers' or args.use_xformers else 'none'
    if 'xformers' in xformers_package:
        try:
            install(xformers_package, ignore=True, no_deps=True)
            import torch # pylint: disable=unused-import
            import xformers # pylint: disable=unused-import
        except Exception as e:
            log.debug(f'xFormers cannot install: {e}')
    elif not args.experimental and not args.use_xformers and opts.get('cross_attention_optimization', '') != 'xFormers':
        uninstall('xformers')
    if opts.get('cuda_compile_backend', '') == 'hidet':
        install('hidet', 'hidet')
    if opts.get('cuda_compile_backend', '') == 'deep-cache':
        install('DeepCache')
    if opts.get('cuda_compile_backend', '') == 'olive-ai':
        install('olive-ai')
    if len(opts.get('optimum_quanto_weights', [])):
        install('optimum-quanto==0.2.7', 'optimum-quanto')
    if len(opts.get('torchao_quantization', [])):
        install('torchao==0.10.0', 'torchao')
    if opts.get('samples_format', 'jpg') == 'jxl' or opts.get('grid_format', 'jpg') == 'jxl':
        install('pillow-jxl-plugin==1.3.5', 'pillow-jxl-plugin')
    if not args.experimental:
        uninstall('wandb', quiet=True)
        uninstall('pynvml', quiet=True)
    ts('addons', t_start)


# check cudnn
def check_cudnn():
    import site
    site_packages = site.getsitepackages()
    cuda_path = os.environ.get('CUDA_PATH', '')
    if cuda_path == '':
        for site_package in site_packages:
            folder = os.path.join(site_package, 'nvidia', 'cudnn', 'lib')
            if os.path.exists(folder) and folder not in cuda_path:
                cuda_path = f"{cuda_path}:{folder}"
                if cuda_path.startswith(':'):
                    cuda_path = cuda_path[1:]
                os.environ['CUDA_PATH'] = cuda_path


# check torch version
def check_torch():
    log.info('Verifying torch installation')
    t_start = time.time()
    if args.skip_torch:
        log.info('Torch: skip tests')
        return
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    allow_cuda = not (args.use_rocm or args.use_directml or args.use_ipex or args.use_openvino)
    allow_rocm = not (args.use_cuda or args.use_directml or args.use_ipex or args.use_openvino)
    allow_ipex = not (args.use_cuda or args.use_rocm or args.use_directml or args.use_openvino)
    allow_directml = not (args.use_cuda or args.use_rocm or args.use_ipex or args.use_openvino)
    allow_openvino = not (args.use_cuda or args.use_rocm or args.use_ipex or args.use_directml)
    log.debug(f'Torch overrides: cuda={args.use_cuda} rocm={args.use_rocm} ipex={args.use_ipex} directml={args.use_directml} openvino={args.use_openvino} zluda={args.use_zluda}')
    # log.debug(f'Torch allowed: cuda={allow_cuda} rocm={allow_rocm} ipex={allow_ipex} diml={allow_directml} openvino={allow_openvino}')
    torch_command = os.environ.get('TORCH_COMMAND', '')

    if sys.platform != 'win32':
        if args.use_zluda:
            log.error('ZLUDA is only supported on Windows')
        if args.use_directml:
            log.error('DirectML is only supported on Windows')

    if torch_command != '':
        is_cuda_available = False
        is_ipex_available = False
        is_rocm_available = False
    else:
        is_cuda_available = allow_cuda and (args.use_cuda or shutil.which('nvidia-smi') is not None or os.path.exists(os.path.join(os.environ.get('SystemRoot') or r'C:\Windows', 'System32', 'nvidia-smi.exe')))
        is_ipex_available = allow_ipex and (args.use_ipex or shutil.which('sycl-ls') is not None or shutil.which('sycl-ls.exe') is not None or os.environ.get('ONEAPI_ROOT') is not None or os.path.exists('/opt/intel/oneapi') or os.path.exists("C:/Program Files (x86)/Intel/oneAPI") or os.path.exists("C:/oneAPI") or os.path.exists("C:/Program Files/Intel/Intel Graphics Software"))
        is_rocm_available = False

        if not is_cuda_available and not is_ipex_available and allow_rocm:
            from modules import rocm
            is_rocm_available = allow_rocm and (args.use_rocm or args.use_zluda or rocm.is_installed) # late eval to avoid unnecessary import

        if is_cuda_available and args.use_cuda: # prioritize cuda
            torch_command = install_cuda()
        elif is_rocm_available and (args.use_rocm or args.use_zluda): # prioritize rocm
            torch_command = install_rocm_zluda()
        elif allow_ipex and args.use_ipex: # prioritize ipex
            torch_command = install_ipex()
        elif allow_openvino and args.use_openvino: # prioritize openvino
            torch_command = install_openvino()
        elif is_cuda_available:
            torch_command = install_cuda()
        elif is_rocm_available:
            torch_command = install_rocm_zluda()
        elif is_ipex_available:
            torch_command = install_ipex()
        else:
            machine = platform.machine()
            if sys.platform == 'darwin':
                torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision')
            elif allow_directml and args.use_directml and ('arm' not in machine and 'aarch' not in machine):
                log.info('DirectML: selected')
                torch_command = os.environ.get('TORCH_COMMAND', 'torch==2.4.1 torchvision torch-directml==0.2.4.dev240913')
                if 'torch' in torch_command and not args.version:
                    install(torch_command, 'torch torchvision')
                install('onnxruntime-directml', 'onnxruntime-directml', ignore=True)
            else:
                log.warning('Torch: CPU-only version installed')
                torch_command = os.environ.get('TORCH_COMMAND', 'torch torchvision')

    if args.version:
        return

    if 'torch' in torch_command:
        if not installed('torch'):
            log.info(f'Torch: download and install in progress... cmd="{torch_command}"')
            install('--upgrade pip', 'pip', reinstall=True) # pytorch rocm is too large for older pip
        install(torch_command, 'torch torchvision', quiet=True)

    try:
        import torch
        try:
            import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
            log.info(f'Torch backend: type=IPEX version={ipex.__version__}')
        except Exception:
            pass
        if 'cpu' in torch.__version__:
            if is_cuda_available:
                if args.use_cuda:
                    log.warning(f'Torch: version="{torch.__version__}" CPU version installed and CUDA is selected - reinstalling')
                    install(torch_command, 'torch torchvision', quiet=True, reinstall=True, force=True) # foce reinstall
                else:
                    log.warning(f'Torch: version="{torch.__version__}" CPU version installed and CUDA is available - consider reinstalling')
            elif is_rocm_available:
                if args.use_rocm:
                    log.warning(f'Torch: version="{torch.__version__}" CPU version installed and ROCm is selected - reinstalling')
                    install(torch_command, 'torch torchvision', quiet=True, reinstall=True, force=True) # foce reinstall
                else:
                    log.warning(f'Torch: version="{torch.__version__}" CPU version installed and ROCm is available - consider reinstalling')
        if hasattr(torch, "xpu") and torch.xpu.is_available() and allow_ipex:
            if shutil.which('icpx') is not None:
                log.info(f'{os.popen("icpx --version").read().rstrip()}')
            for device in range(torch.xpu.device_count()):
                log.info(f'Torch detected: gpu="{torch.xpu.get_device_name(device)}" vram={round(torch.xpu.get_device_properties(device).total_memory / 1024 / 1024)} units={torch.xpu.get_device_properties(device).max_compute_units}')
        elif torch.cuda.is_available() and (allow_cuda or allow_rocm):
            if torch.version.cuda and allow_cuda:
                log.info(f'Torch backend: version="{torch.__version__}" type=CUDA CUDA={torch.version.cuda} cuDNN={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}')
            elif torch.version.hip and allow_rocm:
                log.info(f'Torch backend: version="{torch.__version__}" type=ROCm HIP={torch.version.hip}')
            else:
                log.warning('Unknown Torch backend')
            for device in [torch.cuda.device(i) for i in range(torch.cuda.device_count())]:
                log.info(f'Torch detected: gpu="{torch.cuda.get_device_name(device)}" vram={round(torch.cuda.get_device_properties(device).total_memory / 1024 / 1024)} arch={torch.cuda.get_device_capability(device)} cores={torch.cuda.get_device_properties(device).multi_processor_count}')
        else:
            try:
                if args.use_directml and allow_directml:
                    import torch_directml # pylint: disable=import-error
                    dml_ver = pkg_resources.get_distribution("torch-directml")
                    log.warning(f'Torch backend: DirectML ({dml_ver})')
                    log.warning('DirectML: end-of-life')
                    for i in range(0, torch_directml.device_count()):
                        log.info(f'Torch detected GPU: {torch_directml.device_name(i)}')
            except Exception:
                log.warning("Torch reports CUDA not available")
    except Exception as e:
        log.error(f'Torch cannot load: {e}')
        if not args.ignore:
            sys.exit(1)

    if is_rocm_available:
        rocm.postinstall()
    if not args.skip_all:
        install_torch_addons()
    check_cudnn()
    if args.profile:
        pr.disable()
        print_profile(pr, 'Torch')
    ts('torch', t_start)


# check modified files
def check_modified_files():
    t_start = time.time()
    if args.quick:
        return
    if args.skip_git:
        return
    try:
        res = git('status --porcelain')
        files = [x[2:].strip() for x in res.split('\n')]
        files = [x for x in files if len(x) > 0 and (not x.startswith('extensions')) and (not x.startswith('wiki')) and (not x.endswith('.json')) and ('.log' not in x)]
        deleted = [x for x in files if not os.path.exists(x)]
        if len(deleted) > 0:
            log.warning(f'Deleted files: {deleted}')
        modified = [x for x in files if os.path.exists(x) and not os.path.isdir(x)]
        if len(modified) > 0:
            log.warning(f'Modified files: {modified}')
    except Exception:
        pass
    ts('files', t_start)


# install required packages
def install_packages():
    t_start = time.time()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    # log.info('Install: verifying packages')
    clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git")
    install(clip_package, 'clip', quiet=True)
    install('open-clip-torch', no_deps=True, quiet=True)
    # tensorflow_package = os.environ.get('TENSORFLOW_PACKAGE', 'tensorflow==2.13.0')
    # tensorflow_package = os.environ.get('TENSORFLOW_PACKAGE', None)
    # if tensorflow_package is not None:
    #    install(tensorflow_package, 'tensorflow-rocm' if 'rocm' in tensorflow_package else 'tensorflow', ignore=True, quiet=True)
    if args.profile:
        pr.disable( )
        print_profile(pr, 'Packages')
    ts('packages', t_start)


# run extension installer
def run_extension_installer(folder):
    path_installer = os.path.realpath(os.path.join(folder, "install.py"))
    if not os.path.isfile(path_installer):
        return
    try:
        is_builtin = 'extensions-builtin' in folder
        log.debug(f'Extension installer: builtin={is_builtin} file="{path_installer}"')
        if is_builtin:
            module_spec = importlib.util.spec_from_file_location(os.path.basename(folder), path_installer)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.abspath(".")
            if os.environ.get('PYTHONPATH', None) is not None:
                seperator = ';' if sys.platform == 'win32' else ':'
                env['PYTHONPATH'] += seperator + os.environ.get('PYTHONPATH', None)
            result = subprocess.run(f'"{sys.executable}" "{path_installer}"', shell=True, env=env, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=folder)
            txt = result.stdout.decode(encoding="utf8", errors="ignore")
            debug(f'Extension installer: file="{path_installer}" {txt}')
            if result.returncode != 0:
                errors.append(f'ext: {os.path.basename(folder)}')
                if len(result.stderr) > 0:
                    txt = txt + '\n' + result.stderr.decode(encoding="utf8", errors="ignore")
                log.error(f'Extension installer error: {path_installer}')
                log.debug(txt)
    except Exception as e:
        log.error(f'Extension installer exception: {e}')


# get list of all enabled extensions
def list_extensions_folder(folder, quiet=False):
    disabled_extensions_all = opts.get('disable_all_extensions', 'none')
    if disabled_extensions_all != 'none':
        return []
    disabled_extensions = opts.get('disabled_extensions', [])
    enabled_extensions = [x for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x)) and x not in disabled_extensions and not x.startswith('.')]
    if not quiet:
        log.info(f'Extensions: path="{folder}" enabled={enabled_extensions}')
    return enabled_extensions


# run installer for each installed and enabled extension and optionally update them
def install_extensions(force=False):
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    pkg_resources._initialize_master_working_set() # pylint: disable=protected-access
    pkgs = [f'{p.project_name}=={p._version}' for p in pkg_resources.working_set] # pylint: disable=protected-access,not-an-iterable
    log.debug(f'Installed packages: {len(pkgs)}')
    from modules.paths import extensions_builtin_dir, extensions_dir
    extensions_duplicates = []
    extensions_enabled = []
    extensions_disabled = [e.lower() for e in opts.get('disabled_extensions', [])]
    extension_folders = [extensions_builtin_dir] if args.safe else [extensions_builtin_dir, extensions_dir]
    res = []
    for folder in extension_folders:
        if not os.path.isdir(folder):
            continue
        extensions = list_extensions_folder(folder, quiet=True)
        log.debug(f'Extensions all: {extensions}')
        for ext in extensions:
            if os.path.basename(ext).lower() in extensions_disabled:
                continue
            t_start = time.time()
            if ext in extensions_enabled:
                extensions_duplicates.append(ext)
                continue
            extensions_enabled.append(ext)
            if args.upgrade or force:
                try:
                    res.append(update(os.path.join(folder, ext)))
                except Exception:
                    res.append(f'Extension update error: {os.path.join(folder, ext)}')
                    log.error(f'Extension update error: {os.path.join(folder, ext)}')
            if not args.skip_extensions:
                commit = extensions_commit.get(os.path.basename(ext), None)
                if commit is not None:
                    log.debug(f'Extension force: name="{ext}" commit={commit}')
                    res.append(git(f'checkout {commit}', os.path.join(folder, ext)))
                run_extension_installer(os.path.join(folder, ext))
            pkg_resources._initialize_master_working_set() # pylint: disable=protected-access
            try:
                updated = [f'{p.project_name}=={p._version}' for p in pkg_resources.working_set] # pylint: disable=protected-access,not-an-iterable
                diff = [x for x in updated if x not in pkgs]
                pkgs = updated
                if len(diff) > 0:
                    log.info(f'Extension installed packages: {ext} {diff}')
            except Exception as e:
                log.error(f'Extension installed unknown package: {e}')
            ts(ext, t_start)
    log.info(f'Extensions enabled: {extensions_enabled}')
    if len(extensions_duplicates) > 0:
        log.warning(f'Extensions duplicates: {extensions_duplicates}')
    if args.profile:
        pr.disable()
        print_profile(pr, 'Extensions')
    # ts('extensions', t_start)
    return '\n'.join(res)


# initialize and optionally update submodules
def install_submodules(force=True):
    t_start = time.time()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    log.info('Verifying submodules')
    txt = git('submodule')
    # log.debug(f'Submodules list: {txt}')
    if force and 'no submodule mapping found' in txt and 'extension-builtin' not in txt:
        txt = git('submodule')
        git_reset()
        log.info('Continuing setup')
    git('submodule --quiet update --init --recursive')
    git('submodule --quiet sync --recursive')
    submodules = txt.splitlines()
    res = []
    for submodule in submodules:
        try:
            name = submodule.split()[1].strip()
            if args.upgrade:
                res.append(update(name))
            else:
                branch(name)
        except Exception:
            log.error(f'Submodule update error: {submodule}')
    setup_logging()
    if args.profile:
        pr.disable()
        print_profile(pr, 'Submodule')
    ts('submodules', t_start)
    return '\n'.join(res)


def reload(package, desired=None):
    loaded = package in sys.modules
    if not loaded:
        return
    current = sys.modules[package].__version__ if hasattr(sys.modules[package], "__version__") else None
    if desired is not None and current == desired:
        return
    modules = [m for m in sys.modules if m.startswith(package)]
    for m in modules:
        del sys.modules[m]
    sys.modules[package] = importlib.import_module(package)
    log.debug(f'Reload: package={package} version={sys.modules[package].__version__ if hasattr(sys.modules[package], "__version__") else "N/A"}')


def ensure_base_requirements():
    t_start = time.time()
    setuptools_version = '69.5.1'

    def update_setuptools():
        local_log = logging.getLogger('sdnext.installer')
        global pkg_resources, setuptools, distutils # pylint: disable=global-statement
        # python may ship with incompatible setuptools
        subprocess.run(f'"{sys.executable}" -m pip install setuptools=={setuptools_version}', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # need to delete all references to modules to be able to reload them otherwise python will use cached version
        modules = [m for m in sys.modules if m.startswith('setuptools') or m.startswith('pkg_resources') or m.startswith('distutils')]
        for m in modules:
            del sys.modules[m]
        try:
            setuptools = importlib.import_module('setuptools')
            sys.modules['setuptools'] = setuptools
        except ImportError as e:
            local_log.info(f'Python: version={platform.python_version()} platform={platform.system()} bin="{sys.executable}" venv="{sys.prefix}"')
            local_log.critical(f'Import: setuptools {e}')
            os._exit(1)
        try:
            distutils = importlib.import_module('distutils')
            sys.modules['distutils'] = distutils
        except ImportError as e:
            local_log.info(f'Python: version={platform.python_version()} platform={platform.system()} bin="{sys.executable}" venv="{sys.prefix}"')
            local_log.critical(f'Import: distutils {e}')
            os._exit(1)
        try:
            pkg_resources = importlib.import_module('pkg_resources')
            sys.modules['pkg_resources'] = pkg_resources
        except ImportError as e:
            local_log.info(f'Python: version={platform.python_version()} platform={platform.system()} bin="{sys.executable}" venv="{sys.prefix}"')
            local_log.critical(f'Import: pkg_resources {e}')
            os._exit(1)

    try:
        global pkg_resources, setuptools # pylint: disable=global-statement
        import pkg_resources # pylint: disable=redefined-outer-name
        import setuptools # pylint: disable=redefined-outer-name
        if setuptools.__version__ != setuptools_version:
            update_setuptools()
    except ImportError:
        update_setuptools()

    # used by installler itself so must be installed before requirements
    install('rich==14.1.0', 'rich', quiet=True)
    install('psutil', 'psutil', quiet=True)
    install('requests==2.32.3', 'requests', quiet=True)
    ts('base', t_start)


def install_gradio():
    # pip install gradio==3.43.2 installs:
    # aiofiles-23.2.1 altair-5.5.0 annotated-types-0.7.0 anyio-4.9.0 attrs-25.3.0 certifi-2025.6.15 charset_normalizer-3.4.2 click-8.2.1 contourpy-1.3.2 cycler-0.12.1 fastapi-0.115.14 ffmpy-0.6.0 filelock-3.18.0 fonttools-4.58.4 fsspec-2025.5.1 gradio-3.43.2 gradio-client-0.5.0 h11-0.16.0 hf-xet-1.1.5 httpcore-1.0.9 httpx-0.28.1 huggingface-hub-0.33.1 idna-3.10 importlib-resources-6.5.2 jinja2-3.1.6 jsonschema-4.24.0 jsonschema-specifications-2025.4.1 kiwisolver-1.4.8 markupsafe-2.1.5 matplotlib-3.10.3 narwhals-1.45.0 numpy-1.26.4 orjson-3.10.18 packaging-25.0 pandas-2.3.0 pillow-10.4.0 pydantic-2.11.7 pydantic-core-2.33.2 pydub-0.25.1 pyparsing-3.2.3 python-dateutil-2.9.0.post0 python-multipart-0.0.20 pytz-2025.2 pyyaml-6.0.2 referencing-0.36.2 requests-2.32.4 rpds-py-0.25.1 semantic-version-2.10.0 six-1.17.0 sniffio-1.3.1 starlette-0.46.2 tqdm-4.67.1 typing-extensions-4.14.0 typing-inspection-0.4.1 tzdata-2025.2 urllib3-2.5.0 uvicorn-0.35.0 websockets-11.0.3
    install('gradio==3.43.2', no_deps=True)
    install('gradio-client==0.5.0', no_deps=True, quiet=True)
    install('dctorch==0.1.2', no_deps=True, quiet=True)
    pkgs = ['fastapi', 'websockets', 'aiofiles', 'ffmpy', 'pydub', 'uvicorn', 'semantic-version', 'altair', 'python-multipart', 'matplotlib']
    for pkg in pkgs:
        if not installed(pkg, quiet=True):
            install(pkg, quiet=True)


def install_pydantic():
    if args.new:
        install('pydantic==2.11.7', ignore=True, quiet=True)
        reload('pydantic', '2.11.7')
    else:
        install('pydantic==1.10.21', ignore=True, quiet=True)
        reload('pydantic', '1.10.21')


def install_opencv():
    install('opencv-python==4.12.0.88', ignore=True, quiet=True)
    install('opencv-python-headless==4.12.0.88', ignore=True, quiet=True)
    install('opencv-contrib-python==4.12.0.88', ignore=True, quiet=True)
    install('opencv-contrib-python-headless==4.12.0.88', ignore=True, quiet=True)


def install_insightface():
    install('git+https://github.com/deepinsight/insightface@29b6cd65aa0e9ae3b6602de3c52e9d8949c8ee86#subdirectory=python-package', 'insightface') # insightface==0.7.3 with patches
    if args.new:
        uninstall('albumentations')
        install('albumentationsx')
    else:
        uninstall('albumentationsx')
        install('albumentations==1.4.3', ignore=True, quiet=True)
    install_pydantic()


def install_optional():
    t_start = time.time()
    log.info('Installing optional requirements...')
    install('--no-build-isolation git+https://github.com/Disty0/BasicSR@23c1fb6f5c559ef5ce7ad657f2fa56e41b121754', 'basicsr', ignore=True, quiet=True)
    install('--no-build-isolation git+https://github.com/Disty0/GFPGAN@ae0f7e44fafe0ef4716f3c10067f8f379b74c21c', 'gfpgan', ignore=True, quiet=True)
    install('av', ignore=True, quiet=True)
    install('beautifulsoup4', ignore=True, quiet=True)
    install('clean-fid', ignore=True, quiet=True)
    install('clip_interrogator==0.6.0', ignore=True, quiet=True)
    install('Cython', ignore=True, quiet=True)
    install('gguf', ignore=True, quiet=True)
    install('hf_transfer', ignore=True, quiet=True)
    install('hf_xet', ignore=True, quiet=True)
    install('nvidia-ml-py', ignore=True, quiet=True)
    install('pillow-jxl-plugin==1.3.5', ignore=True, quiet=True)
    install('ultralytics==8.3.40', ignore=True, quiet=True)
    install('git+https://github.com/tencent-ailab/IP-Adapter.git', 'ip_adapter', ignore=True, quiet=True)
    # install('torchao==0.10.0', ignore=True, quiet=True)
    # install('bitsandbytes==0.47.0', ignore=True, quiet=True)
    # install('optimum-quanto==0.2.7', ignore=True, quiet=True)
    try:
        import gguf
        scripts_dir = os.path.join(os.path.dirname(gguf.__file__), '..', 'scripts')
        if os.path.exists(scripts_dir):
            os.rename(scripts_dir, scripts_dir + '_gguf')
    except Exception:
        pass
    ts('optional', t_start)


def install_requirements():
    t_start = time.time()
    if args.skip_requirements and not args.requirements:
        return
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    if int(sys.version_info.minor) >= 13:
        install('audioop-lts')
    if not installed('diffusers', quiet=True): # diffusers are not installed, so run initial installation
        global quick_allowed # pylint: disable=global-statement
        quick_allowed = False
        log.info('Install requirements: this may take a while...')
        pip('install -r requirements.txt')
    if args.optional:
        quick_allowed = False
        install_optional()
    installed('torch', reload=True) # reload packages cache
    log.info('Install: verifying requirements')
    if args.new:
        log.debug('Install: flag=new')
    with open('requirements.txt', 'r', encoding='utf8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '' and not line.startswith('#') and line is not None]
        for line in lines:
            if not installed(line, quiet=True):
                _res = install(line)
    install_pydantic()
    install_opencv()
    if args.profile:
        pr.disable()
        print_profile(pr, 'Requirements')
    ts('requirements', t_start)


# set environment variables controling the behavior of various libraries
def set_environment():
    log.debug('Setting environment tuning')
    os.environ.setdefault('ACCELERATE', 'True')
    os.environ.setdefault('ATTN_PRECISION', 'fp16')
    os.environ.setdefault('ClDeviceGlobalMemSizeAvailablePercent', '100')
    os.environ.setdefault('CUDA_AUTO_BOOST', '1')
    os.environ.setdefault('CUDA_CACHE_DISABLE', '0')
    os.environ.setdefault('CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT', '0')
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('FORCE_CUDA', '1')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('K_DIFFUSION_USE_COMPILE', '0')
    os.environ.setdefault('KINETO_LOG_LEVEL', '3')
    os.environ.setdefault('NEOReadDebugKeys', '1')
    os.environ.setdefault('NUMEXPR_MAX_THREADS', '16')
    os.environ.setdefault('PYTHONHTTPSVERIFY', '0')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('PYTORCH_ENABLE_XPU_FALLBACK', '1')
    os.environ.setdefault('RUNAI_STREAMER_CHUNK_BYTESIZE', '2097152')
    os.environ.setdefault('RUNAI_STREAMER_LOG_LEVEL', 'DEBUG' if os.environ.get('SD_LOAD_DEBUG') else 'WARNING')
    os.environ.setdefault('RUNAI_STREAMER_MEMORY_LIMIT', '-1')
    os.environ.setdefault('SAFETENSORS_FAST_GPU', '1')
    os.environ.setdefault('SYCL_CACHE_PERSISTENT', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', '0')
    os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
    os.environ.setdefault('UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS', '1')
    os.environ.setdefault('USE_TORCH', '1')
    os.environ.setdefault('UV_INDEX_STRATEGY', 'unsafe-any-match')
    os.environ.setdefault('UV_NO_BUILD_ISOLATION', '1')
    os.environ.setdefault('UVICORN_TIMEOUT_KEEP_ALIVE', '60')
    allocator = f'garbage_collection_threshold:{opts.get("torch_gc_threshold", 80)/100:0.2f},max_split_size_mb:512'
    if opts.get("torch_malloc", "native") == 'cudaMallocAsync':
        allocator += ',backend:cudaMallocAsync'
    if opts.get("torch_expandable_segments", False):
        allocator += ',expandable_segments:True'
    os.environ.setdefault('PYTORCH_ALLOC_CONF', allocator)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', allocator)
    os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', allocator)
    log.debug(f'Torch allocator: "{allocator}"')


def check_extensions():
    newest_all = os.path.getmtime('requirements.txt')
    from modules.paths import extensions_builtin_dir, extensions_dir
    extension_folders = [extensions_builtin_dir] if args.safe else [extensions_builtin_dir, extensions_dir]
    disabled_extensions_all = opts.get('disable_all_extensions', 'none')
    if disabled_extensions_all != 'none':
        log.info(f'Extensions: disabled={disabled_extensions_all}')
    else:
        log.info(f'Extensions: disabled={opts.get("disabled_extensions", [])}')
    for folder in extension_folders:
        if not os.path.isdir(folder):
            continue
        extensions = list_extensions_folder(folder)
        for ext in extensions:
            newest = 0
            extension_dir = os.path.join(folder, ext)
            if not os.path.isdir(extension_dir):
                log.debug(f'Extension listed as installed but folder missing: {extension_dir}')
                continue
            for f in os.listdir(extension_dir):
                if '.json' in f or '.csv' in f or '__pycache__' in f:
                    continue
                mtime = os.path.getmtime(os.path.join(extension_dir, f))
                newest = max(newest, mtime)
            newest_all = max(newest_all, newest)
            # log.debug(f'Extension version: {time.ctime(newest)} {folder}{os.path.sep}{ext}')
    return round(newest_all)


def get_version(force=False):
    t_start = time.time()
    if (version is None) or (version.get('branch', 'unknown') == 'unknown') or force:
        try:
            subprocess.run('git config log.showsignature false', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
        except Exception:
            pass
        try:
            res = subprocess.run('git log --pretty=format:"%h %ad" -1 --date=short', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            ver = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else '  '
            commit, updated = ver.split(' ')
            version['commit'], version['updated'] = commit, updated
        except Exception as e:
            log.warning(f'Version: where=commit {e}')
        try:
            res = subprocess.run('git remote get-url origin', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            origin = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            branch_name = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            version['url'] = origin.replace('\n', '').removesuffix('.git') + '/tree/' + branch_name.replace('\n', '')
            version['branch'] = branch_name.replace('\n', '')
            if version['branch'] == 'HEAD':
                log.warning('Version: detached state detected')
        except Exception as e:
            log.warning(f'Version: where=branch {e}')
        cwd = os.getcwd()
        try:
            if os.path.exists('extensions-builtin/sdnext-modernui'):
                os.chdir('extensions-builtin/sdnext-modernui')
                res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
                branch_ui = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
                branch_ui = 'dev' if 'dev' in branch_ui else 'main'
                version['ui'] = branch_ui
            else:
                version['ui'] = 'unavailable'
        except Exception as e:
            log.warning(f'Version: where=modernui {e}')
            version['ui'] = 'unknown'
        finally:
            os.chdir(cwd)
        try:
            if os.environ.get('SD_KANVAS_DISABLE', None) is not None:
                version['kanvas'] = 'disabled'
            elif os.path.exists('extensions-builtin/sdnext-kanvas'):
                os.chdir('extensions-builtin/sdnext-kanvas')
                res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
                branch_kanvas = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
                branch_kanvas = 'dev' if 'dev' in branch_kanvas else 'main'
                version['kanvas'] = branch_kanvas
            else:
                version['kanvas'] = 'unavailable'
        except Exception as e:
            log.warning(f'Version: where=kanvas {e}')
            version['kanvas'] = 'unknown'
        finally:
            os.chdir(cwd)
    ts('version', t_start)
    return version


def check_ui(ver):
    def same(ver):
        core = ver['branch'] if ver is not None and 'branch' in ver else 'unknown'
        ui = ver['ui'] if ver is not None and 'ui' in ver else 'unknown'
        return (core == ui) or (core == 'master' and ui == 'main') or (core == 'dev' and ui == 'dev') or (core == 'HEAD')

    t_start = time.time()
    if not same(ver):
        log.debug(f'Branch mismatch: sdnext={ver["branch"]} ui={ver["ui"]}')
        cwd = os.getcwd()
        try:
            os.chdir('extensions-builtin/sdnext-modernui')
            target = 'dev' if 'dev' in ver['branch'] else 'main'
            git('checkout ' + target, ignore=True, optional=True)
            os.chdir(cwd)
            ver = get_version(force=True)
            if not same(ver):
                log.debug(f'Branch synchronized: {ver["branch"]}')
            else:
                log.debug(f'Branch sync failed: sdnext={ver["branch"]} ui={ver["ui"]}')
        except Exception as e:
            log.debug(f'Branch switch: {e}')
        os.chdir(cwd)
    ts('ui', t_start)


def check_venv():
    def try_relpath(p):
        try:
            return os.path.relpath(p)
        except ValueError:
            return p

    t_start = time.time()
    import site
    pkg_path = [try_relpath(p) for p in site.getsitepackages() if os.path.exists(p)]
    log.debug(f'Packages: prefix={try_relpath(sys.prefix)} site={pkg_path}')
    for p in pkg_path:
        invalid = []
        for f in os.listdir(p):
            if f.startswith('~'):
                invalid.append(f)
        if len(invalid) > 0:
            log.warning(f'Packages: site="{p}" invalid={invalid} removing')
        for f in invalid:
            fn = os.path.join(p, f)
            try:
                if os.path.isdir(fn):
                    shutil.rmtree(fn)
                elif os.path.isfile(fn):
                    os.unlink(fn)
            except Exception as e:
                log.error(f'Packages: site={p} invalid={f} error={e}')
    ts('venv', t_start)


# check version of the main repo and optionally upgrade it
def check_version(reset=True): # pylint: disable=unused-argument
    if opts.get('offline_mode', False):
        log.warning('Offline mode enabled')
        args.skip_git = True # pylint: disable=attribute-defined-outside-init
        args.skip_all = True # pylint: disable=attribute-defined-outside-init
        return
    t_start = time.time()
    if args.skip_all:
        return
    if not os.path.exists('.git'):
        log.warning('Not a git repository')
        args.skip_git = True # pylint: disable=attribute-defined-outside-init
    ver = get_version()
    log.info(f'Version: {print_dict(ver)}')
    branch_name = ver.get('branch', None) if ver is not None else 'master'
    if branch_name is None or branch_name == 'unknown':
        branch_name = 'master'
    if args.version or args.skip_git:
        return
    check_ui(ver)
    commit = git('rev-parse HEAD')
    global git_commit # pylint: disable=global-statement
    git_commit = commit[:7]
    if args.quick:
        return
    try:
        import requests
    except ImportError:
        return
    commits = None
    branch_names = []
    try:
        branches = requests.get('https://api.github.com/repos/vladmandic/sdnext/branches', timeout=10).json()
        branch_names = [b['name'] for b in branches if 'name' in b]
        log.trace(f'Repository branches: active={branch_name} available={branch_names}')
    except Exception as e:
        log.error(f'Repository: failed to get branches: {e}')
        return
    if branch_name not in branch_names:
        log.warning(f'Repository: branch={branch_name} skipping update')
        ts('latest', t_start)
        return
    try:
        commits = requests.get(f'https://api.github.com/repos/vladmandic/sdnext/branches/{branch_name}', timeout=10).json()
        latest = commits['commit']['sha']
        if len(latest) != 40:
            log.error(f'Repository error: commit={latest} invalid')
        elif latest != commit and args.upgrade:
            global quick_allowed # pylint: disable=global-statement
            quick_allowed = False
            log.info('Updating main repository')
            try:
                git('add .')
                git('stash')
                update('.', keep_branch=True)
                # git('git stash pop')
                ver = git('log -1 --pretty=format:"%h %ad"')
                log.info(f'Repository upgraded: {ver}')
                log.warning('Server restart is recommended to apply changes')
                if ver == latest: # double check
                    restart()
            except Exception:
                if not reset:
                    log.error('Repository error upgrading')
                else:
                    log.warning('Repository: retrying upgrade...')
                    git_reset()
                    check_version(reset=False)
        else:
            dt = commits["commit"]["commit"]["author"]["date"]
            commit = commits["commit"]["sha"][:8]
            log.info(f'Version: app=sd.next latest={dt} hash={commit} branch={branch_name}')
    except Exception as e:
        log.error(f'Repository failed to check version: {e} {commits}')
    ts('latest', t_start)


def update_wiki():
    t_start = time.time()
    if args.upgrade:
        log.info('Updating Wiki')
        try:
            update(os.path.join(os.path.dirname(__file__), "wiki"))
        except Exception:
            log.error('Wiki update error')
    ts('wiki', t_start)


# check if we can run setup in quick mode
def check_timestamp():
    if not quick_allowed or not os.path.isfile(log_file):
        return False
    if args.quick:
        return True
    if args.skip_git:
        return True
    ok = True
    setup_time = -1
    version_time = -1
    with open(log_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            if 'Setup complete without errors' in line:
                setup_time = int(line.split(' ')[-1])
    try:
        version_time = git('log -1 --pretty=format:"%at"')
        version_time = ''.join(filter(str.isdigit, version_time))
        version_time = int(version_time) if len(version_time) > 0 else -1
        log.debug(f'Timestamp repository update time: {time.ctime(version_time)}')
    except Exception as e:
        log.error(f'Timestamp local repository version: {e}')
    if setup_time == -1:
        return False
    log.debug(f'Timestamp previous setup time: {time.ctime(setup_time)}')
    if setup_time < version_time or version_time == -1:
        ok = False
    extension_time = check_extensions()
    log.debug(f'Timestamp latest extensions time: {time.ctime(extension_time)}')
    if setup_time < extension_time:
        ok = False
    log.debug(f'Timestamp: version:{version_time} setup:{setup_time} extension:{extension_time}')
    if args.reinstall:
        ok = False
    return ok


def add_args(parser):
    import argparse
    group_install = parser.add_argument_group('Install')
    group_install.add_argument('--quick', default=os.environ.get("SD_QUICK",False), action='store_true', help="Bypass version checks, default: %(default)s")
    group_install.add_argument('--reset', default=os.environ.get("SD_RESET",False), action='store_true', help="Reset main repository to latest version, default: %(default)s")
    group_install.add_argument('--upgrade', '--update', default=os.environ.get("SD_UPGRADE",False), action='store_true', help="Upgrade main repository to latest version, default: %(default)s")
    group_install.add_argument('--requirements', default=os.environ.get("SD_REQUIREMENTS",False), action='store_true', help="Force re-check of requirements, default: %(default)s")
    group_install.add_argument('--reinstall', default=os.environ.get("SD_REINSTALL",False), action='store_true', help="Force reinstallation of all requirements, default: %(default)s")
    group_install.add_argument('--uv', default=os.environ.get("SD_UV",False), action='store_true', help="Use uv instead of pip to install the packages")
    group_install.add_argument('--optional', default=os.environ.get("SD_OPTIONAL",False), action='store_true', help="Force installation of optional requirements, default: %(default)s")
    group_install.add_argument('--skip-requirements', default=os.environ.get("SD_SKIPREQUIREMENTS",False), action='store_true', help="Skips checking and installing requirements, default: %(default)s")
    group_install.add_argument('--skip-extensions', default=os.environ.get("SD_SKIPEXTENSION",False), action='store_true', help="Skips running individual extension installers, default: %(default)s")
    group_install.add_argument('--skip-git', default=os.environ.get("SD_SKIPGIT",False), action='store_true', help="Skips running all GIT operations, default: %(default)s")
    group_install.add_argument('--skip-torch', default=os.environ.get("SD_SKIPTORCH",False), action='store_true', help="Skips running Torch checks, default: %(default)s")
    group_install.add_argument('--skip-all', default=os.environ.get("SD_SKIPALL",False), action='store_true', help="Skips running all checks, default: %(default)s")
    group_install.add_argument('--skip-env', default=os.environ.get("SD_SKIPENV",False), action='store_true', help="Skips setting of env variables during startup, default: %(default)s")

    group_compute = parser.add_argument_group('Compute Engine')
    group_compute.add_argument("--device-id", type=str, default=os.environ.get("SD_DEVICEID", None), help="Select the default GPU device to use, default: %(default)s")
    group_compute.add_argument("--use-cuda", default=os.environ.get("SD_USECUDA",False), action='store_true', help="Force use nVidia CUDA backend, default: %(default)s")
    group_compute.add_argument("--use-ipex", default=os.environ.get("SD_USEIPEX",False), action='store_true', help="Force use Intel OneAPI XPU backend, default: %(default)s")
    group_compute.add_argument("--use-rocm", default=os.environ.get("SD_USEROCM",False), action='store_true', help="Force use AMD ROCm backend, default: %(default)s")
    group_compute.add_argument('--use-zluda', default=os.environ.get("SD_USEZLUDA", False), action='store_true', help="Force use ZLUDA, AMD GPUs only, default: %(default)s")
    group_compute.add_argument("--use-openvino", default=os.environ.get("SD_USEOPENVINO",False), action='store_true', help="Use Intel OpenVINO backend, default: %(default)s")
    group_compute.add_argument('--use-directml', default=os.environ.get("SD_USEDIRECTML",False), action='store_true', help="Use DirectML if no compatible GPU is detected, default: %(default)s")
    group_compute.add_argument("--use-xformers", default=os.environ.get("SD_USEXFORMERS",False), action='store_true', help="Force use xFormers cross-optimization, default: %(default)s")
    group_compute.add_argument("--use-nightly", default=os.environ.get("SD_USENIGHTLY",False), action='store_true', help="Force use nightly torch builds, default: %(default)s")

    group_paths = parser.add_argument_group('Paths')
    group_paths.add_argument("--ckpt", type=str, default=os.environ.get("SD_MODEL", None), help="Path to model checkpoint to load immediately, default: %(default)s")
    group_paths.add_argument("--data-dir", type=str, default=os.environ.get("SD_DATADIR", ''), help="Base path where all user data is stored, default: %(default)s")
    group_paths.add_argument("--models-dir", type=str, default=os.environ.get("SD_MODELSDIR", 'models'), help="Base path where all models are stored, default: %(default)s",)
    group_paths.add_argument("--extensions-dir", type=str, default=os.environ.get("SD_EXTENSIONSDIR", None), help="Base path where all extensions are stored, default: %(default)s",)

    group_ui = parser.add_argument_group('UI')
    group_ui.add_argument('--theme', type=str, default=os.environ.get("SD_THEME", None), help='Override UI theme')
    group_ui.add_argument('--locale', type=str, default=os.environ.get("SD_LOCALE", None), help='Override UI locale')

    group_http = parser.add_argument_group('HTTP')
    group_http.add_argument("--server-name", type=str, default=os.environ.get("SD_SERVERNAME", None), help="Sets hostname of server, default: %(default)s")
    group_http.add_argument("--tls-keyfile", type=str, default=os.environ.get("SD_TLSKEYFILE", None), help="Enable TLS and specify key file, default: %(default)s")
    group_http.add_argument("--tls-certfile", type=str, default=os.environ.get("SD_TLSCERTFILE", None), help="Enable TLS and specify cert file, default: %(default)s")
    group_http.add_argument("--tls-selfsign", action="store_true", default=os.environ.get("SD_TLSSELFSIGN", False), help="Enable TLS with self-signed certificates, default: %(default)s")
    group_http.add_argument("--cors-origins", type=str, default=os.environ.get("SD_CORSORIGINS", None), help="Allowed CORS origins as comma-separated list, default: %(default)s")
    group_http.add_argument("--cors-regex", type=str, default=os.environ.get("SD_CORSREGEX", None), help="Allowed CORS origins as regular expression, default: %(default)s")
    group_http.add_argument('--subpath', type=str, default=os.environ.get("SD_SUBPATH", None), help='Customize the URL subpath for usage with reverse proxy')
    group_http.add_argument("--autolaunch", default=os.environ.get("SD_AUTOLAUNCH", False), action='store_true', help="Open the UI URL in the system's default browser upon launch")
    group_http.add_argument("--auth", type=str, default=os.environ.get("SD_AUTH", None), help='Set access authentication like "user:pwd,user:pwd""')
    group_http.add_argument("--auth-file", type=str, default=os.environ.get("SD_AUTHFILE", None), help='Set access authentication using file, default: %(default)s')
    group_http.add_argument("--allowed-paths", nargs='+', default=[], type=str, required=False, help="add additional paths to paths allowed for web access")
    group_http.add_argument("--share", default=os.environ.get("SD_SHARE", False), action='store_true', help="Enable UI accessible through Gradio site, default: %(default)s")
    group_http.add_argument("--insecure", default=os.environ.get("SD_INSECURE", False), action='store_true', help="Enable extensions tab regardless of other options, default: %(default)s")
    group_http.add_argument("--listen", default=os.environ.get("SD_LISTEN", False), action='store_true', help="Launch web server using public IP address, default: %(default)s")
    group_http.add_argument("--port", type=int, default=os.environ.get("SD_PORT", 7860), help="Launch web server with given server port, default: %(default)s")

    group_diag = parser.add_argument_group('Diagnostics')
    group_diag.add_argument('--experimental', default=os.environ.get("SD_EXPERIMENTAL",False), action='store_true', help="Allow unsupported versions of libraries, default: %(default)s")
    group_diag.add_argument('--ignore', default=os.environ.get("SD_IGNORE",False), action='store_true', help="Ignore any errors and attempt to continue")
    group_diag.add_argument('--new', default=os.environ.get("SD_NEW",False), action='store_true', help="Force newer/untested version of libraries, default: %(default)s")
    group_diag.add_argument('--safe', default=os.environ.get("SD_SAFE",False), action='store_true', help="Run in safe mode with no user extensions")
    group_diag.add_argument('--test', default=os.environ.get("SD_TEST",False), action='store_true', help="Run test only and exit")
    group_diag.add_argument('--version', default=False, action='store_true', help="Print version information")
    group_diag.add_argument("--monitor", default=os.environ.get("SD_MONITOR", 0), help="Run memory monitor, default: %(default)s")
    group_diag.add_argument("--status", default=os.environ.get("SD_STATUS", 120), help="Run server is-alive status, default: %(default)s")

    group_log = parser.add_argument_group('Logging')
    group_log.add_argument("--log", type=str, default=os.environ.get("SD_LOG", None), help="Set log file, default: %(default)s")
    group_log.add_argument('--debug', default=not os.environ.get("SD_NODEBUG",False), action='store_true', help="Run with debug logging, default: %(default)s")
    group_log.add_argument("--trace", default=os.environ.get("SD_TRACE", False), action='store_true', help="Run with trace logging, default: %(default)s")
    group_log.add_argument("--profile", default=os.environ.get("SD_PROFILE", False), action='store_true', help="Run profiler, default: %(default)s")
    group_log.add_argument('--docs', default=not os.environ.get("SD_NODOCS", False), action='store_true', help = "Mount API docs, default: %(default)s")
    group_log.add_argument("--api-log", default=not os.environ.get("SD_NOAPILOG", False), action='store_true', help="Log all API requests")

    group_nargs = parser.add_argument_group('Other')
    group_nargs.add_argument('args', type=str, nargs='*', help=argparse.SUPPRESS)


def parse_args(parser):
    # command line args
    global args # pylint: disable=global-statement
    if "USED_VSCODE_COMMAND_PICKARGS" in os.environ:
        import shlex
        argv = shlex.split(" ".join(sys.argv[1:])) if "USED_VSCODE_COMMAND_PICKARGS" in os.environ else sys.argv[1:]
        log.debug('VSCode Launch')
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()
    return args


def extensions_preload(parser):
    t_start = time.time()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
    if args.safe:
        log.info('Running in safe mode without user extensions')
    try:
        from modules.script_loading import preload_extensions
        from modules.paths import extensions_builtin_dir, extensions_dir
        extension_folders = [extensions_builtin_dir] if args.safe else [extensions_builtin_dir, extensions_dir]
        preload_time = {}
        for ext_dir in extension_folders:
            t0 = time.time()
            preload_extensions(ext_dir, parser)
            t1 = time.time()
            preload_time[ext_dir] = round(t1 - t0, 2)
        log.debug(f'Extension preload: {preload_time}')
    except Exception:
        log.error('Error running extension preloading')
    if args.profile:
        pr.disable()
        print_profile(pr, 'Preload')
    ts('preload', t_start)


def git_reset(folder='.'):
    t_start = time.time()
    log.warning('Running GIT reset')
    global quick_allowed # pylint: disable=global-statement
    quick_allowed = False
    b = branch(folder)
    if b is None or b == '':
        b = 'master'
    git('add .')
    git('stash')
    git('merge --abort', folder=None, ignore=True)
    git('fetch --all')
    git(f'reset --hard origin/{b}')
    git(f'checkout {b}')
    git('submodule update --init --recursive')
    git('submodule sync --recursive')
    log.info('GIT reset complete')
    ts('reset', t_start)


def read_options():
    t_start = time.time()
    global opts # pylint: disable=global-statement
    if os.path.isfile(args.config):
        with open(args.config, "r", encoding="utf8") as file:
            try:
                opts = json.load(file)
                if type(opts) is str:
                    opts = json.loads(opts)
            except Exception as e:
                log.error(f'Error reading options file: {file} {e}')
    ts('options', t_start)
