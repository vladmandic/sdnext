#!/usr/bin/env python

from functools import lru_cache
import os
import sys
import time
import shlex
import subprocess

import installer
from modules.logger import log


debug_install = log.debug if os.environ.get('SD_INSTALL_DEBUG', None) is not None else lambda *args, **kwargs: None
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")
sys.argv += shlex.split(commandline_args)
args = None
parser = None
script_path = None
extensions_dir = None
git = os.environ.get('GIT', "git")
index_url = os.environ.get('INDEX_URL', "")
stored_commit_hash = None
dir_repos = "repositories"
python = sys.executable # used by some extensions to run python
skip_install = False # parsed by some extensions


try:
    from modules.timer import launch, init
    rec = launch.record
    init_summary = init.summary
except Exception:
    rec = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment
    init_summary = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment


def init_args():
    global parser, args # pylint: disable=global-statement
    import modules.cmd_args
    modules.cmd_args.parse_args()
    parser = modules.cmd_args.parser
    installer.add_args(parser)
    args, _ = parser.parse_known_args()
    rec('args')


def init_paths():
    global script_path, extensions_dir # pylint: disable=global-statement
    import modules.paths
    script_path = modules.paths.script_path
    extensions_dir = modules.paths.extensions_dir
    sys.path.insert(0, script_path)
    rec('paths')


def get_custom_args():
    custom = {}
    for arg in vars(args):
        default = parser.get_default(arg)
        current = getattr(args, arg)
        if current != default:
            custom[arg] = getattr(args, arg)
    log.info(f'Command line args: {sys.argv[1:]} {installer.print_dict(custom)}')
    if os.environ.get('SD_ENV_DEBUG', None) is not None:
        env = os.environ.copy()
        if 'PATH' in env:
            del env['PATH']
        if 'PS1' in env:
            del env['PS1']
        log.trace(f'Environment: {installer.print_dict(env)}')
    env = [f'{k}={v}' for k, v in os.environ.items() if k.startswith('SD_')]
    ld = [f'{k}={v}' for k, v in os.environ.items() if k.startswith('LD_')]
    log.debug(f'Flags: sd={env} ld={ld}')
    rec('args')


@lru_cache
def commit_hash(): # compatibility function
    global stored_commit_hash # pylint: disable=global-statement
    if stored_commit_hash is not None:
        return stored_commit_hash
    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"
    rec('commit')
    return stored_commit_hash


@lru_cache
def run(command, desc=None, errdesc=None, custom_env=None, live=False): # compatibility function
    if desc is not None:
        log.info(desc)
    if live:
        result = subprocess.run(command, check=False, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'} Command: {command} Error code: {result.returncode}""")
        return ''
    result = subprocess.run(command, capture_output=True, check=False, shell=True, env=os.environ if custom_env is None else custom_env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"""{errdesc or 'Error running command'}: {command} code: {result.returncode}
{result.stdout}
{result.stderr}
""")
    return result.stdout


def check_run(command): # compatibility function
    result = subprocess.run(command, check=False, capture_output=True, shell=True)
    return result.returncode == 0


@lru_cache
def is_installed(pkg): # compatibility function
    return installer.installed(pkg)


@lru_cache
def repo_dir(name: str): # compatibility function
    return os.path.join(script_path, dir_repos, name)


@lru_cache
def run_python(code, desc=None, errdesc=None): # compatibility function
    return run(f'"{sys.executable}" -c "{code}"', desc, errdesc)


@lru_cache
def run_pip(pkg, desc=None): # compatibility function
    forbidden = ['onnxruntime', 'opencv-python']
    if desc is None:
        desc = pkg
    for f in forbidden:
        if f in pkg:
            debug_install(f'Blocked package installation: package={f}')
            return True
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{sys.executable}" -m pip {pkg} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


@lru_cache
def check_run_python(code): # compatibility function
    return check_run(f'"{sys.executable}" -c "{code}"')


def git_clone(url, tgt, _name, commithash=None): # compatibility function
    installer.clone(url, tgt, commithash)


def run_extension_installer(ext_dir): # compatibility function
    installer.run_extension_installer(ext_dir)


def get_memory_stats(detailed:bool=False):
    from modules.memstats import ram_stats, memory_stats
    if not detailed:
        res = ram_stats()
        return f'{res["used"]}/{res["total"]}'
    else:
        res = memory_stats()
        return res


def clean_server():
    t0 = time.time()
    import gc
    modules_loaded = sorted(sys.modules.keys())
    modules_to_remove = ['webui', 'modules', 'scripts', 'gradio',
                         'onnx', 'torch', 'pytorch', 'lightning', 'tensor', 'diffusers', 'transformers', 'tokenize', 'safetensors', 'gguf', 'accelerate', 'peft', 'triton', 'huggingface',
                         'PIL', 'cv2', 'timm', 'numpy', 'scipy', 'sympy', 'sklearn', 'skimage', 'sqlalchemy', 'flash_attn', 'bitsandbytes', 'xformers', 'matplotlib', 'optimum', 'pandas', 'pi', 'git', 're', 'altair',
                         'framepack', 'nudenet', 'agent_scheduler', 'basicsr', 'war',
                         'fastapi', 'urllib', 'uvicorn', 'web', 'http', 'google', 'starlette', 'socket']
    removed_removed = []
    for module_loaded in modules_loaded:
        for module_to_remove in modules_to_remove:
            if module_loaded.startswith(module_to_remove):
                try:
                    del sys.modules[module_loaded]
                    removed_removed.append(module_loaded)
                except Exception:
                    pass
    collected = gc.collect() # python gc
    modules_cleaned = sorted(sys.modules.keys())
    # modules_keys = [m.split('.')[0] for m in modules_cleaned if not m.startswith('_')]
    # modules_sorted = {}
    # for module_key in modules_keys:
    #     modules_sorted[module_key] = len([m for m in modules_cleaned if m.startswith(module_key)])
    # log.trace(f'Server modules: {modules_sorted}')
    t1 = time.time()
    log.trace(f'Server modules: total={len(modules_loaded)} unloaded={len(removed_removed)} remaining={len(modules_cleaned)} gc={collected} time={t1-t0:.2f}')


def start_server(immediate=True, server=None):
    if args.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    import gc
    import importlib.util
    collected = 0
    if server is not None:
        server = None
        collected = gc.collect()
    if not immediate:
        time.sleep(3)
    if collected > 0:
        log.debug(f'Memory: {get_memory_stats()} collected={collected}')
    module_spec = importlib.util.spec_from_file_location('webui', 'webui.py')
    server = importlib.util.module_from_spec(module_spec)
    log.debug(f'Starting module: {server}')
    module_spec.loader.exec_module(server)

    uvicorn = None
    if args.test:
        log.info("Test only")
        log.critical('Logging: level=critical')
        log.error('Logging: level=error')
        log.warning('Logging: level=warning')
        log.info('Logging: level=info')
        log.debug('Logging: level=debug')
        log.trace('Logging: level=trace')
        server.wants_restart = False
    else:
        uvicorn = server.webui(restart=not immediate)
    if args.profile:
        pr.disable()
        installer.print_profile(pr, 'WebUI')
    rec('server')
    return uvicorn, server


def main():
    global args # pylint: disable=global-statement
    init_args() # setup argparser and default folders
    installer.args = args
    installer.setup_logging(debug=args.debug, trace=args.trace, filename=args.log)
    log.info('Starting SD.Next')
    installer.get_logfile()
    try:
        sys.excepthook = installer.custom_excepthook
    except Exception:
        pass
    installer.read_options()
    if args.skip_all:
        args.quick = True
    installer.check_python()
    if args.reset:
        installer.git_reset()
    if args.skip_git or args.skip_all:
        log.info('Skipping GIT operations')
    log.info(f'Platform: {installer.print_dict(installer.get_platform())}')
    installer.check_version()
    installer.check_venv()
    log.info(f'Args: {sys.argv[1:]}')
    if not args.skip_env and not args.skip_all:
        installer.set_environment()
    if args.uv:
        installer.install('uv', 'uv')
    installer.install_gradio()
    installer.check_torch()
    installer.check_onnx()
    installer.check_transformers()
    installer.check_diffusers()
    installer.check_modified_files()
    if args.test:
        log.info('Startup: mode=test')
        installer.quick_allowed = False
    if args.reinstall:
        log.info('Startup: mode=reinstall')
        installer.quick_allowed = False

    if args.skip_all:
        log.info('Startup: skip=all')
        installer.quick_allowed = True
        init_paths()
    else:
        installer.install_requirements()
        if installer.check_timestamp():
            log.info('Startup: launch=quick')
            init_paths()
            installer.check_extensions()
        else:
            log.info('Startup: launch=full')
            installer.install_submodules()
            init_paths()
            installer.install_extensions()
            installer.install_requirements() # redo requirements since extensions may change them
            if len(installer.errors) == 0:
                log.debug(f'Setup complete without errors: {round(time.time())}')
                installer.update_state()
            else:
                log.warning(f'Setup complete with errors: {installer.errors}')
                log.warning(f'See log file for more details: {installer.log_file}')
        if args.enso and not args.skip_git:
            from modules import enso
            enso.install()
            if args.upgrade:
                enso.update()
    installer.extensions_preload(parser) # adds additional args from extensions
    args = installer.parse_args(parser)
    log.info(f'Installer time: {init_summary()}')
    get_custom_args()
    installer.update_wiki()

    if args.upgrade:
        if installer.restart_required:
            log.warning('Installer: restarting now to apply upgrade...')
            if '--upgrade' in sys.argv:
                sys.argv.remove('--upgrade')
            if '--update' in sys.argv:
                sys.argv.remove('--update')
            installer.restart(sys.argv)
        else:
            log.debug('Installer: no modifications')

    uv, instance = start_server(immediate=True, server=None)
    t_server = time.time()
    t_monitor = time.time()

    while True:
        try:
            alive = uv.thread.is_alive()
            requests = uv.server_state.total_requests if hasattr(uv, 'server_state') else 0
        except Exception:
            alive = False
            requests = 0
        t_current = time.time()
        status_rate = float(args.status) if args.status >= 0 else installer.opts.get('server_status', 120)
        monitor_rate = float(args.monitor) if args.monitor >= 0 else installer.opts.get('server_monitor', 0)
        if float(status_rate) > 0 and (t_current - t_server) > float(status_rate):
            s = instance.state.status()
            if (s.timestamp is None) or (s.step == 0): # dont spam during active job
                log.trace(f'Server: alive={alive} requests={requests} memory={get_memory_stats()} {s}')
            t_server = t_current
        if float(monitor_rate) > 0 and t_current - t_monitor > float(monitor_rate):
            log.trace(f'Monitor: {get_memory_stats(detailed=True)}')
            t_monitor = t_current
            # from modules.api.validate import get_api_stats
            # get_api_stats()
            # from modules import memstats
            # memstats.get_objects()
        if not alive:
            if uv is not None and uv.wants_restart:
                clean_server()
                log.info('Server restarting...')
                # uv, instance = start_server(immediate=False, server=instance)
                os.execv(sys.executable, ['python'] + sys.argv)
            else:
                log.info('Exiting...')
                break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
