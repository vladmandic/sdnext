#!/usr/bin/env python

import os
import sys
import time
import shlex
import subprocess
from functools import lru_cache
import installer


debug_install = installer.log.debug if os.environ.get('SD_INSTALL_DEBUG', None) is not None else lambda *args, **kwargs: None
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
    installer.log.info(f'Command line args: {sys.argv[1:]} {installer.print_dict(custom)}')
    if os.environ.get('SD_ENV_DEBUG', None) is not None:
        env = os.environ.copy()
        if 'PATH' in env:
            del env['PATH']
        if 'PS1' in env:
            del env['PS1']
        installer.log.trace(f'Environment: {installer.print_dict(env)}')
    env = [f'{k}={v}' for k, v in os.environ.items() if k.startswith('SD_')]
    installer.log.debug(f'Env flags: {env}')
    ldpreload = os.environ.get('LD_PRELOAD', None)
    ldpath = os.environ.get('LD_LIBRARY_PATH', None)
    if ldpreload is not None or ldpath is not None:
        installer.log.debug(f'Linker flags: preload="{ldpreload}" path="{ldpath}"')
    rec('args')


@lru_cache()
def commit_hash(): # compatbility function
    global stored_commit_hash # pylint: disable=global-statement
    if stored_commit_hash is not None:
        return stored_commit_hash
    try:
        stored_commit_hash = run(f"{git} rev-parse HEAD").strip()
    except Exception:
        stored_commit_hash = "<none>"
    rec('commit')
    return stored_commit_hash


@lru_cache()
def run(command, desc=None, errdesc=None, custom_env=None, live=False): # compatbility function
    if desc is not None:
        installer.log.info(desc)
    if live:
        result = subprocess.run(command, check=False, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'} Command: {command} Error code: {result.returncode}""")
        return ''
    result = subprocess.run(command, stdout=subprocess.PIPE, check=False, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)
    if result.returncode != 0:
        raise RuntimeError(f"""{errdesc or 'Error running command'}: {command} code: {result.returncode}
{result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else ''}
{result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else ''}
""")
    return result.stdout.decode(encoding="utf8", errors="ignore")


def check_run(command): # compatbility function
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


@lru_cache()
def is_installed(pkg): # compatbility function
    return installer.installed(pkg)


@lru_cache()
def repo_dir(name): # compatbility function
    return os.path.join(script_path, dir_repos, name)


@lru_cache()
def run_python(code, desc=None, errdesc=None): # compatbility function
    return run(f'"{sys.executable}" -c "{code}"', desc, errdesc)


@lru_cache()
def run_pip(pkg, desc=None): # compatbility function
    forbidden = ['onnxruntime', 'opencv-python']
    if desc is None:
        desc = pkg
    for f in forbidden:
        if f in pkg:
            debug_install('Blocked package installation: package={f}')
            return True
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{sys.executable}" -m pip {pkg} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


@lru_cache()
def check_run_python(code): # compatbility function
    return check_run(f'"{sys.executable}" -c "{code}"')


def git_clone(url, tgt, _name, commithash=None): # compatbility function
    installer.clone(url, tgt, commithash)


def run_extension_installer(ext_dir): # compatbility function
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
                         'framepack', 'nudenet', 'agent_scheduler', 'basicsr', 'gfpgan', 'war',
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
    modules_keys = [m.split('.')[0] for m in modules_cleaned if not m.startswith('_')]
    modules_sorted = {}
    for module_key in modules_keys:
        modules_sorted[module_key] = len([m for m in modules_cleaned if m.startswith(module_key)])
    installer.log.trace(f'Server modules: {modules_sorted}')
    t1 = time.time()
    installer.log.trace(f'Server modules: total={len(modules_loaded)} unloaded={len(removed_removed)} remaining={len(modules_cleaned)} gc={collected} time={t1-t0:.2f}')


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
        installer.log.debug(f'Memory: {get_memory_stats()} collected={collected}')
    module_spec = importlib.util.spec_from_file_location('webui', 'webui.py')
    server = importlib.util.module_from_spec(module_spec)
    installer.log.debug(f'Starting module: {server}')
    module_spec.loader.exec_module(server)
    uvicorn = None
    if args.test:
        installer.log.info("Test only")
        installer.log.critical('Logging: level=critical')
        installer.log.error('Logging: level=error')
        installer.log.warning('Logging: level=warning')
        installer.log.info('Logging: level=info')
        installer.log.debug('Logging: level=debug')
        installer.log.trace('Logging: level=trace')
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
    installer.ensure_base_requirements()
    init_args() # setup argparser and default folders
    installer.args = args
    installer.setup_logging()
    installer.log.info('Starting SD.Next')
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
        installer.log.info('Skipping GIT operations')
    installer.check_version()
    installer.log.info(f'Platform: {installer.print_dict(installer.get_platform())}')
    installer.check_venv()
    installer.log.info(f'Args: {sys.argv[1:]}')
    if not args.skip_env or args.skip_all:
        installer.set_environment()
    if args.uv:
        installer.install("uv", "uv")
    installer.install_gradio()
    installer.check_torch()
    installer.check_onnx()
    installer.check_transformers()
    installer.check_diffusers()
    installer.install_sentencepiece()
    installer.check_modified_files()
    if args.test:
        installer.log.info('Startup: test mode')
        installer.quick_allowed = False
    if args.reinstall:
        installer.log.info('Startup: force reinstall of all packages')
        installer.quick_allowed = False
    if args.skip_all:
        installer.log.info('Startup: skip all')
        installer.quick_allowed = True
        init_paths()
    else:
        installer.install_requirements()
        installer.install_packages()
        if installer.check_timestamp():
            installer.log.info('Startup: quick launch')
            init_paths()
            installer.check_extensions()
        else:
            installer.log.info('Startup: standard')
            installer.install_submodules()
            init_paths()
            installer.install_extensions()
            installer.install_requirements() # redo requirements since extensions may change them
            installer.update_wiki()
            if len(installer.errors) == 0:
                installer.log.debug(f'Setup complete without errors: {round(time.time())}')
            else:
                installer.log.warning(f'Setup complete with errors: {installer.errors}')
                installer.log.warning(f'See log file for more details: {installer.log_file}')
    installer.extensions_preload(parser) # adds additional args from extensions
    args = installer.parse_args(parser)
    installer.log.info(f'Installer time: {init_summary()}')
    get_custom_args()

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
        if float(args.status) > 0 and (t_current - t_server) > float(args.status):
            s = instance.state.status()
            if (s.timestamp is None) or (s.step == 0): # dont spam during active job
                installer.log.trace(f'Server: alive={alive} requests={requests} memory={get_memory_stats()} {s}')
            t_server = t_current
        if float(args.monitor) > 0 and t_current - t_monitor > float(args.monitor):
            installer.log.trace(f'Monitor: {get_memory_stats(detailed=True)}')
            t_monitor = t_current
        if not alive:
            if uv is not None and uv.wants_restart:
                clean_server()
                installer.log.info('Server restarting...')
                # uv, instance = start_server(immediate=False, server=instance)
                os.execv(sys.executable, ['python'] + sys.argv)
            else:
                installer.log.info('Exiting...')
                break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
