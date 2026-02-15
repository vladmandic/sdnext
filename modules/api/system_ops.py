"""System operations API — restart, profiling, update, benchmark.

Provides REST endpoints for system management operations that were previously
only accessible through the Gradio UI.
"""

import os
import json
import threading
from modules import shared


# ---------------------------------------------------------------------------
# Server control
# ---------------------------------------------------------------------------

def post_restart():
    shared.log.info('API: restart request received')
    threading.Timer(1.0, shared.restart_server, kwargs={'restart': True}).start()
    return {"status": "restarting"}


def post_profiling():
    shared.cmd_opts.profile = not shared.cmd_opts.profile
    shared.log.info(f'API: profiling {"enabled" if shared.cmd_opts.profile else "disabled"}')
    return {"enabled": shared.cmd_opts.profile}


# ---------------------------------------------------------------------------
# Update check / apply
# ---------------------------------------------------------------------------

def get_update_check():
    import installer as i
    try:
        origin = i.git('remote get-url origin').splitlines()[0]
        branch = i.git('rev-parse --abbrev-ref HEAD').splitlines()[0]
        url = origin.removesuffix('.git') + '/tree/' + branch
        ver = i.git('log --pretty=format:"%h %ad" -1 --date=short').splitlines()[0]
        current_hash, current_date = ver.split(' ')
        i.git('fetch')
        ver = i.git(f'log origin/{branch} --pretty=format:"%h %ad" -1 --date=short').splitlines()[0]
        latest_hash, latest_date = ver.split(' ')
        return {
            "url": url,
            "branch": branch,
            "current_date": current_date,
            "current_hash": current_hash,
            "latest_date": latest_date,
            "latest_hash": latest_hash,
            "up_to_date": current_hash == latest_hash,
        }
    except Exception as e:
        shared.log.error(f'API update check: {e}')
        return {"error": str(e)}


def post_update_apply(rebase: bool = True, submodules: bool = True, extensions: bool = True):
    import installer as i
    messages = []
    try:
        if rebase:
            i.git('add .')
            i.git('stash')
        res = i.update('.', keep_branch=True, rebase=rebase)
        messages.append(res)
    except Exception as e:
        messages.append(f'Repository upgrade error: {e}')
    if submodules:
        try:
            res = i.install_submodules(force=True)
            messages.append(res)
        except Exception as e:
            messages.append(f'Submodule upgrade error: {e}')
    if extensions:
        try:
            res = i.install_extensions(force=True)
            messages.append(res)
        except Exception as e:
            messages.append(f'Extension upgrade error: {e}')
    # Re-check version after update
    info = get_update_check()
    changed = not info.get("up_to_date", True) if "error" not in info else False
    return {"status": '\n'.join(messages), "changed": changed, "version": info}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def post_benchmark_run(level: str = "quick", steps: str = "normal", width: int = 512, height: int = 512):
    try:
        import importlib
        spec = importlib.util.spec_from_file_location("benchmark", os.path.join(os.path.dirname(__file__), '..', '..', 'extensions-builtin', 'sd-extension-system-info', 'benchmark.py'))
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
    except Exception as e:
        return {"results": [], "error": f"Failed to load benchmark module: {e}"}
    if level == "quick":
        batches = [1]
    elif level == "extensive":
        batches = [1, 2, 4, 8, 16]
    else:
        batches = [1, 2, 4]
    results = []
    for batch in batches:
        its = benchmark.run_benchmark(batch, steps, width, height)
        results.append({"batch": batch, "its": its})
        if its == "error":
            break
    return {"results": results, "error": None}


def get_benchmark_results():
    bench_file = os.path.join(os.path.dirname(__file__), '..', '..', 'extensions-builtin', 'sd-extension-system-info', 'scripts', 'benchmark-data-local.json')
    headers = ['timestamp', 'it/s', 'version', 'system', 'libraries', 'gpu', 'flags', 'settings', 'username', 'note', 'hash']
    if os.path.isfile(bench_file) and os.path.getsize(bench_file) > 0:
        try:
            with open(bench_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"headers": headers, "data": data}
        except Exception as e:
            return {"headers": headers, "data": [], "error": str(e)}
    return {"headers": headers, "data": []}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_api():
    api = shared.api
    api.add_api_route("/sdapi/v1/server/restart", post_restart, methods=["POST"])
    api.add_api_route("/sdapi/v1/server/profiling", post_profiling, methods=["POST"])
    api.add_api_route("/sdapi/v1/update/check", get_update_check, methods=["GET"])
    api.add_api_route("/sdapi/v1/update/apply", post_update_apply, methods=["POST"])
    api.add_api_route("/sdapi/v1/benchmark/run", post_benchmark_run, methods=["POST"])
    api.add_api_route("/sdapi/v1/benchmark/results", get_benchmark_results, methods=["GET"])
