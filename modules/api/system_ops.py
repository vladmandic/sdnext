"""System operations API — restart, profiling, update, benchmark, storage.

Provides REST endpoints for system management operations that were previously
only accessible through the Gradio UI.
"""

import os
import json
import threading
from modules import shared, paths
from modules.logger import log


# ---------------------------------------------------------------------------
# Server control
# ---------------------------------------------------------------------------

def post_restart():
    """
    Restart the server.

    Triggers a graceful restart after a 1-second delay. Returns immediately
    with a status message; the server will restart asynchronously.
    """
    log.info('API: restart request received')
    threading.Timer(1.0, shared.restart_server, kwargs={'restart': True}).start()
    return {"status": "restarting"}


def post_shutdown():
    """
    Shut down the server.

    Triggers a clean exit after a 1-second delay so the HTTP response
    is sent before the process terminates.
    """
    log.info('API: shutdown request received')
    threading.Timer(1.0, os._exit, args=[0]).start()
    return {"status": "shutting_down"}


def post_profiling():
    """
    Toggle profiling mode.

    Enables or disables runtime profiling for performance diagnostics.
    Returns the new profiling state.
    """
    shared.cmd_opts.profile = not shared.cmd_opts.profile
    log.info(f'API: profiling {"enabled" if shared.cmd_opts.profile else "disabled"}')
    return {"enabled": shared.cmd_opts.profile}


# ---------------------------------------------------------------------------
# Update check / apply
# ---------------------------------------------------------------------------

def get_update_check():
    """
    Check for available updates.

    Fetches the latest commit from the remote repository and compares it to
    the current local commit. Returns branch, current/latest hashes and dates,
    and whether the installation is up to date.
    """
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
        log.error(f'API update check: {e}')
        return {"error": str(e)}


def post_update_apply(rebase: bool = True, submodules: bool = True, extensions: bool = True):
    """
    Apply available updates.

    Pulls the latest code from the remote repository. Optionally updates
    submodules and extensions. Returns status messages, whether changes
    were applied, and the new version info.
    """
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
    """
    Run a performance benchmark.

    Generates images at the specified resolution across one or more batch sizes
    depending on ``level``: ``quick`` (batch 1), ``normal`` (1/2/4), or
    ``extensive`` (1/2/4/8/16). Returns iterations-per-second for each batch.
    """
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
    """
    Get saved benchmark results.

    Returns previously recorded benchmark data from the local results file,
    including timestamps, iterations-per-second, system info, and GPU details.
    """
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
# Storage
# ---------------------------------------------------------------------------

def _dir_size(path: str, recursive: bool = False, max_depth: int = 1):
    """Return total size in bytes of a directory. Non-recursive by default for speed on huge dirs."""
    total = 0
    if not path or not os.path.isdir(path):
        return total
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif recursive and entry.is_dir(follow_symlinks=False) and max_depth > 0:
                    total += _dir_size(entry.path, recursive=True, max_depth=max_depth - 1)
            except OSError:
                pass
    except OSError:
        pass
    return total


def _file_size(path: str):
    try:
        return os.path.getsize(path) if os.path.isfile(path) else 0
    except OSError:
        return 0


def get_storage():
    """Return disk usage grouped by category."""
    data_dir = paths.data_path or '.'
    result = {}

    # Caches
    cache_files = [
        ("cache.json", os.path.join(data_dir, "data", "cache.json")),
        ("metadata.json", os.path.join(data_dir, "data", "metadata.json")),
        ("extensions.json", os.path.join(data_dir, "data", "extensions.json")),
        ("jobs.db", os.path.join(data_dir, "jobs.db")),
    ]
    result["caches"] = [{"label": label, "path": p, "size": _file_size(p)} for label, p in cache_files]

    # Temp
    temp_dir = getattr(shared.opts, 'temp_dir', '') or os.path.join(data_dir, "tmp")
    result["temp"] = [{"label": "Temp", "path": temp_dir, "size": _dir_size(temp_dir, recursive=True, max_depth=2)}]

    # HuggingFace cache
    hf_dir = getattr(shared.opts, 'hfcache_dir', '') or os.path.join(paths.models_path, 'huggingface')
    result["huggingface"] = [{"label": "HuggingFace cache", "path": hf_dir, "size": _dir_size(hf_dir, recursive=False)}]

    # Outputs — resolve via the same base+specific logic the gallery uses
    base_samples = getattr(shared.opts, 'outdir_samples', '')
    base_grids = getattr(shared.opts, 'outdir_grids', '')
    output_candidates = [
        ("Text-to-Image", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_txt2img_samples', ''))),
        ("Image-to-Image", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_img2img_samples', ''))),
        ("Control", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_control_samples', ''))),
        ("Extras", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_extras_samples', ''))),
        ("Saved", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_save', ''))),
        ("Video", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_video', ''))),
        ("Init images", paths.resolve_output_path(base_samples, getattr(shared.opts, 'outdir_init_images', ''))),
        ("Grids (txt2img)", paths.resolve_output_path(base_grids, getattr(shared.opts, 'outdir_txt2img_grids', ''))),
        ("Grids (img2img)", paths.resolve_output_path(base_grids, getattr(shared.opts, 'outdir_img2img_grids', ''))),
        ("Grids (control)", paths.resolve_output_path(base_grids, getattr(shared.opts, 'outdir_control_grids', ''))),
    ]
    # Deduplicate — multiple grid types often resolve to the same folder
    seen_paths: dict[str, str] = {}
    outputs = []
    for label, p in output_candidates:
        if not p:
            continue
        resolved = os.path.normpath(os.path.abspath(p))
        if resolved in seen_paths:
            # Merge label into the first occurrence
            seen_paths[resolved] += f" / {label}"
            for entry in outputs:
                if os.path.normpath(os.path.abspath(entry["path"])) == resolved:
                    entry["label"] = seen_paths[resolved]
                    break
        else:
            seen_paths[resolved] = label
            outputs.append({"label": label, "path": p, "size": _dir_size(p, recursive=True, max_depth=3)})
    result["outputs"] = outputs

    # Models
    model_dirs = [
        ("Checkpoints", getattr(shared.opts, 'ckpt_dir', os.path.join(paths.models_path, 'Stable-diffusion'))),
        ("Diffusers", getattr(shared.opts, 'diffusers_dir', os.path.join(paths.models_path, 'Diffusers'))),
        ("VAE", getattr(shared.opts, 'vae_dir', os.path.join(paths.models_path, 'VAE'))),
        ("UNET", getattr(shared.opts, 'unet_dir', os.path.join(paths.models_path, 'UNET'))),
        ("Text encoders", getattr(shared.opts, 'te_dir', os.path.join(paths.models_path, 'Text-encoder'))),
        ("LoRA", getattr(shared.cmd_opts, 'lora_dir', os.path.join(paths.models_path, 'Lora'))),
        ("Embeddings", getattr(shared.cmd_opts, 'embeddings_dir', os.path.join(paths.models_path, 'embeddings'))),
        ("Control", getattr(shared.opts, 'control_dir', os.path.join(paths.models_path, 'control'))),
    ]
    result["models"] = [{"label": label, "path": p, "size": _dir_size(p, recursive=False)} for label, p in model_dirs if p]

    log.debug('API: storage scan complete')
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_api():
    api = shared.api
    api.add_api_route("/sdapi/v2/server/restart", post_restart, methods=["POST"], tags=["System"])
    api.add_api_route("/sdapi/v2/server/shutdown", post_shutdown, methods=["POST"], tags=["System"])
    api.add_api_route("/sdapi/v2/server/profiling", post_profiling, methods=["POST"], tags=["System"])
    api.add_api_route("/sdapi/v2/update/check", get_update_check, methods=["GET"], tags=["System"])
    api.add_api_route("/sdapi/v2/update/apply", post_update_apply, methods=["POST"], tags=["System"])
    api.add_api_route("/sdapi/v2/benchmark/run", post_benchmark_run, methods=["POST"], tags=["System"])
    api.add_api_route("/sdapi/v2/benchmark/results", get_benchmark_results, methods=["GET"], tags=["System"])
    api.add_api_route("/sdapi/v2/storage", get_storage, methods=["GET"], tags=["System"])
