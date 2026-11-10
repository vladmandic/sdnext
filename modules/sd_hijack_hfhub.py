import os
import time
import tqdm
from modules.logger import log


debug = log.trace if os.environ.get('SD_DOWNLOAD_DEBUG', None) is not None else lambda *args, **kwargs: None
orig_http_get = None
orig_xet_get = None
orig_build_hf_headers = None


def prune_hf_cache():
    from modules import shared
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.utils import _shared_blobs
    try:
        cache_info = scan_cache_dir(cache_dir=shared.opts.hfcache_dir)
        revisions = sorted({
            revision.commit_hash
            for repo in cache_info.repos
            for revision in repo.revisions
            if not revision.refs
        })
        incomplete_files = cache_info.incomplete_files
        unreferenced_blobs = _shared_blobs.unreferenced_shared_blobs(cache_info.cache_dir) if cache_info.cache_dir is not None else {}
        if not revisions and not incomplete_files and not unreferenced_blobs:
            return 'nothing to prune'

        strategy = cache_info.delete_revisions(*revisions)
        total_freed = strategy.expected_freed_size + cache_info.incomplete_size_on_disk + sum(unreferenced_blobs.values())
        strategy.execute()
        for incomplete_file in incomplete_files:
            try:
                incomplete_file.file_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                log.warning(f'Hugging Face cache prune: incomplete file="{incomplete_file.file_path}" error: {e}')
        if cache_info.cache_dir is not None:
            for store_path in unreferenced_blobs:
                _shared_blobs.sweep_shared_blob(store_path, cache_dir=cache_info.cache_dir)
        return f'revisions={len(revisions)} incomplete={len(incomplete_files)} unreferenced={len(unreferenced_blobs)} freed={total_freed}'
    except Exception as e:
        log.error(f'HF cache prune: {e}')
        return f'failed: {e}'


def clean_user_agent(headers):
    ua = headers.get('user-agent', None)
    if ua is None:
        return headers
    ua = ua.replace('unknown/None;', 'sdnext/0.0.0;')
    if 'quant_config' in ua:
        ua = ua.split('quant_config')[0]
    ua = ua.rstrip(' ;')
    # ua += '; telemetry/off'
    headers['user-agent'] = ua
    return headers


def http_get_hijack(*args, **kwargs):
    from modules.shared import state, opts
    if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".json"):
        return orig_http_get(*args, **kwargs)
    jobid = state.begin('Download')
    fn = kwargs.get("displayed_filename", None)
    size = kwargs.get("expected_size", None)
    if 'headers' in kwargs:
        kwargs['headers'] = clean_user_agent(kwargs['headers'])
    if fn and not fn.endswith(".json") and size is not None and size > 10240:
        log.debug(f'Download: type=http mode="{opts.hf_transfer_mode}" fn="{fn}" size={size}')
    debug(f'Download start: type=http args={args} kwargs={kwargs}')
    t0 = time.time()
    res = orig_http_get(*args, **kwargs)
    t1 = time.time()
    debug(f'Download end: type=http res={res} time={t1-t0:.2f} perf={size/(t1-t0)/1024/1024:.2f} MB/s')
    state.end(jobid)
    return res


def xet_get_hijack(*args, **kwargs):
    from modules.shared import state, opts
    if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".json"):
        return orig_xet_get(*args, **kwargs)
    jobid = state.begin('Download')
    fn = kwargs.get("displayed_filename", None)
    size = kwargs.get("expected_size", None)
    if 'headers' in kwargs:
        kwargs['headers'] = clean_user_agent(kwargs['headers'])
    if fn and not fn.endswith(".json"):
        log.debug(f'Download: type=xet mode="{opts.hf_transfer_mode}" fn="{fn}" size={size}')
    debug(f'Download start: type=xet args={args} kwargs={kwargs}')

    bar_format = 'Download {percentage:3.0f}% {bar:15} {n:.1f}/{total:.1f}{postfix} {elapsed} {remaining} ' + '\x1b[38;5;71m' + '{desc}' + '\x1b[0m'
    kwargs['_tqdm_bar'] = tqdm.tqdm(*args, bar_format=bar_format, colour='#327fba', unit='MiB', unit_scale=1/(1024*1024), desc=fn, total=size, ncols=120)

    res = orig_xet_get(*args, **kwargs)
    debug(f'Download end: type=xet res={res}')
    state.end(jobid)
    return res


def resolve_token():
    from huggingface_hub import get_token
    return os.environ.get("HF_TOKEN") or get_token() # HF_HUB_DISABLE_IMPLICIT_TOKEN is set, so hub calls send only what sdnext passes


def build_hf_headers_hijack(*args, **kwargs):
    token = kwargs.get('token', None)
    if token is None or token is True:
        token = resolve_token()
        kwargs['token'] = token
    headers = orig_build_hf_headers(*args, **kwargs)
    headers = clean_user_agent(headers)
    return headers


def init_hijack():
    from huggingface_hub import file_download
    global orig_http_get, orig_xet_get, orig_build_hf_headers # pylint: disable=global-statement
    if orig_http_get is None or orig_xet_get is None:
        orig_http_get = file_download.http_get
        orig_xet_get = file_download.xet_get
        orig_build_hf_headers = file_download.build_hf_headers # pylint: disable=protected-access
        file_download.http_get = http_get_hijack
        file_download.xet_get = xet_get_hijack
        file_download.build_hf_headers = build_hf_headers_hijack # pylint: disable=protected-access
