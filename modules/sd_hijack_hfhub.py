import os
from modules.logger import log


debug = log.trace if os.environ.get('SD_DOWNLOAD_DEBUG', None) is not None else lambda *args, **kwargs: None
orig_http_get = None
orig_xet_get = None


def http_get_hijack(*args, **kwargs):
    from modules.shared import state
    if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".json"):
        return orig_http_get(*args, **kwargs)
    jobid = state.begin('Download')
    fn = kwargs.get("displayed_filename", None)
    size = kwargs.get("expected_size", None)
    if fn and not fn.endswith(".json"):
        log.debug(f'Download start: type=http fn="{fn}" size={size}')
    debug(f'Download start: type=http args={args} kwargs={kwargs}')
    res = orig_http_get(*args, **kwargs)
    debug(f'Download end: type=http res={res}')
    state.end(jobid)
    return res


def xet_get_hijack(*args, **kwargs):
    from modules.shared import state
    if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".json"):
        return orig_xet_get(*args, **kwargs)
    jobid = state.begin('Download')
    fn = kwargs.get("displayed_filename", None)
    size = kwargs.get("expected_size", None)
    if fn and not fn.endswith(".json"):
        log.debug(f'Download start: type=xet fn="{fn}" size={size}')
    debug(f'Download start: type=xet args={args} kwargs={kwargs}')
    res = orig_xet_get(*args, **kwargs)
    debug(f'Download end: type=xet res={res}')
    state.end(jobid)
    return res


def init_hijack():
    from huggingface_hub import file_download
    global orig_http_get, orig_xet_get # pylint: disable=global-statement
    if orig_http_get is None or orig_xet_get is None:
        orig_http_get = file_download.http_get
        orig_xet_get = file_download.xet_get
        file_download.http_get = http_get_hijack
        file_download.xet_get = xet_get_hijack
