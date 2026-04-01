import os
from types import SimpleNamespace
from modules import paths
from modules.logger import log


dir_timestamps = {}
dir_cache = {}


def listdir(path):
    if not os.path.exists(path):
        return []
    mtime = os.path.getmtime(path)
    if path in dir_timestamps and mtime == dir_timestamps[path]:
        return dir_cache[path]
    else:
        dir_cache[path] = [os.path.join(path, f) for f in os.listdir(path)]
        dir_timestamps[path] = mtime
        return dir_cache[path]


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return
    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)
    for root, _dirs, files in os.walk(path, followlinks=True):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue
            yield os.path.join(root, filename)


def html_path(filename):
    return os.path.join(paths.script_path, "html", filename)


def html(filename):
    path = html_path(filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()
    return ""


def req(url_addr, headers = None, **kwargs):
    import requests
    if headers is None:
        headers = { 'Content-type': 'application/json' }
    try:
        res = requests.get(url_addr, timeout=30, headers=headers, verify=False, allow_redirects=True, **kwargs)
    except Exception as err:
        log.error(f'HTTP request error: url={url_addr} {err}')
        res = { 'status_code': 500, 'text': f'HTTP request error: url={url_addr} {err}' }
        res = SimpleNamespace(**res)
    return res


class TotalTQDM: # compatibility with previous global-tqdm
    # import tqdm
    def __init__(self):
        pass
    def reset(self):
        pass
    def update(self):
        pass
    def updateTotal(self, new_total):
        pass
    def clear(self):
        pass
total_tqdm = TotalTQDM()
