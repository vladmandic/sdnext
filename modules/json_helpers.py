import os
import sys
import contextlib
import threading
import time
import json
from typing import overload, Literal
import orjson
from modules.logger import log

path_locks: dict[str, threading.RLock] = {}
path_locks_guard = threading.Lock()


def path_lock(filename: str | os.PathLike[str]) -> threading.RLock:
    """One lock per file path; threads of this process serialize on it, other processes are covered by atomic replace."""
    key = os.path.normcase(os.path.realpath(filename))
    with path_locks_guard:
        lock = path_locks.get(key)
        if lock is None:
            lock = path_locks[key] = threading.RLock()
            with contextlib.suppress(OSError):
                os.remove(f"{key}.lock")  # left behind by the file lock this replaces
        return lock


def read_bytes(filename: str | os.PathLike[str], attempts: int = 5, delay: float = 0.01) -> bytes:
    """Read a whole file, waiting out an open that Windows refuses while a replace of the same name is in flight."""
    for attempt in range(attempts):
        try:
            with open(filename, "rb") as file:
                return file.read()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    return b""


def replace_file(source: str, target: str, attempts: int = 10, delay: float = 0.05):
    """os.replace that waits out a target another handle holds open, which Windows reports as a permission error."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


@overload
def readfile(filename: str | os.PathLike[str], silent: bool = False, lock: bool = False, *, as_type: Literal["dict"]) -> dict: ...
@overload
def readfile(filename: str | os.PathLike[str], silent: bool = False, lock: bool = False, *, as_type: Literal["list"]) -> list: ...
@overload
def readfile(filename: str | os.PathLike[str], silent: bool = False, lock: bool = False) -> dict | list: ...
def readfile(filename: str | os.PathLike[str], silent: bool = False, lock: bool = False, *, as_type="") -> dict | list:
    """Read a JSON file; lock=True serializes with writers of the same path in this process."""
    data = {} if as_type == "dict" else []

    with path_lock(filename) if lock else contextlib.nullcontext():
        try:
            t0 = time.time()
            b = read_bytes(filename)
            if len(b) == 0:
                if not silent:
                    log.warning(f'Read: file="{filename}" empty')
                return {} if as_type == "dict" else []
            data = orjson.loads(b)  # pylint: disable=no-member
            t1 = time.time()
            if not silent:
                fn = f"{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}"  # pylint: disable=protected-access
                log.debug(f'Read: file="{filename}" json={len(data)} bytes={os.path.getsize(filename)} time={t1 - t0:.3f} fn={fn}')
        except FileNotFoundError as err:
            if not silent:
                log.debug(f'Read failed: file="{filename}" {err}')
        except Exception as err:
            if not silent:
                log.error(f'Read failed: file="{filename}" {err}')

    if isinstance(data, list) and as_type == "dict":
        if not data:
            return {}
        log.warning(f"Read: Expected dictionary from '{filename}' but got list")
        data0 = data[0]
        if isinstance(data0, dict):
            return data0
        return {}
    if isinstance(data, dict) and as_type == "list":
        if not data:
            return []
        log.warning(f"Read: Expected list from '{filename}' but got dictionary")
        return [data]
    return data


def writefile(obj: dict | list, filename: str | os.PathLike[str], mode="w", silent=False, atomic=True):
    """Write obj as JSON through a temp file and replace; writes to the same path from this process run one at a time, in call order."""
    import copy
    import tempfile

    def default(obj):
        log.error(f'Save: file="{filename}" not a valid object: {obj}')
        return str(obj)

    if mode != "w":
        atomic = False  # append cannot go through a temp file

    with path_lock(filename):
        try:
            t0 = time.time()
            snapshot = obj.copy() if isinstance(obj, (dict, list)) else obj # dict.copy and list.copy run under the GIL, so a concurrent insert cannot interrupt them
            data = copy.deepcopy(snapshot)
            for k, v in list(data.items()) if isinstance(data, dict) else []: # validate each key-by-key to avoid global exceptions
                try:
                    _tmp = json.dumps(v, indent=2, default=default, allow_nan=False, ensure_ascii=False)
                except Exception as err:
                    if not silent:
                        log.error(f'Save: file="{filename}" key="{k}" value="{v}" {err}')
                    del data[k]
            output = json.dumps(data, indent=2, default=default)
        except Exception as err:
            log.error(f'Save failed: file="{filename}" {err}')
            return

        try:
            if atomic:
                target = os.path.realpath(filename)  # replace the file a symlink points at, not the symlink
                fd, temp_name = tempfile.mkstemp(dir=os.path.dirname(target), prefix=f"{os.path.basename(target)}.", suffix=".tmp")
                try:
                    with os.fdopen(fd, mode, encoding="utf8") as f:
                        f.write(output)
                        f.flush()
                        os.fsync(f.fileno())
                    replace_file(temp_name, target)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.remove(temp_name)
                    raise
            else:
                with open(filename, mode=mode, encoding="utf8") as file:
                    file.write(output)
            t1 = time.time()
            if not silent:
                datalength = len(data)
                log.debug(f'Save: file="{filename}" json={datalength} bytes={len(output)} time={t1 - t0:.3f}')
        except Exception as err:
            log.error(f'Save failed: file="{filename}" {err}')
