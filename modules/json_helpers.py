import os
import sys
import contextlib
import time
import json
from typing import overload, Literal
import fasteners
import orjson
from modules.logger import log

locking_available = True  # used by file read/write locking


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
    global locking_available  # pylint: disable=global-statement
    data = {} if as_type == "dict" else []
    lock_file = None
    locked = False

    if lock and locking_available:
        try:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock")
            lock_file.logger.disabled = True  # type: ignore - False positive. Bad typing in Fasteners.
            locked = lock_file.acquire_read_lock(blocking=True, timeout=3)
            if not locked:
                log.warning(f'File read lock: file="{filename}" timeout')
        except Exception as err:
            lock_file = None
            locking_available = False
            log.error(f'File read lock: file="{filename}" {err}')
            locked = False

    try:
        # if not os.path.exists(filename):
        #    return {}
        t0 = time.time()
        with open(filename, "rb") as file:
            b = file.read()
            if len(b) == 0:
                return {} if as_type == "dict" else []
            data = orjson.loads(b)  # pylint: disable=no-member
        # if type(data) is str:
        #    data = json.loads(data)
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

    try:
        if locking_available and lock_file is not None:
            lock_file.release_read_lock()  # the lock file stays: removing it after release races other holders
    except Exception as err:
        log.error(f'File read lock release: file="{filename}" {err}')

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


def writefile(obj: dict | list, filename: str | os.PathLike[str], mode="w", silent=False, atomic=False):
    import copy
    import tempfile

    global locking_available # pylint: disable=global-statement
    lock_file = None
    locked = False

    def default(obj):
        log.error(f'Save: file="{filename}" not a valid object: {obj}')
        return str(obj)

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
        if locking_available:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock") if locking_available else None
            lock_file.logger.disabled = True  # type: ignore - False positive. Bad typing in Fasteners.
            locked = lock_file.acquire_write_lock(blocking=True, timeout=3) if lock_file is not None else False
            if not locked:
                log.warning(f'File write lock: file="{filename}" timeout')
    except Exception as err:
        locking_available = False
        lock_file = None
        log.error(f'File write lock: file="{filename}" {err}')
        locked = False

    try:
        if atomic:
            fd, temp_name = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(filename)), prefix=f"{os.path.basename(filename)}.", suffix=".tmp")
            try:
                with os.fdopen(fd, mode, encoding="utf8") as f:
                    f.write(output)
                    f.flush()
                    os.fsync(f.fileno())
                replace_file(temp_name, filename)
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

    try:
        if locking_available and lock_file is not None:
            lock_file.release_write_lock()  # the lock file stays: removing it after release races other holders
    except Exception as err:
        log.error(f'File write lock release: file="{filename}" {err}')
