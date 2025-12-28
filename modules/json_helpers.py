import os
import sys
import time
import json
from typing import overload, Literal
import fasteners
import orjson
from installer import log


locking_available = True # used by file read/write locking


@overload
def readfile(filename: str, silent: bool = False, lock: bool = False, *, as_type: Literal["dict"]) -> dict: ...
@overload
def readfile(filename: str, silent: bool = False, lock: bool = False, *, as_type: Literal["list"]) -> list: ...
@overload
def readfile(filename: str, silent: bool = False, lock: bool = False) -> dict | list: ...
def readfile(filename: str, silent: bool = False, lock: bool = False, *, as_type="") -> dict | list:
    global locking_available # pylint: disable=global-statement
    data = {} if as_type == "dict" else []
    lock_file = None
    locked = False
    if lock and locking_available:
        try:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock")
            lock_file.logger.disabled = True
            locked = lock_file.acquire_read_lock(blocking=True, timeout=3)
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
            data = orjson.loads(b) # pylint: disable=no-member
        # if type(data) is str:
        #    data = json.loads(data)
        t1 = time.time()
        if not silent:
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            log.debug(f'Read: file="{filename}" json={len(data)} bytes={os.path.getsize(filename)} time={t1-t0:.3f} fn={fn}')
    except FileNotFoundError as err:
        if not silent:
            log.debug(f'Reading failed: {filename} {err}')
    except Exception as err:
        if not silent:
            log.error(f'Reading failed: {filename} {err}')
    try:
        if locking_available and lock_file is not None:
            lock_file.release_read_lock()
        if locked and os.path.exists(f"{filename}.lock"):
            os.remove(f"{filename}.lock")
    except Exception:
        locking_available = False
    if isinstance(data, list) and as_type == "dict":
        log.warning(f"Read: Expected dictionary from '{filename}' but got list")
        data0 = data[0]
        if isinstance(data0, dict):
            return data0
        return {}
    if isinstance(data, dict) and as_type == "list":
        log.warning(f"Read: Expected list from '{filename}' but got dictionary")
        return [data]
    return data


def writefile(data, filename, mode='w', silent=False, atomic=False):
    import tempfile
    global locking_available # pylint: disable=global-statement
    lock_file = None
    locked = False

    def default(obj):
        log.error(f'Save: file="{filename}" not a valid object: {obj}')
        return str(obj)

    try:
        t0 = time.time()
        # skipkeys=True, ensure_ascii=True, check_circular=True, allow_nan=True
        if type(data) == dict:
            output = json.dumps(data, indent=2, default=default)
        elif type(data) == list:
            output = json.dumps(data, indent=2, default=default)
        elif isinstance(data, object):
            simple = {}
            for k in data.__dict__:
                if data.__dict__[k] is not None:
                    simple[k] = data.__dict__[k]
            output = json.dumps(simple, indent=2, default=default)
        else:
            raise ValueError('not a valid object')
    except Exception as err:
        log.error(f'Save failed: file="{filename}" {err}')
        return
    try:
        if locking_available:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock") if locking_available else None
            lock_file.logger.disabled = True
            locked = lock_file.acquire_write_lock(blocking=True, timeout=3) if lock_file is not None else False
    except Exception as err:
        locking_available = False
        lock_file = None
        log.error(f'File write lock: file="{filename}" {err}')
        locked = False
    try:
        if atomic:
            with tempfile.NamedTemporaryFile(mode=mode, encoding="utf8", delete=False, dir=os.path.dirname(filename)) as f:
                f.write(output)
                f.flush()
                os.fsync(f.fileno())
                os.replace(f.name, filename)
        else:
            with open(filename, mode=mode, encoding="utf8") as file:
                file.write(output)
        t1 = time.time()
        if not silent:
            log.debug(f'Save: file="{filename}" json={len(data)} bytes={len(output)} time={t1-t0:.3f}')
    except Exception as err:
        log.error(f'Save failed: file="{filename}" {err}')
    try:
        if locking_available and lock_file is not None:
            lock_file.release_write_lock()
        if locked and os.path.exists(f"{filename}.lock"):
            os.remove(f"{filename}.lock")
    except Exception:
        locking_available = False
