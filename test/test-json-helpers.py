#!/usr/bin/env python
"""
Offline tests for modules.json_helpers on Linux and Windows.

Covers:

- atomic saves replace a fresh or existing target and leave no temp file, including when the save fails
- an atomic save waits out a target another thread briefly holds open, which Windows otherwise refuses
- overlapping locked reads and atomic writes never fail or read a torn file, and no lock file is created
- locked readers never see a torn file while unlocked writers rewrite it in place
- a lock file left behind by the former file lock is removed
- concurrent inserts into a shared dict never drop a save
- default writes are atomic, so unlocked readers never see a torn file either
- an atomic write through a symlink replaces the file it points at and keeps the link
- a read waits out an open that is refused while a replace is in flight
- an empty file reads as empty and is reported unless the read is silent
- readfile returns what writefile wrote, as dict and as list
- the hash cache saves cleanly while other threads keep adding hashes

No running server required.

Usage:
    python test/test-json-helpers.py
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)
sys.argv = [sys.argv[0]]

from modules.logger import log  # pylint: disable=wrong-import-position
from modules import json_helpers  # pylint: disable=wrong-import-position


class ErrorLog:
    """Counts error and warning lines from json_helpers while passing everything else to the real logger."""

    def __init__(self, inner):
        self.inner = inner
        self.errors = []
        self.warnings = []
        self.lock = threading.Lock()

    def error(self, message, *args, **kwargs):
        with self.lock:
            self.errors.append(str(message))

    def warning(self, message, *args, **kwargs):
        with self.lock:
            self.warnings.append(str(message))

    def count(self, needle):
        return sum(1 for m in self.errors if needle in m)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def fresh_helper():
    importlib.reload(json_helpers)
    json_helpers.log = ErrorLog(log)
    return json_helpers


def run_threads(worker, count):
    pool = [threading.Thread(target=worker, args=(n,)) for n in range(count)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()


def leftovers(folder, target):
    keep = {os.path.basename(target), os.path.basename(target) + '.lock'}
    return sorted(set(os.listdir(folder)) - keep)


def test_atomic_save_replaces_target(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'atomic.json')
    jh.writefile({'first': 1}, target, silent=True, atomic=True)
    assert jh.readfile(target, silent=True, as_type='dict') == {'first': 1}, 'fresh target not written'
    jh.writefile({'second': 2}, target, silent=True, atomic=True)
    assert jh.readfile(target, silent=True, as_type='dict') == {'second': 2}, 'existing target not replaced'
    assert leftovers(folder, target) == [], f'temp files left: {leftovers(folder, target)}'
    assert jh.log.errors == [], jh.log.errors


def test_failed_atomic_save_leaves_no_temp(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'blocked.json')
    os.mkdir(target)  # a directory cannot be replaced by a file
    jh.writefile({'x': 1}, target, silent=True, atomic=True)
    assert jh.log.count('Save failed') == 1, jh.log.errors
    assert os.path.isdir(target), 'target directory was replaced'
    assert leftovers(folder, target) == [], f'temp files left: {leftovers(folder, target)}'


def test_atomic_save_waits_for_open_target(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'open.json')
    jh.writefile({'v': 0}, target, silent=True)
    opened = threading.Event()
    release = threading.Event()

    def holder(_n):
        with open(target, 'rb') as f:
            f.read(1)
            opened.set()
            release.wait(5)

    thread = threading.Thread(target=holder, args=(0,))
    thread.start()
    opened.wait(5)
    threading.Timer(0.2, release.set).start()
    jh.writefile({'v': 1}, target, silent=True, atomic=True)
    thread.join()
    assert jh.readfile(target, silent=True, as_type='dict') == {'v': 1}, 'save did not wait for the open target'
    assert jh.log.errors == [], jh.log.errors
    assert leftovers(folder, target) == [], f'temp files left: {leftovers(folder, target)}'


def test_overlapping_locked_access(folder, threads=8, iterations=25):
    jh = fresh_helper()
    target = os.path.join(folder, 'locked.json')
    jh.writefile({'seed': 0}, target, silent=True)
    torn = []

    def worker(n):
        for i in range(iterations):
            if (n + i) % 2:
                jh.writefile({'n': n, 'i': i}, target, silent=True, atomic=True)
            elif not jh.readfile(target, silent=True, lock=True, as_type='dict'):
                torn.append((n, i))

    run_threads(worker, threads)
    assert jh.log.errors == [], jh.log.errors
    assert torn == [], f'{len(torn)} reads returned nothing'
    assert not os.path.exists(target + '.lock'), 'a lock file was created'
    assert leftovers(folder, target) == [], f'temp files left: {leftovers(folder, target)}'


def test_locked_readers_never_see_torn_writes(folder, writers=8, per_writer=15, readers=2):
    jh = fresh_helper()
    target = os.path.join(folder, 'config.json')
    jh.writefile({'seed': 0}, target, silent=True)
    stop = threading.Event()
    torn = []

    def reader(_n):
        while not stop.is_set():
            if not jh.readfile(target, silent=True, lock=True, as_type='dict'):
                torn.append(1)

    def writer(n):
        for i in range(per_writer):
            jh.writefile({'writer': n, 'i': i, 'blob': 'x' * (40000 + 1000 * n)}, target, silent=True, atomic=False)

    pool = [threading.Thread(target=reader, args=(r,)) for r in range(readers)]
    for t in pool:
        t.start()
    run_threads(writer, writers)
    stop.set()
    for t in pool:
        t.join()
    assert torn == [], f'{len(torn)} reads returned nothing'
    with open(target, encoding='utf8') as f:
        json.load(f)
    assert jh.log.errors == [], jh.log.errors


def test_legacy_lock_file_removed(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'legacy.json')
    with open(target + '.lock', 'w', encoding='utf8'):
        pass
    jh.writefile({'x': 1}, target, silent=True)
    assert not os.path.exists(target + '.lock'), 'legacy lock file kept'
    assert jh.log.errors == [], jh.log.errors


def test_concurrent_inserts_keep_every_save(folder, threads=4, per_thread=40):
    jh = fresh_helper()
    target = os.path.join(folder, 'cache.json')
    cache = {}
    payload = {'metadata': {f'k{i}': 'x' * 64 for i in range(200)}, 'tensors': list(range(500))}

    def worker(n):
        for i in range(per_thread):
            cache[f'{n}-{i}'] = payload
            jh.writefile(cache, target, silent=True, atomic=True)

    run_threads(worker, threads)
    assert jh.log.count('changed size') == 0, f'{jh.log.count("changed size")} saves dropped'
    assert jh.log.errors == [], jh.log.errors
    with open(target, encoding='utf8') as f:
        assert len(json.load(f)) == threads * per_thread, 'last save is incomplete'


def test_default_writes_never_tear(folder, writers=4, per_writer=20, readers=2):
    jh = fresh_helper()
    target = os.path.join(folder, 'default.json')
    jh.writefile({'seed': 0}, target, silent=True)
    stop = threading.Event()
    torn = []

    def reader(_n):
        while not stop.is_set():
            if not jh.readfile(target, silent=True, as_type='dict'):
                torn.append(1)
            time.sleep(0.001)  # real readers do not spin; a spinning reader starves the writer's retry on Windows

    def writer(n):
        for i in range(per_writer):
            jh.writefile({'writer': n, 'i': i, 'blob': 'x' * (40000 + 1000 * n)}, target, silent=True)

    pool = [threading.Thread(target=reader, args=(r,)) for r in range(readers)]
    for t in pool:
        t.start()
    run_threads(writer, writers)
    stop.set()
    for t in pool:
        t.join()
    assert torn == [], f'{len(torn)} unlocked reads returned nothing'
    assert jh.log.errors == [], jh.log.errors
    assert leftovers(folder, target) == [], f'temp files left: {leftovers(folder, target)}'


def test_atomic_write_keeps_symlink(folder):
    jh = fresh_helper()
    real = os.path.join(folder, 'real.json')
    link = os.path.join(folder, 'link.json')
    jh.writefile({'v': 0}, real, silent=True)
    try:
        os.symlink(real, link)
    except OSError as e:
        log.info(f'  SKIP symlink not available: {e}')
        return
    jh.writefile({'v': 1}, link, silent=True)
    assert os.path.islink(link), 'symlink was replaced by a file'
    assert jh.readfile(real, silent=True, as_type='dict') == {'v': 1}, 'target of the symlink not updated'
    assert sorted(os.listdir(folder)) == ['link.json', 'real.json'], os.listdir(folder)
    assert jh.log.errors == [], jh.log.errors


def test_read_waits_out_refused_open(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'refused.json')
    jh.writefile({'x': 1}, target, silent=True)
    calls = []
    real_open = open

    def refuse_twice(*args, **kwargs):
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError(13, 'replace in flight')
        return real_open(*args, **kwargs)

    jh.open = refuse_twice  # module globals shadow the builtin inside read_bytes
    try:
        assert jh.readfile(target, silent=True, as_type='dict') == {'x': 1}
    finally:
        del jh.open
    assert len(calls) == 3, f'{len(calls)} open attempts'
    assert jh.log.errors == [], jh.log.errors


def test_empty_file_is_reported(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'empty.json')
    with open(target, 'w', encoding='utf8'):
        pass
    assert jh.readfile(target, as_type='dict') == {}
    assert jh.readfile(target, silent=True, as_type='list') == []
    assert len(jh.log.warnings) == 1 and 'empty' in jh.log.warnings[0], jh.log.warnings
    assert jh.log.errors == [], jh.log.errors


def test_hash_cache_saves_under_concurrent_adds(folder, adders=4, per_adder=200):
    jh = fresh_helper()
    from modules import hashes  # pylint: disable=import-outside-toplevel
    saved_filename = hashes.cache_filename
    hashes.cache_filename = os.path.join(folder, 'cache.json')
    hashes.cache('hashes').clear()
    hashes.cache('hashes-addnet').clear()
    stop = threading.Event()

    def adder(n):
        for i in range(per_adder):
            hashes.cache('hashes').add_hash(f'checkpoint/{n}-{i}', 1.0, 'a' * 64)
            if i % 10 == 9:
                hashes.save_cache()

    def churn(_n):  # a second store whose size keeps changing while the saves run, bounded so the snapshots stay small
        i = 0
        while not stop.is_set():
            store = hashes.cache('hashes-addnet')
            if i % 500 == 499:
                store.clear()
            else:
                store.add_hash(f'lora/{i % 500}', 1.0, 'b' * 64)
            i += 1

    thread = threading.Thread(target=churn, args=(0,))
    thread.start()
    try:
        run_threads(adder, adders)
    finally:
        stop.set()
        thread.join()
        hashes.cache_filename = saved_filename
    assert jh.log.errors == [], jh.log.errors
    with open(os.path.join(folder, 'cache.json'), encoding='utf8') as f:
        on_disk = json.load(f)
    assert len(on_disk['hashes']) == adders * per_adder, f'{len(on_disk["hashes"])} of {adders * per_adder} hashes on disk'
    hashes.cache('hashes').clear()
    hashes.cache('hashes-addnet').clear()


def test_roundtrip(folder):
    jh = fresh_helper()
    target = os.path.join(folder, 'roundtrip.json')
    data = {'a': 1, 'b': [1, 2, 3], 'c': {'d': 'e'}, 'f': None}
    jh.writefile(data, target, silent=True)
    assert jh.readfile(target, silent=True, as_type='dict') == data
    jh.writefile([1, 'two', {'three': 3}], target, silent=True, atomic=True)
    assert jh.readfile(target, silent=True, as_type='list') == [1, 'two', {'three': 3}]
    assert jh.readfile(os.path.join(folder, 'missing.json'), silent=True, as_type='dict') == {}
    assert jh.log.errors == [], jh.log.errors


def run_all():
    tests = [
        test_atomic_save_replaces_target,
        test_failed_atomic_save_leaves_no_temp,
        test_atomic_save_waits_for_open_target,
        test_overlapping_locked_access,
        test_locked_readers_never_see_torn_writes,
        test_legacy_lock_file_removed,
        test_concurrent_inserts_keep_every_save,
        test_default_writes_never_tear,
        test_atomic_write_keeps_symlink,
        test_read_waits_out_refused_open,
        test_empty_file_is_reported,
        test_hash_cache_saves_under_concurrent_adds,
        test_roundtrip,
    ]
    passed = 0
    failed = 0
    for fn in tests:
        folder = tempfile.mkdtemp(prefix='sdnext-json-')
        try:
            fn(folder)
            log.info(f'  PASS {fn.__name__}')
            passed += 1
        except Exception as e:
            log.error(f'  FAIL {fn.__name__}: {type(e).__name__}: {e}')
            failed += 1
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    log.warning(f'Total: {passed} passed, {failed} failed')
    return failed == 0


if __name__ == '__main__':
    t0 = time.time()
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
