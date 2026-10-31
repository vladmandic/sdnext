#!/usr/bin/env python
"""
Offline tests for the image metadata log (``save_log_fn``, "Save metadata to JSON file").

Each saved image appends one JSON object per line to ``<name>.jsonl``; the file is never rewritten.

Covers:

- the first saves creating the log, ids counting lines from 0, every line parsing
- an append keeping the file's inode and writing a line, not the whole log
- a ``.json`` log of the same name left untouched, whatever it holds
- an existing log continued, a torn last line kept on a line of its own, a log changed by another process recounted
- the configured name resolving to ``.jsonl`` under the data folder
- a log that cannot be written leaving the save itself and the next save working

No running server required.

Usage:
    python test/test-save-log.py
"""

import os
import sys
import json
import time
import shutil
import logging
import tempfile
import threading
import contextlib

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from PIL import Image                                                # pylint: disable=wrong-import-position
from modules import shared, devices, paths                           # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position
from modules.json_helpers import writefile                           # pylint: disable=wrong-import-position

devices.torch_gc = lambda *args, **kwargs: None
results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-save-log-', suffix='-work')
images_dir = os.path.join(work_dir, 'images')
os.makedirs(images_dir)
paths.data_path = work_dir # relative log names resolve here
paths.params_path = os.path.join(work_dir, 'params.txt') # every save with metadata writes it
saves = 0


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, status: str, name: str, detail: str = ''):
    results[cat][{'PASS': 'passed', 'FAIL': 'failed', 'SKIP': 'skipped'}[status]] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if status == 'PASS':
        log.info(msg)
    elif status == 'SKIP':
        log.warning(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if isinstance(ok, str):
            record(cat, 'SKIP', name, ok)
        else:
            record(cat, 'FAIL' if ok is False else 'PASS', name)
    except AssertionError as e:
        record(cat, 'FAIL', name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, 'FAIL', name, f'exception: {type(e).__name__}: {e}')


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):  # pylint: disable=redefined-outer-name
        self.records.append(record)

    def messages(self, level: int = logging.DEBUG) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


@contextlib.contextmanager
def capture_log():
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


@contextlib.contextmanager
def log_name(value: str):
    previous = {k: shared.opts.data.get(k, None) for k in ('save_log_fn', 'save_to_dirs', 'image_metadata')}
    shared.opts.data['save_log_fn'] = value
    shared.opts.data['save_to_dirs'] = False
    shared.opts.data['image_metadata'] = True
    try:
        yield
    finally:
        for k, v in previous.items():
            shared.opts.data[k] = v


def save(timeout: float = 60) -> str:
    """One image save through the save thread, failing instead of hanging when the thread no longer answers."""
    global saves # pylint: disable=global-statement
    from modules.image.save import save_image
    saves += 1
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault('result', save_image(Image.new('RGB', (8, 8)), path=images_dir, forced_filename=f'image-{saves}', extension='png', info=f'probe parameters {saves}')), daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), f'save {saves} did not return within {timeout}s'
    return out['result'][0]


def lines(fn: str) -> list[str]:
    with open(fn, encoding='utf8') as f:
        return f.read().splitlines()


def write_bytes(fn: str, data: bytes):
    with open(fn, 'wb') as f:
        f.write(data)


def read_bytes(fn: str) -> bytes:
    with open(fn, 'rb') as f:
        return f.read()


def jsonl(count: int) -> bytes:
    return b''.join(json.dumps({'id': i, 'filename': f'old-{i}', 'time': '2026-10-31T00:00:00', 'info': f'old {i}'}).encode() + b'\n' for i in range(count))


# ============================================================
# append
# ============================================================

def test_first_saves_create_the_log():
    target = os.path.join(work_dir, 'first.jsonl')
    with log_name(os.path.join(work_dir, 'first.json')):
        names = [save(), save()]
    assert os.path.isfile(target), f'no log at {target}, folder holds {sorted(os.listdir(work_dir))}'
    assert not os.path.exists(os.path.join(work_dir, 'first.json')), 'a .json log was written'
    entries = [json.loads(line) for line in lines(target)]
    assert [e['id'] for e in entries] == [0, 1], f'ids {[e["id"] for e in entries]}'
    assert [e['filename'] + '.png' for e in entries] == names, f'filenames {[e["filename"] for e in entries]} for saves {names}'
    assert all(e['info'].startswith('probe parameters') for e in entries), f'info {[e["info"] for e in entries]}'


def test_append_keeps_the_file():
    target = os.path.join(work_dir, 'inode.jsonl')
    with log_name(target):
        save()
        for _ in range(2):
            before = os.stat(target).st_ino
            save()
            after = os.stat(target).st_ino
            assert after == before, f'log replaced: inode {before} -> {after}'
    assert len(lines(target)) == 3, f'{len(lines(target))} lines'


def test_one_save_writes_one_line():
    """A save against a 1000-entry log writes a line, not the log; /proc/self/io counts every byte this process writes."""
    if not os.path.isfile('/proc/self/io'):
        return 'no /proc/self/io'
    def written() -> int:
        with open('/proc/self/io', encoding='ascii') as f:
            return int(next(line for line in f if line.startswith('wchar:')).split()[1])
    old = [{'id': i, 'filename': f'old-{i}', 'time': '2026-10-31T00:00:00', 'info': 'x' * 1000} for i in range(1000)]
    writefile(old, os.path.join(work_dir, 'volume.json'), silent=True) # the list as the whole-file log wrote it
    write_bytes(os.path.join(work_dir, 'volume.jsonl'), jsonl(1000))
    with log_name(os.path.join(work_dir, 'volume.json')):
        save() # first use counts the lines
        start = written()
        save()
        delta = written() - start
    assert delta < 64 * 1024, f'one save wrote {delta} bytes'


# ============================================================
# existing files
# ============================================================

def test_json_log_left_untouched():
    fixtures = {
        'dict': b'{"id": 0, "info": "not a list"}',
        'torn': b'[\n  {\n    "id": 0,\n    "filename": "a",\n    "info": "cut he',
        'list': json.dumps([{'id': 0, 'filename': 'a', 'time': 't', 'info': 'i'}], indent=2).encode(),
    }
    changed = {}
    for name, data in fixtures.items():
        fn = os.path.join(work_dir, f'keep-{name}.json')
        write_bytes(fn, data)
        with log_name(fn):
            save()
        if read_bytes(fn) != data:
            changed[name] = read_bytes(fn)[:80]
    assert not changed, f'.json logs changed: {changed}'
    missing = [name for name in fixtures if not os.path.isfile(os.path.join(work_dir, f'keep-{name}.jsonl'))]
    assert not missing, f'no .jsonl beside {missing}'


def test_existing_log_continues():
    target = os.path.join(work_dir, 'continue.jsonl')
    write_bytes(target, jsonl(50))
    with log_name(target):
        save()
    entry = json.loads(lines(target)[-1])
    assert entry['id'] == 50, f"id {entry['id']} after 50 lines"
    assert read_bytes(target).startswith(jsonl(50)), 'existing lines changed'


def test_torn_last_line_kept_apart():
    target = os.path.join(work_dir, 'torn.jsonl')
    write_bytes(target, jsonl(3) + b'{"id": 3, "filename": "cut')
    with log_name(target):
        save()
    got = lines(target)
    assert len(got) == 5, f'{len(got)} lines: {got[-2:]}'
    assert got[3] == '{"id": 3, "filename": "cut', f'torn line changed: {got[3]!r}'
    entry = json.loads(got[4])
    assert entry['id'] == 4, f"id {entry['id']} on line 4"


def test_log_changed_elsewhere_recounted():
    target = os.path.join(work_dir, 'drift.jsonl')
    with log_name(target):
        save()
        save()
        write_bytes(target, jsonl(1)) # truncated by another process
        save()
        with open(target, 'ab') as f: # appended by another process
            f.write(jsonl(1))
        save()
    ids = [json.loads(line)['id'] for line in lines(target)]
    assert ids == [0, 1, 0, 3], f'ids {ids}'


def test_name_resolves_to_jsonl():
    os.makedirs(os.path.join(work_dir, 'logs'), exist_ok=True)
    expected = {
        os.path.join('logs', 'plain'): 'plain.jsonl',
        os.path.join('logs', 'named.json'): 'named.jsonl',
        os.path.join('logs', 'lines.jsonl'): 'lines.jsonl',
        os.path.join('logs', 'upper.JSON'): 'upper.jsonl',
        os.path.join('logs', 'other.txt'): 'other.txt.jsonl',
    }
    for value in expected:
        with log_name(value):
            save()
    found = sorted(os.listdir(os.path.join(work_dir, 'logs')))
    assert found == sorted(expected.values()), f'logs {found}'


# ============================================================
# failures
# ============================================================

def test_unwritable_log_keeps_saving():
    blocked = os.path.join(work_dir, 'blocked')
    os.makedirs(f'{blocked}.jsonl') # a folder where the log would go
    with capture_log() as captured, log_name(blocked):
        name = save()
    assert os.path.isfile(name), f'image not saved: {name}'
    errors = [m for m in captured.messages(logging.ERROR) if 'blocked' in m]
    assert errors, f'no error logged for the unwritable log: {captured.messages(logging.WARNING)}'
    target = os.path.join(work_dir, 'after.jsonl')
    with log_name(target):
        save()
    assert len(lines(target)) == 1, 'the next save did not log'


def run_all():
    log.warning('=== Save log ===')
    cat = category('append')
    for fn in [
        test_first_saves_create_the_log,
        test_append_keeps_the_file,
        test_one_save_writes_one_line,
    ]:
        run_test(cat, fn)
    cat = category('existing')
    for fn in [
        test_json_log_left_untouched,
        test_existing_log_continues,
        test_torn_last_line_kept_apart,
        test_log_changed_elsewhere_recounted,
        test_name_resolves_to_jsonl,
    ]:
        run_test(cat, fn)
    cat = category('failures')
    run_test(cat, test_unwritable_log_keeps_saving)

    log.warning('=== Results ===')
    total = {'passed': 0, 'failed': 0, 'skipped': 0}
    for cat_name, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        for key in total:
            total[key] += info[key]
    log.warning(f"Total: {total['passed']} passed, {total['failed']} failed, {total['skipped']} skipped")
    return total['failed'] == 0


if __name__ == '__main__':
    t0 = time.time()
    try:
        ok = run_all()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
