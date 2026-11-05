#!/usr/bin/env python
"""
Offline tests for settings files whose folder does not exist yet.

Covers:

- ``config.json`` and ``secrets.json`` saved by the ``modules.shared`` import into a new data folder (``SD_DATADIR``)
- the same with ``SD_CONFIG`` and ``SD_SECRETS`` in two other new folders

Each case imports in a fresh interpreter, since ``modules.paths`` resolves the folders once per process.

No running server required.

Usage:
    python test/test-data-dir.py
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

from modules.errors import log  # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-data-dir-', suffix='-work')

IMPORT_SHARED = '''
import os, sys, json, time
sys.argv = [sys.argv[0]]
import modules.shared
files = [modules.shared.cmd_opts.config, modules.shared.cmd_opts.secrets]
for _i in range(50): # Options.save writes from a thread
    if all(os.path.isfile(f) for f in files):
        break
    time.sleep(0.1)
print("SAVED " + json.dumps({f: os.path.isfile(f) for f in files}))
'''


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


def import_shared(env_extra: dict) -> dict:
    """Import modules.shared in a fresh interpreter, returning which settings files exist afterwards."""
    env = {k: v for k, v in os.environ.items() if k not in ('SD_DATADIR', 'SD_CONFIG', 'SD_SECRETS')}
    env.update(env_extra, SD_INSTALL_QUIET='1')
    proc = subprocess.run([sys.executable, '-c', IMPORT_SHARED], cwd=script_dir, env=env, capture_output=True, text=True, timeout=300, check=False)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f'import failed: {output[-2000:]}'
    failed = [line.strip() for line in output.splitlines() if 'Save failed' in line]
    assert failed == [], f'failed saves: {failed}'
    saved = next((line[len('SAVED '):] for line in proc.stdout.splitlines() if line.startswith('SAVED ')), None)
    assert saved is not None, f'no result: {output[-2000:]}'
    return json.loads(saved)


def test_new_data_dir():
    """a data folder that does not exist yet receives config.json and secrets.json"""
    data_dir = os.path.join(work_dir, 'data-dir', 'data')
    saved = import_shared({'SD_DATADIR': data_dir})
    expected = {os.path.join(data_dir, 'config.json'): True, os.path.join(data_dir, 'secrets.json'): True}
    assert saved == expected, f'saved: {saved}'


def test_new_config_folders():
    """config.json and secrets.json given in two other folders that do not exist yet are saved there"""
    config = os.path.join(work_dir, 'files', 'config', 'config.json')
    secrets = os.path.join(work_dir, 'files', 'secrets', 'secrets.json')
    saved = import_shared({'SD_DATADIR': os.path.join(work_dir, 'files', 'data'), 'SD_CONFIG': config, 'SD_SECRETS': secrets})
    assert saved == {config: True, secrets: True}, f'saved: {saved}'


def run_all():
    log.warning('=== Data dir ===')
    cat = category('new folders')
    for fn in [
        test_new_data_dir,
        test_new_config_folders,
    ]:
        run_test(cat, fn)

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
