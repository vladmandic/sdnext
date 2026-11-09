#!/usr/bin/env python
"""
Offline tests for the CUDA allocator settings ``installer.set_environment`` exports.

Covers:

- the default string from the GC threshold option
- the expandable segments option on its own and with the cudaMallocAsync backend
- a ``PYTORCH_CUDA_ALLOC_CONF`` set before launch, which is kept

Each case runs ``set_environment`` in a fresh interpreter, since it only fills variables that are unset.

No running server required.

Usage:
    python test/test-torch-allocator.py
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
work_dir = tempfile.mkdtemp(prefix='sdnext-torch-allocator-')
NAMES =('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF')

SET_ENVIRONMENT = '''
import os, sys, json
sys.argv = [sys.argv[0]]
import installer
installer.setup_logging(filename=os.environ['TEST_LOG'])
installer.opts = json.loads(os.environ['TEST_OPTS'])
installer.set_environment()
print("ENV " + json.dumps({k: os.environ.get(k) for k in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_HIP_ALLOC_CONF')}))
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


def exported(opts: dict, preset: dict | None = None) -> dict:
    """Run set_environment with these options in a fresh interpreter and return the three allocator variables."""
    env = {k: v for k, v in os.environ.items() if k not in NAMES}
    env.update(preset or {}, TEST_OPTS=json.dumps(opts), TEST_LOG=os.path.join(work_dir, 'test.log'), SD_INSTALL_QUIET='1')
    proc = subprocess.run([sys.executable, '-c', SET_ENVIRONMENT], cwd=script_dir, env=env, capture_output=True, text=True, timeout=120, check=False)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, f'set_environment failed: {output[-2000:]}'
    line = next((line[len('ENV '):] for line in proc.stdout.splitlines() if line.startswith('ENV ')), None)
    assert line is not None, f'no result: {output[-2000:]}'
    return json.loads(line)


def test_default_string():
    """without options the GC threshold and max split size are exported under all three names"""
    env = exported({})
    expected = 'garbage_collection_threshold:0.80,max_split_size_mb:512'
    assert env == dict.fromkeys(NAMES, expected), f'env: {env}'


def test_gc_threshold_option():
    """the GC threshold option sets the threshold"""
    env = exported({'torch_gc_threshold': 70})
    assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'garbage_collection_threshold:0.70,max_split_size_mb:512', f'env: {env}'


def test_expandable_alone():
    """expandable segments are exported without the GC threshold and max split size"""
    env = exported({'torch_expandable_segments': True, 'torch_gc_threshold': 70})
    assert env == dict.fromkeys(NAMES, 'expandable_segments:True'), f'env: {env}'


def test_expandable_with_async_backend():
    """the cudaMallocAsync backend is kept beside expandable segments"""
    env = exported({'torch_expandable_segments': True, 'torch_malloc': 'cudaMallocAsync'})
    assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'expandable_segments:True,backend:cudaMallocAsync', f'env: {env}'


def test_preset_variable_kept():
    """a PYTORCH_CUDA_ALLOC_CONF set before launch is left as it is"""
    env = exported({'torch_expandable_segments': True}, preset={'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:256'})
    assert env['PYTORCH_CUDA_ALLOC_CONF'] == 'max_split_size_mb:256', f'env: {env}'


def run_all():
    log.warning('=== Torch allocator ===')
    cat = category('allocator')
    for fn in [
        test_default_string,
        test_gc_threshold_option,
        test_expandable_alone,
        test_expandable_with_async_backend,
        test_preset_variable_kept,
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
