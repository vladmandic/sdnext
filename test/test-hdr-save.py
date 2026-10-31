#!/usr/bin/env python
"""
Offline test for the HDR script's 16-bit PNG save on paths with non-ASCII characters.

``cv2.imwrite`` writes nothing and returns False for such paths on Windows; the script encodes
with ``cv2.imencode`` and writes the bytes itself. The ``cv2.imwrite`` control is logged so a
Windows run shows the failure the save avoids.

No running server required.

Usage:
    python test/test-hdr-save.py
"""

import os
import sys
import time
import shutil
import tempfile
import importlib.util

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

import cv2                                                           # pylint: disable=wrong-import-position
import numpy as np                                                   # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position

results = {'passed': 0, 'failed': 0}
work_dir = tempfile.mkdtemp(prefix='sdnext-hdr-save-', suffix='-work')


def load_hdr_script():
    spec = importlib.util.spec_from_file_location('hdr_script', os.path.join(script_dir, 'scripts', 'hdr.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hdr_array() -> np.ndarray:
    """16-bit three-channel gradient, the dtype and layout HDRScript.merge returns."""
    y, x = np.mgrid[0:32, 0:48]
    return np.stack([x * 1300, y * 2000, (x + y) * 800], axis=-1).astype(np.uint16)


def test_hdr_png_round_trip_on_non_ascii_path():
    hdr = load_hdr_script()
    folder = os.path.join(work_dir, 'выход 出力') # Cyrillic and CJK
    os.makedirs(folder)
    array = hdr_array()
    control = os.path.join(folder, 'control.png')
    log.info(f'  control: cv2.imwrite non-ascii path returned={cv2.imwrite(control, array)} file={os.path.isfile(control)}')
    fn = os.path.join(folder, 'image-ü-hdr.png')
    assert hdr.save_hdr_png(fn, array) is True, 'save reported failure'
    assert os.path.isfile(fn), f'no file at "{fn}"'
    read = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert read is not None and read.dtype == np.uint16 and read.shape == array.shape, f'read back {None if read is None else (read.dtype, read.shape)}'
    assert np.array_equal(read, array), 'pixels changed in the round trip'


def run_all():
    log.warning('=== HDR save ===')
    for fn in [test_hdr_png_round_trip_on_non_ascii_path]:
        try:
            fn()
            results['passed'] += 1
            log.info(f'  PASS: {fn.__name__}')
        except AssertionError as e:
            results['failed'] += 1
            log.error(f'  FAIL: {fn.__name__} ({e})')
        except Exception as e:  # pylint: disable=broad-except
            results['failed'] += 1
            log.error(f'  FAIL: {fn.__name__} (exception: {type(e).__name__}: {e})')
    log.warning(f"Total: {results['passed']} passed, {results['failed']} failed")
    return results['failed'] == 0


if __name__ == '__main__':
    t0 = time.time()
    try:
        ok = run_all()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
