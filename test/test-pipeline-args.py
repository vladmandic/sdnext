#!/usr/bin/env python
"""
Offline tests for the call arguments modules.processing_args builds for a pipeline.

Covers:

- a resized hires pass on a modular pipeline that takes condition images, called at the hires size

No running server required.

Usage:
    python test/test-pipeline-args.py
"""

import os
import sys
import time
from types import SimpleNamespace

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
from modules import shared                                           # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position
from modules.modeldata import model_data                             # pylint: disable=wrong-import-position

model_data.locked = False  # webui.py unlocks it at startup
results: dict[str, dict] = {}


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
        log.info(msg, extra={'markup': False})
    elif status == 'SKIP':
        log.warning(msg, extra={'markup': False})
    else:
        log.error(msg, extra={'markup': False})


def run_test(cat: str, fn):
    name = fn.__name__
    previous = shared.sd_model
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
    finally:
        shared.sd_model = previous


class ConditionModularPipeline:
    """What the argument builders read from a modular pipeline that takes condition images."""
    max_condition_images = 10
    vae_scale_factor = 8
    patch_size = None


def hires_p(**overrides):
    values = {
        'ops': [], 'task_args': {}, 'width': 704, 'height': 1280, 'denoising_strength': 0.5,
        'is_hr_pass': True, 'hr_resize_mode': 1, 'hr_upscaler': 'Resize Lanczos', 'hr_upscale_to_x': 1056, 'hr_upscale_to_y': 1920,
        'init_images': [Image.new('RGB', (1056, 1920))],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ============================================================
# size
# ============================================================

def test_modular_hires_pass_at_hires_size():
    from modules import processing_args
    model = ConditionModularPipeline()
    shared.sd_model = model
    args = processing_args.task_modular_kwargs(hires_p(), model)
    assert (args.get('width'), args.get('height')) == (1056, 1920), f"hires call sized {args.get('width')}x{args.get('height')}"


def test_modular_base_pass_at_request_size():
    """Control: outside hires the request sets the size, whatever the condition image's own size."""
    from modules import processing_args
    model = ConditionModularPipeline()
    shared.sd_model = model
    p = hires_p(is_hr_pass=False, init_images=[Image.new('RGB', (1024, 1024))])
    args = processing_args.task_modular_kwargs(p, model)
    assert (args.get('width'), args.get('height')) == (704, 1280), f"base call sized {args.get('width')}x{args.get('height')}"


def run_all():
    log.warning('=== Pipeline args ===')
    cat = category('size')
    for fn in [
        test_modular_hires_pass_at_hires_size,
        test_modular_base_pass_at_request_size,
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
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
