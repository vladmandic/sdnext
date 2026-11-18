#!/usr/bin/env python
"""
Offline tests for the DLSS pass over a finished job (``scripts/dlss_ext.py`` ``DLSSScript.postprocess``).

A job that returns a grid puts it first in ``Processed.images``, built from the renders and saved before
scripts postprocess. The DLSS pass enhances the samples only; the grid is returned as it was saved.

Covers:

- a leading grid stays out of the DLSS pass and comes back as the same image, without added metadata
- the samples are enhanced, saved once with their own infotexts, and follow the grid
- a job without a grid passes every image (control)
- DLSS off leaves the images as they were (control)
- an exception in the pass leaves the grid in place

``dlss_run`` and ``process_samples`` are stubs. No model, no running server.

Usage:
    python test/test-dlss-batch.py
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

from PIL import Image                                            # pylint: disable=wrong-import-position
from modules import shared                                       # pylint: disable=wrong-import-position,unused-import
from modules.errors import log                                   # pylint: disable=wrong-import-position
from scripts import dlss_ext                                     # pylint: disable=wrong-import-position

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


def job(count: int, grid: bool):
    """A finished job as process_images hands it to scripts postprocess: optional grid first, then the samples."""
    samples = [Image.new('RGB', (64, 64), (40 * i, 0, 0)) for i in range(count)]
    infotexts = [f'sample {i}' for i in range(count)]
    images = list(samples)
    first = 0
    if grid:
        images.insert(0, Image.new('RGB', (64 * count, 64), (0, 0, 255)))
        infotexts.insert(0, 'grid')
        first = 1
    pp = SimpleNamespace(images=images, infotexts=infotexts, index_of_first_image=first)
    p = SimpleNamespace(xyz=False, ops=[], do_not_save_samples=True, dlss_do_not_save=False,
                        prompts=['last batch'], negative_prompts=[''], seeds=[count - 1],
                        all_prompts=[f'prompt {i}' for i in range(count)], all_negative_prompts=[''] * count, all_seeds=list(range(count)))
    return p, pp, samples


class Stubs:
    """dlss_run and process_samples replaced; each call records what it was given."""

    def __init__(self, mode='enhance'):
        self.mode = mode
        self.dlss_inputs = []
        self.saved = []
        self.saved_seeds = []

    def dlss_run(self, p, pp, *args, **kwargs): # pylint: disable=unused-argument
        self.dlss_inputs.append(list(pp.images))
        if self.mode == 'off':
            return None
        if self.mode == 'raise':
            raise RuntimeError('dlss failed')
        pp.images = [Image.new('RGB', image.size, (0, 255, 0)) for image in pp.images]
        return pp

    def process_samples(self, p, samples, skip_postprocess_image=False, save_only=False): # pylint: disable=unused-argument
        self.saved.append((list(samples), save_only))
        self.saved_seeds.append(list(p.seeds))
        return list(samples), [f'dlss {i}' for i in range(len(samples))]

    def __enter__(self):
        self.real = dlss_ext.dlss_run, dlss_ext.processing.process_samples
        dlss_ext.dlss_run = self.dlss_run
        dlss_ext.processing.process_samples = self.process_samples
        return self

    def __exit__(self, *exc):
        dlss_ext.dlss_run, dlss_ext.processing.process_samples = self.real
        return False


def postprocess(p, pp):
    script = dlss_ext.DLSSScript.__new__(dlss_ext.DLSSScript) # skips the xyz registration the constructor does
    return script.postprocess(p, pp, ['NeuralRender'])


def test_grid_stays_out_of_the_pass():
    p, pp, samples = job(3, grid=True)
    grid = pp.images[0]
    grid_info = dict(grid.info)
    with Stubs() as stubs:
        out = postprocess(p, pp)
    assert len(stubs.dlss_inputs) == 1, f'calls={len(stubs.dlss_inputs)}'
    given = stubs.dlss_inputs[0]
    assert len(given) == 3 and all(a is b for a, b in zip(given, samples)), f'the pass was given {[im.size for im in given]}, expected the 3 samples only'
    assert out.images[0] is grid, 'the grid must come back as the image that was saved'
    assert dict(grid.info) == grid_info, f'grid metadata changed: {grid.info}'
    assert out.infotexts[0] == 'grid', out.infotexts
    return True


def test_samples_enhanced_and_saved_once():
    p, pp, _ = job(3, grid=True)
    with Stubs() as stubs:
        out = postprocess(p, pp)
    assert len(stubs.saved) == 1 and stubs.saved[0][1] is True, f'saves={[(len(s), so) for s, so in stubs.saved]}'
    assert len(stubs.saved[0][0]) == 3, f'saved {len(stubs.saved[0][0])} images, expected the 3 samples'
    assert stubs.saved_seeds == [[0, 1, 2]], f'seeds while saving {stubs.saved_seeds}'
    assert len(out.images) == 4 and all(im.getpixel((0, 0)) == (0, 255, 0) for im in out.images[1:]), 'the samples must be the enhanced images, after the grid'
    assert out.infotexts == ['grid', 'dlss 0', 'dlss 1', 'dlss 2'], out.infotexts
    assert p.seeds == [2], f'the batch seeds must be restored, got {p.seeds}'
    return True


def test_job_without_grid_passes_every_image():
    p, pp, samples = job(2, grid=False)
    with Stubs() as stubs:
        out = postprocess(p, pp)
    assert len(stubs.dlss_inputs[0]) == 2 and all(a is b for a, b in zip(stubs.dlss_inputs[0], samples))
    assert len(out.images) == 2 and out.infotexts == ['dlss 0', 'dlss 1'], out.infotexts
    return True


def test_dlss_off_leaves_images():
    p, pp, _ = job(3, grid=True)
    before = list(pp.images)
    with Stubs(mode='off'):
        out = postprocess(p, pp)
    assert len(out.images) == 4 and all(a is b for a, b in zip(out.images, before)), 'images changed with DLSS off'
    assert out.infotexts == ['grid', 'sample 0', 'sample 1', 'sample 2'], out.infotexts
    return True


def test_failed_pass_keeps_the_grid():
    p, pp, _ = job(3, grid=True)
    before = list(pp.images)
    raised = False
    with Stubs(mode='raise'):
        try:
            postprocess(p, pp)
        except RuntimeError:
            raised = True
    assert raised, 'the stub must have raised'
    assert len(pp.images) == 4 and all(a is b for a, b in zip(pp.images, before)), f'images after a failed pass: {[im.size for im in pp.images]}'
    return True


def run_all():
    log.warning('=== DLSS over a finished job ===')
    cat = category('dlss-batch')
    for fn in [
        test_grid_stays_out_of_the_pass,
        test_samples_enhanced_and_saved_once,
        test_job_without_grid_passes_every_image,
        test_dlss_off_leaves_images,
        test_failed_pass_keeps_the_grid,
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
