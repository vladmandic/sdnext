#!/usr/bin/env python
"""
Offline tests for how the xyz grids parse axis values and build each cell's prompt.

Covers:

- blank entries in a values list (a trailing comma and space) add no cell, in the Script grid,
  the always-on grid and the dropdown restore; escaped commas and inner spaces are kept

Both grids run their real run and process code. Image generation is replaced by a recorder that
runs process_init on the cell and keeps the prompts the pipeline would encode, and the grid
drawing by a loop that calls every cell. No model is loaded.

Usage:
    python test/test-xyz-prompt.py
"""

import os
import sys
from contextlib import contextmanager
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

import gradio as gr                                               # pylint: disable=wrong-import-position
from modules.errors import log                                    # pylint: disable=wrong-import-position
from modules import processing                                    # pylint: disable=wrong-import-position
from scripts import xyz_grid as grid_script                       # pylint: disable=wrong-import-position
from scripts import xyz_grid_on as grid_on                        # pylint: disable=wrong-import-position


results: dict[str, dict] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'tests': []}
    return name


def record(cat: str, passed: bool, name: str, detail: str = ''):
    status = 'PASS' if passed else 'FAIL'
    results[cat]['passed' if passed else 'failed'] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if passed:
        log.info(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    try:
        ok = fn()
        record(cat, ok is not False, fn.__name__)
    except AssertionError as e:
        record(cat, False, fn.__name__, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, fn.__name__, f'exception: {e}')
        import traceback
        traceback.print_exc()


class GridRecorder:
    """Stands in for image generation and grid drawing; keeps what each cell would encode."""

    def __init__(self):
        self.cells: list[dict] = []
        self.infotext_params: list[dict] = []

    def process_images(self, pc):
        processing.process_init(pc)
        self.cells.append({'prompts': list(pc.prompts), 'negative_prompts': list(pc.negative_prompts)})

    def create_infotext(self, pc, *args, **kwargs): # pylint: disable=unused-argument
        self.infotext_params.append(dict(pc.extra_generation_params))
        return ''

    def draw(self, p, xs, ys, zs, cell, **kwargs): # pylint: disable=unused-argument
        for iz, z in enumerate(zs):
            for iy, y in enumerate(ys):
                for ix, x in enumerate(xs):
                    cell(x, y, z, ix, iy, iz)
        return SimpleNamespace(images=[])

    def prompts(self) -> list[str]:
        return [cell['prompts'][0] for cell in self.cells]


@contextmanager
def recorded_grid():
    recorder = GridRecorder()
    saved = (processing.process_images, processing.create_infotext, grid_script.draw_xyz_grid, grid_on.draw_xyz_grid)
    processing.process_images = recorder.process_images
    processing.create_infotext = recorder.create_infotext
    grid_script.draw_xyz_grid = recorder.draw
    grid_on.draw_xyz_grid = recorder.draw
    try:
        yield recorder
    finally:
        processing.process_images, processing.create_infotext, grid_script.draw_xyz_grid, grid_on.draw_xyz_grid = saved


def make_p(prompt: str = 'a cat', negative: str = '', **kwargs):
    p = processing.StableDiffusionProcessingTxt2Img(prompt=prompt, negative_prompt=negative, styles=[], seed=1, steps=1, batch_size=1, n_iter=1, width=64, height=64, **kwargs)
    p.do_not_save_grid = True
    p.do_not_save_samples = True
    return p


def make_script(module):
    script = module.XYZGridScript()
    script.parent = 'txt2img'
    with gr.Blocks():
        script.ui(False)
    return script


script_grid = None
always_on_grid = None


def grid_args(x_label: str, x_values: str = '', x_dropdown=None, z_label: str = 'Nothing', z_values: str = '', no_fixed_seeds: bool = False) -> list:
    return [
        x_label, x_values, x_dropdown or [],
        'Nothing', '', [],
        z_label, z_values, [],
        False, True, no_fixed_seeds, # csv_mode, draw_legend, no_fixed_seeds
        True, False, True, # include_grid, include_subgrids, include_images
        False, False, 0, # include_time, include_text, margin_size
        False, 'None', 2.0, False, 1, 0, # create_video, video_type, video_duration, video_loop, video_pad, video_interpolate
    ]


def run_script_grid(p, *args, **kwargs) -> GridRecorder:
    with recorded_grid() as recorder:
        script_grid.run(p, *grid_args(*args, **kwargs))
    return recorder


def run_always_on_grid(p, *args, **kwargs) -> GridRecorder:
    processing.process_init(p) # process_images_inner fills the prompt lists before the process hook
    grid_on.active = False
    with recorded_grid() as recorder:
        always_on_grid.process(p, True, *grid_args(*args, **kwargs))
    return recorder


def dropdown_restore(script, axis: str, params: dict) -> list:
    restores = [field for _component, field in script.infotext_fields if callable(field)] # the dropdown restores, in X, Y, Z order
    return restores['XYZ'.index(axis)](params)['value']


def test_trailing_comma_adds_no_cell_in_script_grid():
    prompts = run_script_grid(make_p('a cat'), '[Prompt] Search & replace', 'cat, dog, ').prompts()
    assert prompts == ['a cat', 'a dog'], f'prompts={prompts}'


def test_trailing_comma_adds_no_cell_in_always_on_grid():
    prompts = run_always_on_grid(make_p('a cat'), '[Prompt] Search & replace', 'cat, dog, ').prompts()
    assert prompts == ['a cat', 'a dog'], f'prompts={prompts}'


def test_blank_entry_left_out_of_dropdown_restore():
    values = dropdown_restore(always_on_grid, 'X', {'X Values': 'Euler a, , UniPC, '})
    assert values == ['Euler a', 'UniPC'], f'values={values}'


def test_escaped_comma_and_inner_spaces_kept():
    prompts = run_script_grid(make_p('a cat'), '[Prompt] Search & replace', r'a cat ,  a dog\, sleeping ').prompts()
    assert prompts == ['a cat', 'a dog, sleeping'], f'prompts={prompts}'


def run_all():
    global script_grid, always_on_grid # pylint: disable=global-statement
    script_grid = make_script(grid_script)
    always_on_grid = make_script(grid_on)

    log.warning('=== xyz values ===')
    cat = category('values')
    for fn in [
        test_trailing_comma_adds_no_cell_in_script_grid,
        test_trailing_comma_adds_no_cell_in_always_on_grid,
        test_blank_entry_left_out_of_dropdown_restore,
        test_escaped_comma_and_inner_spaces_kept,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat_name, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
