#!/usr/bin/env python
"""
Offline tests for how the xyz grids parse axis values and build each cell's prompt.

Covers:

- blank entries in a values list (a trailing comma and space) add no cell, in the Script grid,
  the always-on grid and the dropdown restore; escaped commas and inner spaces are kept
- a Search & replace axis whose values occur in none of the fields it targets logs one warning
  naming those fields; a value found in any targeted field, prompt or negative, logs none
- a dropdown axis writes its selection to the infotext and each grid's dropdown restore reads it
  back from that grid's own keys; the Script grid's keys, Fixed seed keys included, stay apart
  from the always-on grid's; a text axis records the values as typed
- in the always-on grid, which runs after process_init has built the prompt lists, axes that edit
  the prompt or styles (Search & replace all, Prompt order, Styles) reach the prompts each cell
  encodes, the base styles and negative prompt still apply once, and a write into one cell's
  prompt list does not reach the next cell

Both grids run their real run and process code. Image generation is replaced by a recorder that
runs process_init on the cell and keeps the prompts the pipeline would encode, and the grid
drawing by a loop that calls every cell. No model is loaded.

Usage:
    python test/test-xyz-prompt.py
"""

import logging
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


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def captured_warnings():
    handler = Capture()
    log.addHandler(handler)
    try:
        log.warning('xyz-test probe')
        assert handler.messages == ['xyz-test probe'], f'capture not working: {handler.messages}'
        yield handler
    finally:
        log.removeHandler(handler)


def search_warnings(handler: Capture) -> list[str]:
    return [m for m in handler.messages if m.startswith('XYZ grid: search')]


def make_p(prompt: str = 'a cat', negative: str = '', styles: list | None = None, **kwargs):
    p = processing.StableDiffusionProcessingTxt2Img(prompt=prompt, negative_prompt=negative, styles=styles or [], seed=1, steps=1, batch_size=1, n_iter=1, width=64, height=64, **kwargs)
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


def test_unmatched_values_warn():
    with captured_warnings() as handler:
        run_script_grid(make_p('a cat'), '[Prompt] Search & replace', 'horse, cow')
    warnings = search_warnings(handler)
    assert len(warnings) == 1 and "'prompt'" in warnings[0] and "'horse'" in warnings[0], f'warnings={warnings}'


def test_unmatched_values_warn_in_always_on_grid():
    with captured_warnings() as handler:
        run_always_on_grid(make_p('a cat'), '[Prompt] Search & replace', 'horse, cow')
    warnings = search_warnings(handler)
    assert len(warnings) == 1, f'warnings={warnings}'


def test_refine_variant_names_the_empty_refine_field():
    with captured_warnings() as handler:
        run_script_grid(make_p('a cat'), '[Prompt] Search & replace refine', 'cat, dog')
    warnings = search_warnings(handler)
    assert len(warnings) == 1 and "'refiner_prompt'" in warnings[0] and "'prompt'" not in warnings[0], f'warnings={warnings}'


def test_any_matching_value_does_not_warn():
    with captured_warnings() as handler:
        run_script_grid(make_p('a lazy dog'), '[Prompt] Search & replace', 'cat, dog, monkey')
        run_script_grid(make_p('a cat', negative='blurry'), '[Prompt] Search & replace', 'blurry, sharp')
        run_script_grid(make_p('a dog', detailer_prompt='a cat face'), '[Prompt] Search & replace all', 'cat, fox')
    warnings = search_warnings(handler)
    assert warnings == [], f'warnings={warnings}'


def infotext_params(recorder: GridRecorder) -> dict:
    params = {}
    for written in recorder.infotext_params:
        params.update({key: str(value) for key, value in written.items()})
    return params


def test_script_grid_dropdown_restores_its_own_values():
    recorder = run_script_grid(make_p('a cat'), '[Sampler] Name', x_dropdown=['Euler a', 'UniPC'])
    params = infotext_params(recorder)
    assert dropdown_restore(script_grid, 'X', params) == ['Euler a', 'UniPC'], f'params={params}'


def test_script_grid_dropdown_ignores_always_on_values():
    recorder = run_script_grid(make_p('a cat'), '[Sampler] Name', x_dropdown=['Euler a', 'UniPC'])
    params = {**infotext_params(recorder), 'X Type': '[Sampler] Name', 'X Values': 'DPM++ 2M'}
    values = dropdown_restore(script_grid, 'X', params)
    assert values == ['Euler a', 'UniPC'], f'values={values}'


def test_always_on_dropdown_restores_its_own_values():
    p = make_p('a cat')
    recorder = run_always_on_grid(p, '[Sampler] Name', x_dropdown=['Euler a', 'UniPC'])
    params = infotext_params(recorder)
    values = dropdown_restore(always_on_grid, 'X', params)
    assert values == ['Euler a', 'UniPC'], f'values={values}'


def test_text_axis_records_the_typed_values():
    params = infotext_params(run_script_grid(make_p('a cat'), '[Prompt] Search & replace', r'cat, a dog\, sleeping'))
    assert params.get('X Script Values') == r'cat, a dog\, sleeping', f'params={params}'


def test_script_grid_fixed_seed_keys_are_its_own():
    recorder = run_script_grid(make_p('a cat'), '[Param] Seed', '1, 2', z_label='[Param] Seed', z_values='3, 4')
    fixed = sorted(key for key in infotext_params(recorder) if key.startswith('Fixed'))
    assert fixed == ['Fixed X Script Values', 'Fixed Z Script Values'], f'fixed={fixed}'


def test_always_on_search_and_replace_all_reaches_the_encode():
    prompts = run_always_on_grid(make_p('a cat'), '[Prompt] Search & replace all', 'cat, dog').prompts()
    assert prompts == ['a cat', 'a dog'], f'prompts={prompts}'


def test_always_on_prompt_order_reaches_the_encode():
    prompts = run_always_on_grid(make_p('red, blue'), '[Prompt] Prompt order', 'red, blue').prompts()
    assert prompts == ['red, blue', 'blue, red'], f'prompts={prompts}'


def test_always_on_styles_axis_reaches_the_encode():
    from modules import shared
    from modules.styles import Style
    saved = dict(shared.prompt_styles.styles)
    shared.prompt_styles.styles['xyz-test style'] = Style('xyz-test style', prompt='{prompt}, oil painting')
    try:
        prompts = run_always_on_grid(make_p('a cat'), '[Network] Styles', x_dropdown=['xyz-test style']).prompts()
    finally:
        shared.prompt_styles.styles.clear()
        shared.prompt_styles.styles.update(saved)
    assert prompts == ['a cat, oil painting'], f'prompts={prompts}'


def test_always_on_cells_keep_the_base_negative_and_styles():
    from modules import shared
    from modules.styles import Style
    saved = dict(shared.prompt_styles.styles)
    shared.prompt_styles.styles['xyz-test base'] = Style('xyz-test base', prompt='{prompt}, detailed', negative_prompt='blurry')
    try:
        p = make_p('a cat', negative='lowres', styles=['xyz-test base'])
        recorder = run_always_on_grid(p, '[Prompt] Search & replace all', 'cat, dog')
    finally:
        shared.prompt_styles.styles.clear()
        shared.prompt_styles.styles.update(saved)
    cells = [(cell['prompts'][0], cell['negative_prompts'][0]) for cell in recorder.cells]
    assert cells == [('a cat, detailed', 'lowres, blurry'), ('a dog, detailed', 'lowres, blurry')], f'cells={cells}'


def test_lora_tag_write_stays_in_its_cell():
    def tag_writer(pc):
        processing.process_init(pc)
        pc.all_prompts[0] += ', tag' # extra_networks_lora.prompt writes tags into all_prompts in place
        recorder.cells.append({'prompts': list(pc.prompts), 'negative_prompts': list(pc.negative_prompts)})
    p = make_p('a cat')
    processing.process_init(p)
    base = list(p.all_prompts)
    grid_on.active = False
    with recorded_grid() as recorder:
        processing.process_images = tag_writer
        always_on_grid.process(p, True, *grid_args('[Prompt] Search & replace all', 'cat, dog, fox'))
    assert p.all_prompts == base, f'base all_prompts={p.all_prompts}'


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

    log.warning('=== xyz search and replace ===')
    cat = category('search')
    for fn in [
        test_unmatched_values_warn,
        test_unmatched_values_warn_in_always_on_grid,
        test_refine_variant_names_the_empty_refine_field,
        test_any_matching_value_does_not_warn,
    ]:
        run_test(cat, fn)

    log.warning('=== xyz infotext restore ===')
    cat = category('restore')
    for fn in [
        test_script_grid_dropdown_restores_its_own_values,
        test_script_grid_dropdown_ignores_always_on_values,
        test_always_on_dropdown_restores_its_own_values,
        test_text_axis_records_the_typed_values,
        test_script_grid_fixed_seed_keys_are_its_own,
    ]:
        run_test(cat, fn)

    log.warning('=== always-on grid prompts ===')
    cat = category('always-on')
    for fn in [
        test_always_on_search_and_replace_all_reaches_the_encode,
        test_always_on_prompt_order_reaches_the_encode,
        test_always_on_styles_axis_reaches_the_encode,
        test_always_on_cells_keep_the_base_negative_and_styles,
        test_lora_tag_write_stays_in_its_cell,
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
