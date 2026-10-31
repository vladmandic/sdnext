#!/usr/bin/env python
"""
Offline unit tests for a generation pass that fails, against a user interrupt.

A pass that raises records its message on ``State.error`` and stops the job through ``interrupted``,
so the job reports ``failed``, returns and saves no images, and the video routes answer 500 with the
message; a user interrupt reports ``interrupted``, keeps its last live preview, and the video routes
answer 499. The pipeline is a stub that leaves a preview, then raises or interrupts during its call.

Covers:

- ``State.fail``: error text, the stop flag, the reported status, cleared by ``begin`` only
- ``process_images`` with a failing and an interrupted pipeline
- ``video_run.run`` with a failing and an interrupted pipeline
- an on-demand component returned to cpu when the pass raises inside its entry point

No running server required.

Usage:
    python test/test-processing-failure.py
"""

import os
import sys
import time
import shutil
import tempfile
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

import torch                                                         # pylint: disable=wrong-import-position
import diffusers                                                     # pylint: disable=wrong-import-position
from diffusers.utils.accelerate_utils import apply_forward_hook      # pylint: disable=wrong-import-position
from PIL import Image                                                # pylint: disable=wrong-import-position
from modules import shared, devices                                  # pylint: disable=wrong-import-position
from modules.modeldata import model_data                             # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position

devices.torch_gc = lambda *args, **kwargs: None  # begin/end call it; nothing here holds device memory
model_data.locked = False  # webui.py unlocks it at startup

results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-processing-failure-')
state = shared.state
PREVIEW = (255, 0, 0)
RESULT = (0, 0, 255)

shared.opts.data['sd_model_checkpoint'] = 'None'  # an empty slot must not trigger a model load
shared.opts.data['sd_model_refiner'] = 'None'
shared.opts.data['samples_save'] = True
shared.opts.data['save_images_add_number'] = False
shared.opts.data['keep_incomplete'] = True
for key in ('outdir_samples', 'outdir_txt2img_samples', 'outdir_video', 'outdir_save', 'outdir_grids', 'outdir_txt2img_grids'):
    shared.opts.data[key] = work_dir


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
        record(cat, 'FAIL', name, f'exception: {e}')
    finally:
        if state.job != '':
            state.end()


class StubPipeline(diffusers.DiffusionPipeline):
    """Pipeline whose call leaves a live preview, then raises, interrupts or returns."""

    def __init__(self, mode: str, preview: bool = True):
        super().__init__()
        self.register_to_config(_name_or_path='stub')
        self.mode = mode
        self.preview = preview
        self.sd_checkpoint_info = SimpleNamespace(name='stub', title='stub', filename='stub.safetensors', hash=None, sha256=None, shorthash=None, model_name='stub', type='diffusers')
        self.sd_model_hash = None

    def __call__(self, prompt=None, negative_prompt=None, num_inference_steps=4, width=64, height=64, generator=None, output_type='pil', num_images_per_prompt=1, **kwargs):  # pylint: disable=unused-argument
        if self.preview:
            state.current_image = Image.new('RGB', (64, 64), PREVIEW)
        if self.mode == 'fail':
            raise RuntimeError('stub pipeline failure')
        if self.mode == 'fail-decode':
            self.audio_vae.decode(torch.zeros(1, 4, device=devices.device))
            raise RuntimeError('stub decode failure')
        if self.mode == 'interrupt':
            state.interrupt()
        return SimpleNamespace(images=[Image.new('RGB', (64, 64), RESULT)])


class BridgeDecoder(torch.nn.Module):
    """Component entered through decode, the shape the group engine gives an on-demand hook."""
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    @apply_forward_hook
    def decode(self, x):
        return self.proj(x)


class WanPipeline(StubPipeline):  # validate_pipeline and resolve_model match video pipelines by class name
    pass


class StubRunner:
    def run(self, p, *args):  # pylint: disable=unused-argument
        return None


def files() -> set:
    return {os.path.relpath(os.path.join(d, f), work_dir) for d, _, fs in os.walk(work_dir) for f in fs}


def generate(mode: str, preview: bool = True, ondemand: torch.nn.Module | None = None) -> dict:
    """One txt2img job on the stub; what it returned, saved and reported."""
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    pipe = StubPipeline(mode, preview)
    if ondemand is not None:
        pipe.audio_vae = ondemand
        pipe.sdnext_ondemand_modules = ['audio_vae']
    shared.sd_model = pipe
    before = files()
    p = StableDiffusionProcessingTxt2Img(prompt='stub', steps=4, width=64, height=64, n_iter=1, batch_size=1, seed=1)
    p.scripts = None
    p.outpath_samples = work_dir
    p.do_not_save_samples = False
    jobid = state.begin('API-TXT', api=True)
    try:
        processed = process_images(p)
        outcome = {
            'images': [img.getpixel((0, 0)) for img in processed.images],
            'error': state.error,
            'status': state.status().status,
        }
    finally:
        state.end(jobid)
    time.sleep(0.5)  # image saves can finish on a worker thread
    outcome['saved'] = sorted(files() - before)
    return outcome


def generate_video(mode: str, preview: bool = True) -> dict:
    """One video job on the stub through video_run; the result or the VideoError it raised."""
    from modules.video_models import video_run
    shared.sd_model = WanPipeline(mode, preview)
    selected, needs_load = video_run.resolve_model(None, None)
    jobid = state.begin('Video', api=True)
    try:
        result = video_run.run(selected, prompt='stub', width=64, height=64, frames=1, steps=4, needs_load=needs_load, scripts=StubRunner(), mp4_video=False, mp4_thumb=False)
        return {'frames': result.num_frames}
    except video_run.VideoError as e:
        return {'code': e.code, 'message': str(e)}
    finally:
        state.end(jobid)


# --- State ---

def test_fail_records_the_message_and_stops_the_job():
    state.begin('Generate')
    state.fail(RuntimeError('out of memory'))
    assert state.error == 'out of memory', state.error
    assert state.interrupted, 'fail must set the stop flag the processing loops check'
    assert state.status().status == 'failed', state.status().status
    return True


def test_fail_without_a_message_names_the_exception():
    state.begin('Generate')
    state.fail(RuntimeError())
    assert state.error == 'RuntimeError', state.error
    return True


def test_error_outlives_stage_ends_until_the_next_begin():
    state.begin('Generate')
    stage = state.begin('Inference')
    state.fail(ValueError('bad shape'))
    state.end(stage)
    assert state.error == 'bad shape', f'error cleared by a stage end: {state.error!r}'
    state.end()
    assert state.error == 'bad shape', f'error cleared by the job end: {state.error!r}'
    state.begin('Generate')
    assert state.error == '' and not state.interrupted, (state.error, state.interrupted)
    return True


def test_user_interrupt_reports_interrupted():
    state.begin('Generate')
    state.interrupt()
    assert state.error == '', state.error
    assert state.status().status == 'interrupted', state.status().status
    return True


# --- Processing ---

def test_failed_pass_reports_failed_with_the_message():
    out = generate('fail', preview=False)
    assert out['error'] == 'stub pipeline failure', out
    assert out['status'] == 'failed', out
    assert out['images'] == [] and out['saved'] == [], out
    return True


def test_failed_pass_returns_and_saves_no_preview():
    out = generate('fail')
    assert out['status'] == 'failed', out
    assert out['images'] == [], f'failed pass returned its live preview: {out}'
    assert out['saved'] == [], f'failed pass saved its live preview: {out}'
    return True


def test_user_interrupt_keeps_the_preview():
    out = generate('interrupt')
    assert out['error'] == '' and out['status'] == 'interrupted', out
    assert out['images'] == [PREVIEW], f'interrupted job should return its last preview: {out}'
    assert len(out['saved']) == 1, f'keep_incomplete should save the preview: {out}'
    return True


def test_failed_decode_returns_the_ondemand_component():
    if not torch.cuda.is_available():
        return 'no accelerator for the on-demand onload'
    from modules import sd_offload_group
    decoder = BridgeDecoder()
    sd_offload_group.apply_group_offload_ondemand(decoder)
    mode = shared.opts.diffusers_offload_mode
    shared.opts.data['diffusers_offload_mode'] = 'group' # the engine that places on-demand components; balanced seams would re-hook the stub
    try:
        out = generate('fail-decode', preview=False, ondemand=decoder)
    finally:
        shared.opts.data['diffusers_offload_mode'] = mode
    assert out['error'] == 'stub decode failure', out
    device = next(decoder.parameters()).device
    assert device.type == 'cpu', f'failed decode left the on-demand component on {device}'
    assert getattr(decoder, 'sdnext_loaded', None) is None, 'the onload record must not outlive the job'
    return True


# --- Video ---

def test_video_failed_pass_answers_500_with_the_message():
    out = generate_video('fail', preview=False)
    assert out == {'code': 500, 'message': 'stub pipeline failure'}, out
    return True


def test_video_failed_pass_with_a_preview_answers_500():
    out = generate_video('fail')
    assert out == {'code': 500, 'message': 'stub pipeline failure'}, f'failed pass returned its live preview as a video: {out}'
    return True


def test_video_user_interrupt_answers_499():
    out = generate_video('interrupt', preview=False)
    assert out == {'code': 499, 'message': 'interrupted'}, out
    return True


def run_all():
    log.warning('=== Processing failure ===')
    cat = category('state')
    for fn in [
        test_fail_records_the_message_and_stops_the_job,
        test_fail_without_a_message_names_the_exception,
        test_error_outlives_stage_ends_until_the_next_begin,
        test_user_interrupt_reports_interrupted,
    ]:
        run_test(cat, fn)
    cat = category('processing')
    for fn in [
        test_failed_pass_reports_failed_with_the_message,
        test_failed_pass_returns_and_saves_no_preview,
        test_user_interrupt_keeps_the_preview,
        test_failed_decode_returns_the_ondemand_component,
    ]:
        run_test(cat, fn)
    cat = category('video')
    for fn in [
        test_video_failed_pass_answers_500_with_the_message,
        test_video_failed_pass_with_a_preview_answers_500,
        test_video_user_interrupt_answers_499,
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
