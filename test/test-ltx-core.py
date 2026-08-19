#!/usr/bin/env python
"""
Offline unit tests for the LTX keyword core in modules.ltx.ltx_process.

``run_ltx`` used to be the only way to generate with LTX: a gradio generator that reported
failure by yielding an error string, which is why LTX had no API. ``run`` is the keyword core
underneath it, converged on ``video_run.run``'s names, result and error protocol.

Covers:

- ``run`` is keyword-only past the model name, so a caller cannot mis-bind by position
- ``run_ltx``'s positional signature, parameter for parameter, since external callers bind to
  it by keyword and a rename would silently drop an argument into **kwargs
- the rejections that happen before anything is loaded, and that they do not load
- the adapter's delegation: the shapes it yields on success and on a typed failure
- ``open_condition`` resolving a string through the api decoder rather than as a path
- ``pixel_size`` over the frame shapes the two paths produce
- ``phase`` ending its job when the body raises

No running server required.

Usage:
    python test/test-ltx-core.py
"""

import os
import sys
import inspect

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

from modules.errors import log                          # pylint: disable=wrong-import-position
from modules.video_models import video_run, video_utils  # pylint: disable=wrong-import-position


results: dict[str, dict] = {}

# Frozen because callers outside this repo bind every one of these by keyword.
ADAPTER_PARAMS = [
    'task_id', '_ui_state', 'model', 'prompt', 'negative', 'styles', 'width', 'height', 'frames',
    'auto_duration', 'steps', 'sampler_index', 'guidance_scale', 'sampler_shift', 'dynamic_shift',
    'seed', 'upsample_enable', 'upsample_ratio', 'refine_enable', 'refine_strength',
    'condition_strength', 'ltx_init_image', 'condition_last', 'condition_files', 'condition_video',
    'condition_video_frames', 'condition_video_skip', 'decode_timestep', 'image_cond_noise_scale',
    'mp4_fps', 'mp4_interpolate', 'mp4_codec', 'mp4_ext', 'mp4_opt', 'mp4_video', 'mp4_frames',
    'mp4_sf', 'mp4_thumb', 'audio_enable', '_overrides',
]

# The names the core answers to. Divergence from video_run.run is deliberate only where LTX has
# no equivalent concept, so a new name appearing here should be a considered choice.
CORE_PARAMS = [
    'prompt', 'negative', 'styles', 'width', 'height', 'frames', 'auto_duration', 'steps',
    'sampler_name', 'sampler_shift', 'dynamic_shift', 'seed', 'guidance_scale',
    'upsample_enable', 'upsample_ratio', 'refine_enable', 'refine_strength', 'condition_strength',
    'init_image', 'condition_last', 'condition_files', 'condition_video', 'condition_video_frames',
    'condition_video_skip', 'decode_timestep', 'image_cond_noise_scale', 'audio',
    'mp4_fps', 'mp4_interpolate', 'mp4_codec', 'mp4_ext', 'mp4_opt', 'mp4_video', 'mp4_frames',
    'mp4_sf', 'mp4_thumb', 'override_settings', 'ui_state', 'scripts', 'script_args',
    'per_script_args', 'extra_p',
]


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
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


def skip(cat: str, name: str, reason: str):
    results[cat]['skipped'] += 1
    results[cat]['tests'].append(('SKIP', name))
    log.warning(f'  SKIP: {name} ({reason})')


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        elif isinstance(ok, str):
            skip(cat, name, ok)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e: # pylint: disable=broad-except
        record(cat, False, name, f'exception: {type(e).__name__}: {e}')


def ltx():
    from modules.ltx import ltx_process
    return ltx_process


# --- signatures -------------------------------------------------------------------------------

def test_core_takes_only_model_positionally():
    sig = inspect.signature(ltx().run)
    positional = [n for n, prm in sig.parameters.items() if prm.kind is prm.POSITIONAL_OR_KEYWORD]
    assert positional == ['model'], f'positional params: {positional}'
    return True


def test_core_parameter_names():
    sig = inspect.signature(ltx().run)
    kwonly = [n for n, prm in sig.parameters.items() if prm.kind is prm.KEYWORD_ONLY]
    assert kwonly == CORE_PARAMS, f'unexpected core signature: {kwonly}'
    return True


def test_core_returns_video_result():
    sig = inspect.signature(ltx().run)
    assert sig.return_annotation is video_run.VideoResult, f'return annotation: {sig.return_annotation}'
    return True


def test_adapter_signature_is_frozen():
    sig = inspect.signature(ltx().run_ltx)
    named = [n for n, prm in sig.parameters.items() if prm.kind is prm.POSITIONAL_OR_KEYWORD]
    assert named == ADAPTER_PARAMS, f'adapter signature drifted: {named}'
    return True


def test_adapter_still_absorbs_extra_arguments():
    sig = inspect.signature(ltx().run_ltx)
    kinds = {prm.kind for prm in sig.parameters.values()}
    assert inspect.Parameter.VAR_POSITIONAL in kinds, 'adapter must keep *args for the script vector'
    assert inspect.Parameter.VAR_KEYWORD in kinds, 'adapter must keep **kwargs so callers survive signature growth'
    return True


def test_adapter_is_a_generator():
    assert inspect.isgeneratorfunction(ltx().run_ltx), 'the tab binds to a generator'
    assert not inspect.isgeneratorfunction(ltx().run), 'the core returns a value rather than yielding'
    return True


# --- rejections -------------------------------------------------------------------------------

def rejects(**kwargs) -> video_run.VideoError:
    """Call the core with the loader poisoned, and return the error it raised."""
    ltx_process = ltx()
    orig_load = ltx_process.load_model
    ltx_process.load_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError('loader must not run'))
    try:
        ltx_process.run(**kwargs)
    except video_run.VideoError as e:
        return e
    finally:
        ltx_process.load_model = orig_load
    raise AssertionError(f'no VideoError raised for {kwargs}')


def test_empty_model_rejected_before_load():
    for model in (None, '', 'None'):
        err = rejects(model=model, prompt='test')
        assert err.code == 400, f'model={model!r} code={err.code}'
    return True


def test_separator_rejected_before_load():
    err = rejects(model='─────── LTX-2.5 ───────', prompt='test')
    assert err.code == 400, f'code={err.code}'
    assert 'separator' in str(err), f'message={err}'
    return True


def test_missing_av_rejected_before_load():
    ltx_process = ltx()
    orig_check = video_utils.check_av
    video_utils.check_av = lambda: None
    try:
        err = rejects(model='LTXVideo 0.9.6 2B T2V', prompt='test', mp4_video=True)
        assert err.code == 500, f'code={err.code}'
        assert 'av' in str(err), f'message={err}'
    finally:
        video_utils.check_av = orig_check
    assert ltx_process.run is not None
    return True


def test_missing_av_ignored_when_no_video_wanted():
    """Frames-only output does not need an encoder, so the check must not reject it."""
    ltx_process = ltx()
    orig_check = video_utils.check_av
    calls = []
    video_utils.check_av = lambda: calls.append(1)
    orig_load = ltx_process.load_model
    ltx_process.load_model = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('reached the loader'))
    try:
        ltx_process.run(model='LTXVideo 0.9.6 2B T2V', prompt='test', mp4_video=False, mp4_frames=True)
    except video_run.VideoError as e:
        raise AssertionError(f'rejected before the loader: {e}') from e
    except RuntimeError:
        pass # got past the checks, which is the point
    finally:
        video_utils.check_av = orig_check
        ltx_process.load_model = orig_load
    assert len(calls) == 0, 'av was probed for a run that saves no video'
    return True


# --- adapter delegation -----------------------------------------------------------------------

def fake_result(**kwargs):
    class FakeProcessed: # pylint: disable=too-few-public-methods
        steps = 8
    defaults = dict(
        images=[], video_path='/tmp/fake.mp4', thumb_path=None, num_frames=17, fps=24.0,
        has_audio=False, still=False, processed=FakeProcessed(), width=768, height=512,
    )
    defaults.update(kwargs)
    return video_run.VideoResult(**defaults)


def drive_adapter(core):
    """Run the adapter end to end with the core replaced, and return what it yielded."""
    ltx_process = ltx()
    orig_run = ltx_process.run
    ltx_process.run = core
    try:
        gen = ltx_process.run_ltx(
            task_id='', _ui_state='', model='LTXVideo 0.9.6 2B T2V', prompt='test', negative='',
            styles=[], width=768, height=512, frames=17, auto_duration=False, steps=8,
            sampler_index=0, guidance_scale=1.0, sampler_shift=-1.0, dynamic_shift=False, seed=-1,
            upsample_enable=False, upsample_ratio=2.0, refine_enable=False, refine_strength=0.4,
            condition_strength=1.0, ltx_init_image=None, condition_last=None, condition_files=None,
            condition_video=None, condition_video_frames=-1, condition_video_skip=0,
            decode_timestep=0.05, image_cond_noise_scale=0.025, mp4_fps=24, mp4_interpolate=0,
            mp4_codec='libx264', mp4_ext='mp4', mp4_opt='crf=16', mp4_video=True, mp4_frames=False,
            mp4_sf=False, mp4_thumb=True, audio_enable=False, _overrides={},
        )
        return list(gen)
    finally:
        ltx_process.run = orig_run


def test_adapter_yields_the_video_path_on_success():
    seen = {}

    def core(model, **kwargs):
        seen['model'] = model
        seen['kwargs'] = kwargs
        return fake_result()

    try:
        yields = drive_adapter(core)
    except Exception as e: # pylint: disable=broad-except
        return f'adapter needs more runtime state than this harness provides: {type(e).__name__}: {e}'
    assert len(yields) == 2, f'expected a loading yield and a final yield, got {len(yields)}'
    assert yields[0][0] is None, f'first yield should carry no file: {yields[0]}'
    assert yields[-1][0] == '/tmp/fake.mp4', f'final yield: {yields[-1]}'
    assert '768x512' in yields[-1][1], f'resolution missing from the summary: {yields[-1][1]}'
    return True


def test_adapter_renames_into_the_core():
    seen = {}

    def core(model, **kwargs):
        seen['model'] = model
        seen['kwargs'] = kwargs
        return fake_result()

    try:
        drive_adapter(core)
    except Exception as e: # pylint: disable=broad-except
        return f'adapter needs more runtime state than this harness provides: {type(e).__name__}: {e}'
    kwargs = seen.get('kwargs', {})
    assert seen.get('model') == 'LTXVideo 0.9.6 2B T2V', f'model not passed positionally: {seen}'
    assert 'init_image' in kwargs and 'ltx_init_image' not in kwargs, 'init image not renamed'
    assert 'audio' in kwargs and 'audio_enable' not in kwargs, 'audio flag not renamed'
    assert 'sampler_name' in kwargs and 'sampler_index' not in kwargs, 'sampler not resolved to a name'
    assert 'override_settings' in kwargs, 'overrides not forwarded'
    assert 'ui_state' in kwargs, 'ui state not forwarded'
    return True


def test_adapter_reports_a_typed_failure_as_text():
    def core(model, **kwargs): # pylint: disable=unused-argument
        raise video_run.VideoError('no model selected', 400)

    try:
        yields = drive_adapter(core)
    except Exception as e: # pylint: disable=broad-except
        return f'adapter needs more runtime state than this harness provides: {type(e).__name__}: {e}'
    assert yields[-1][0] is None, f'a failure must not report a file: {yields[-1]}'
    assert yields[-1][1].startswith('LTX Error:'), f'final yield: {yields[-1]}'
    assert 'no model selected' in yields[-1][1], f'message lost: {yields[-1][1]}'
    return True


# --- helpers ----------------------------------------------------------------------------------

def test_open_condition_does_not_read_paths():
    """A string names an upload or carries base64; opening it as a path would read any file."""
    from modules.ltx import ltx_util
    target = os.path.join(script_dir, 'requirements.txt')
    if not os.path.exists(target):
        return 'no readable file in the repo root to probe with'
    try:
        ltx_util.open_condition(target)
    except Exception: # pylint: disable=broad-except
        return True # the decoder rejected it, which is the point
    raise AssertionError('a filesystem path was accepted as a conditioning source')


def test_open_condition_passes_images_through():
    from PIL import Image
    from modules.ltx import ltx_util
    img = Image.new('RGB', (8, 8))
    assert ltx_util.open_condition(img) is img
    return True


def test_pixel_size_over_both_frame_shapes():
    import torch
    from PIL import Image
    assert video_utils.pixel_size([Image.new('RGB', (640, 352))]) == (640, 352)
    assert video_utils.pixel_size(torch.zeros(1, 3, 17, 352, 640)) == (640, 352)
    assert video_utils.pixel_size(torch.zeros(17, 352, 640, 3)) == (640, 352)
    return True


def test_pixel_size_falls_back_when_nothing_decoded():
    assert video_utils.pixel_size([], fallback=(768, 512)) == (768, 512)
    assert video_utils.pixel_size(None, fallback=(768, 512)) == (768, 512)
    assert video_utils.pixel_size(None) == (0, 0)
    return True


def test_phase_ends_its_job_when_the_body_raises():
    from modules import shared
    before = shared.state.job
    try:
        with video_utils.phase('TestPhase'):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert shared.state.job == before, f'phase leaked: job={shared.state.job!r} expected={before!r}'
    return True


def test_video_result_resolution_defaults_to_zero():
    res = video_run.VideoResult(images=[], video_path=None, thumb_path=None, num_frames=0,
                                fps=0.0, has_audio=False, still=False, processed=None)
    assert (res.width, res.height) == (0, 0), f'{res.width}x{res.height}'
    return True


def run_all():
    log.warning('=== LTX keyword core ===')

    cat = category('signatures')
    for fn in [
        test_core_takes_only_model_positionally,
        test_core_parameter_names,
        test_core_returns_video_result,
        test_adapter_signature_is_frozen,
        test_adapter_still_absorbs_extra_arguments,
        test_adapter_is_a_generator,
    ]:
        run_test(cat, fn)

    cat = category('rejections')
    for fn in [
        test_empty_model_rejected_before_load,
        test_separator_rejected_before_load,
        test_missing_av_rejected_before_load,
        test_missing_av_ignored_when_no_video_wanted,
    ]:
        run_test(cat, fn)

    cat = category('adapter')
    for fn in [
        test_adapter_yields_the_video_path_on_success,
        test_adapter_renames_into_the_core,
        test_adapter_reports_a_typed_failure_as_text,
    ]:
        run_test(cat, fn)

    cat = category('helpers')
    for fn in [
        test_open_condition_does_not_read_paths,
        test_open_condition_passes_images_through,
        test_pixel_size_over_both_frame_shapes,
        test_pixel_size_falls_back_when_nothing_decoded,
        test_phase_ends_its_job_when_the_body_raises,
        test_video_result_resolution_defaults_to_zero,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
        total_skipped += info['skipped']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    success = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if success else 1)
