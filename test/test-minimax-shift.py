#!/usr/bin/env python
"""
Offline unit tests for the MiniMax schedule shift in modules.video_models.video_minimax.

- ``resolve_shift`` takes a positive request value and falls back to the scheduler config otherwise
- ``set_sampler_shift`` writes scheduler, default_scheduler and audio_scheduler and keys the result for infotext
- a request without values resets what the previous request set
- the Default sampler restore, a deepcopy of default_scheduler, carries the shift into the sigma grid
- ``apply_overrides`` records the applied values on the processing object and hands the scheduler one grid point more than the step count

No running server required.

Usage:
    python test/test-minimax-shift.py
"""

import os
import sys
import copy
import types

import torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
_orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = _orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

import diffusers  # pylint: disable=wrong-import-position
from modules.errors import log  # pylint: disable=wrong-import-position
from modules.video_models import video_minimax  # pylint: disable=wrong-import-position


VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
STEPS = 5


results = {}


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
    name = fn.__name__
    try:
        ok = fn()
        record(cat, ok is not False, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'{type(e).__name__}: {e}')


# ============================================================
# Fixtures
# ============================================================

class StubVae:
    def enable_tiling(self):
        pass

    def decode(self, z, *args, **kwargs): # still mode wraps it with the latent-frame padding
        return z


class StubPipe:
    """The parts of the modular pipeline the shim touches: the scheduler pair with the shipped copy, and the canvas and frame constants."""

    canvas_multiple = 32
    vae_frames_per_chunk = 17
    vae_latents_per_chunk = 5
    max_duration = 15.0
    fps = 24
    sdnext_supported_min_frames = 120

    def __init__(self):
        self.scheduler = diffusers.MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        self.audio_scheduler = diffusers.MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        self.default_scheduler = copy.deepcopy(self.scheduler)
        self.vae = StubVae()

    @property
    def min_duration(self):
        return 5.0


def make_p():
    return types.SimpleNamespace(width=1024, height=576, steps=STEPS, frames=124, sampler_name='Default', task_args={}, extra_generation_params={})


def sigmas(shift: float):
    scheduler = diffusers.MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(STEPS)
    return scheduler.sigmas.detach().cpu()


def shifts(pipe):
    return (pipe.scheduler.shift, pipe.default_scheduler.shift, pipe.audio_scheduler.shift)


# ============================================================
# Tests
# ============================================================

def test_resolve_prefers_positive_request():
    scheduler = diffusers.MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    assert video_minimax.resolve_shift(scheduler, 6.0) == 6.0
    assert video_minimax.resolve_shift(scheduler, 0.5) == 0.5
    for absent in (None, -1.0, 0.0):
        assert video_minimax.resolve_shift(scheduler, absent) == VIDEO_SHIFT, f'requested={absent}'


def test_resolve_reads_the_shipped_value_not_the_live_one():
    scheduler = diffusers.MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    scheduler.set_shift(4.0)
    assert video_minimax.resolve_shift(scheduler, None) == VIDEO_SHIFT


def test_set_writes_every_copy_and_keys_the_result():
    pipe = StubPipe()
    applied = video_minimax.set_sampler_shift(pipe, video_shift=6.0, audio_shift=4.0)
    assert applied == {'Video shift': 6.0, 'Audio shift': 4.0}, f'applied={applied}'
    assert shifts(pipe) == (6.0, 6.0, 4.0), f'shifts={shifts(pipe)}'
    assert pipe.scheduler.config['shift'] == VIDEO_SHIFT and pipe.audio_scheduler.config['shift'] == AUDIO_SHIFT


def test_next_request_without_values_resets():
    pipe = StubPipe()
    video_minimax.set_sampler_shift(pipe, video_shift=6.0, audio_shift=4.0)
    applied = video_minimax.set_sampler_shift(pipe)
    assert applied == {'Video shift': VIDEO_SHIFT, 'Audio shift': AUDIO_SHIFT}, f'applied={applied}'
    assert shifts(pipe) == (VIDEO_SHIFT, VIDEO_SHIFT, AUDIO_SHIFT), f'shifts={shifts(pipe)}'
    applied = video_minimax.set_sampler_shift(pipe, video_shift=-1.0, audio_shift=-1.0)
    assert applied == {'Video shift': VIDEO_SHIFT, 'Audio shift': AUDIO_SHIFT}, f'applied={applied}'


def test_default_sampler_restore_carries_the_shift():
    pipe = StubPipe()
    video_minimax.set_sampler_shift(pipe, video_shift=6.0)
    pipe.scheduler = copy.deepcopy(pipe.default_scheduler) # sd_samplers.restore_default
    pipe.scheduler.set_timesteps(STEPS)
    assert torch.allclose(pipe.scheduler.sigmas.detach().cpu(), sigmas(6.0)), 'restored scheduler does not follow the requested shift'
    assert not torch.allclose(pipe.scheduler.sigmas.detach().cpu(), sigmas(VIDEO_SHIFT)), 'requested shift did not change the grid'


def test_unsupported_scheduler_is_skipped():
    pipe = StubPipe()
    pipe.scheduler = diffusers.EulerDiscreteScheduler() # no set_shift and no shift in its config
    assert video_minimax.set_sampler_shift(pipe, video_shift=6.0) == {}
    assert pipe.audio_scheduler.shift == AUDIO_SHIFT


def test_apply_overrides_records_the_applied_values():
    pipe = StubPipe()
    p = make_p()
    video_minimax.apply_overrides(p, pipe, still=False, audio=True, video_shift=6.0, audio_shift=-1.0)
    assert p.extra_generation_params == {'Video shift': 6.0, 'Audio shift': AUDIO_SHIFT}, f'recorded={p.extra_generation_params}'
    assert shifts(pipe) == (6.0, 6.0, AUDIO_SHIFT), f'shifts={shifts(pipe)}'
    assert p.sampler_name == 'Default'
    assert p.steps == STEPS and p.task_args['num_inference_steps'] == STEPS + 1 and pipe.num_timesteps == STEPS, f'steps={p.steps} grid_steps={p.task_args["num_inference_steps"]} total={pipe.num_timesteps}'


def test_apply_overrides_without_values_uses_the_shipped_schedule():
    pipe = StubPipe()
    video_minimax.set_sampler_shift(pipe, video_shift=6.0, audio_shift=4.0) # an earlier request on the same pipe
    p = make_p()
    video_minimax.apply_overrides(p, pipe, still=True, audio=False)
    assert p.extra_generation_params == {'Video shift': VIDEO_SHIFT, 'Audio shift': AUDIO_SHIFT}, f'recorded={p.extra_generation_params}'
    assert shifts(pipe) == (VIDEO_SHIFT, VIDEO_SHIFT, AUDIO_SHIFT), f'shifts={shifts(pipe)}'


# ============================================================
# Main
# ============================================================

def main():
    cat = category('resolve')
    for fn in (test_resolve_prefers_positive_request, test_resolve_reads_the_shipped_value_not_the_live_one):
        run_test(cat, fn)
    cat = category('apply')
    for fn in (test_set_writes_every_copy_and_keys_the_result, test_next_request_without_values_resets, test_default_sampler_restore_carries_the_shift, test_unsupported_scheduler_is_skipped):
        run_test(cat, fn)
    cat = category('overrides')
    for fn in (test_apply_overrides_records_the_applied_values, test_apply_overrides_without_values_uses_the_shipped_schedule):
        run_test(cat, fn)
    failed = sum(r['failed'] for r in results.values())
    passed = sum(r['passed'] for r in results.values())
    log.info(f'MiniMax shift tests: passed={passed} failed={failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
