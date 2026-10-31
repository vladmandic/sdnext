#!/usr/bin/env python
"""
Offline tests for the live preview's sigma correction (``State.do_set_current_image``).

The step callback publishes the latent, the model output and the step's sigmas; the preview turns them into an
estimate of the final image with the formula for the scheduler's prediction type.

Covers:

- a v-prediction scheduler: the preview of a step run after the stage's ``begin`` uses the v-prediction estimate
- an epsilon scheduler: the preview equals the scheduler's own ``pred_original_sample`` for the step
- a flow-match scheduler: the preview equals ``x_t - sigma * v`` for the step

Real diffusers schedulers drive the step; the pipeline is a stub and the decode returns the latent. No running server.

Usage:
    python test/test-live-preview.py
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

import torch                                                                         # pylint: disable=wrong-import-position
from diffusers import EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler        # pylint: disable=wrong-import-position
from modules import shared, devices, processing_callbacks, sd_samplers_common        # pylint: disable=wrong-import-position
from modules.errors import log                                                       # pylint: disable=wrong-import-position

devices.torch_gc = lambda *args, **kwargs: None
results: dict[str, dict] = {}
steps = 10
step_index = 3


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


class StubPipe:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.num_timesteps = steps


def processing():
    return SimpleNamespace(
        steps=steps, ip_adapter_names=[], extra_generation_params={}, correction_skip=True,
        hdr_clamp=False, hdr_mode=0, hdr_maximize=False, hdr_sharpen=0, hdr_color=0, hdr_brightness=0, hdr_tint_ratio=0,
        schedulers_sigma_adjust=1.0, schedulers_sigma_adjust_min=0.2, schedulers_sigma_adjust_max=0.8,
    )


def run_step(scheduler):
    """One real scheduler step at ``step_index``, then the callback and the preview as during generation."""
    gen = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 4, 8, 8, generator=gen)
    model_output = torch.randn(1, 4, 8, 8, generator=gen)
    scheduler.set_timesteps(steps)
    scheduler._step_index = step_index  # pylint: disable=protected-access
    out = scheduler.step(model_output, scheduler.timesteps[step_index], sample)
    decoded = []
    saved_decode = sd_samplers_common.samples_decode
    saved_assign = shared.state.assign_current_image
    saved_opts = {k: shared.opts.data.get(k, None) for k in ('live_preview_force', 'show_progress_type')}
    try:
        sd_samplers_common.samples_decode = lambda latents, *args, **kwargs: latents
        shared.state.assign_current_image = decoded.append
        shared.opts.data['live_preview_force'] = False
        shared.opts.data['show_progress_type'] = 'Approximate'
        pipe = StubPipe(scheduler)
        processing_callbacks.set_callbacks_p(processing())
        shared.state.begin('Inference')
        shared.state.preview_job = -1
        processing_callbacks.diffusers_callback(pipe, step=step_index, timestep=int(scheduler.timesteps[step_index]), kwargs={'latents': out.prev_sample, 'noise_pred': model_output})
        shared.state.do_set_current_image()
        shared.state.end()
    finally:
        sd_samplers_common.samples_decode = saved_decode
        shared.state.assign_current_image = saved_assign
        shared.opts.data.update(saved_opts)
        processing_callbacks.set_callbacks_p(None)
    assert len(decoded) == 1, f'expected one preview, got {len(decoded)}'
    sigma, sigma_next = scheduler.sigmas[step_index], scheduler.sigmas[step_index + 1]
    return SimpleNamespace(preview=decoded[0], out=out, sample=sample, model_output=model_output, sigma=sigma, sigma_next=sigma_next)


def test_v_prediction_preview_after_stage_begin():
    """Red before the fix: begin('Inference') reset the type to epsilon, so the preview used the epsilon estimate."""
    r = run_step(EulerDiscreteScheduler(prediction_type='v_prediction'))
    preview, v, sigma = r.preview, r.model_output, r.sigma
    before_step = r.out.prev_sample - v * (r.sigma_next - sigma)
    v_estimate = v * (-sigma / (sigma**2 + 1) ** 0.5) + before_step / (sigma**2 + 1)
    eps_estimate = before_step - v * sigma
    assert (v_estimate - eps_estimate).abs().mean() > 0.1, 'fixture does not separate the two estimates'
    assert torch.allclose(preview, v_estimate, atol=1e-5), f'preview is not the v-prediction estimate, mean diff {(preview - v_estimate).abs().mean():.4f}, from epsilon {(preview - eps_estimate).abs().mean():.4f}'
    return True


def test_epsilon_preview_matches_scheduler():
    r = run_step(EulerDiscreteScheduler(prediction_type='epsilon'))
    expected = r.out.pred_original_sample
    assert torch.allclose(r.preview, expected, atol=1e-5), f'mean diff {(r.preview - expected).abs().mean():.4f}'
    return True


def test_flow_preview_matches_definition():
    """The flow-match step returns no x0 estimate; x0 = x_t - sigma * v from the sample the step started at."""
    r = run_step(FlowMatchEulerDiscreteScheduler())
    expected = r.sample - r.sigma * r.model_output
    assert torch.allclose(r.preview, expected, atol=1e-5), f'mean diff {(r.preview - expected).abs().mean():.4f}'
    return True


def run_all():
    log.warning('=== Live preview sigma correction ===')
    cat = category('prediction')
    for fn in [
        test_v_prediction_preview_after_stage_begin,
        test_epsilon_preview_matches_scheduler,
        test_flow_preview_matches_definition,
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
