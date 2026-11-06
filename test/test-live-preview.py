#!/usr/bin/env python
"""
Offline tests for the live preview's sigma correction (``State.do_set_current_image``).

The step callback publishes the latent, the model output and the step's sigmas; the preview turns them into an
estimate of the final image with the formula for the scheduler's prediction type.

Covers:

- a v-prediction scheduler: the preview of a step run after the stage's ``begin`` uses the v-prediction estimate
- an epsilon scheduler: the preview equals the scheduler's own ``pred_original_sample`` for the step
- a flow-match scheduler: the preview equals ``x_t - sigma * v`` for the step
- a pipeline that unpacks to a 5D latent: the callback publishes only its 4D middle frame, once per step
- the preview reads the published latent and model output once
- LTX 2 packed video tokens: the callback publishes the middle frame, keyframe tokens appended after the video left out

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


class UnpackPipe:
    """Unpacks to a 5D video latent of three frames, the middle one equal to the input, as Qwen-Image 2.1 does with one frame."""
    vae_scale_factor = 8
    num_timesteps = steps

    def _unpack_latents(self, latents, height, width, vae_scale_factor):  # pylint: disable=unused-argument
        return torch.stack([latents - 1, latents, latents + 1], dim=2)


def test_video_latent_published_once_as_4d():
    """Red before the fix: the callback published the unpacked 5D latent, then its middle frame, so a reader between the two writes got 5D."""
    writes = []

    class RecordingState(type(shared.state)):
        def __setattr__(self, name, value):
            if name in ('current_latent', 'current_noise_pred'):
                writes.append((name, None if value is None else tuple(value.shape)))
            super().__setattr__(name, value)

    gen = torch.Generator().manual_seed(42)
    latents, noise_pred = torch.randn(1, 4, 8, 8, generator=gen), torch.randn(1, 4, 8, 8, generator=gen)
    p = processing()
    p.width, p.height = 64, 64
    state_cls = type(shared.state)
    try:
        processing_callbacks.set_callbacks_p(p)
        shared.state.begin('Inference')
        shared.state.__class__ = RecordingState
        processing_callbacks.diffusers_callback(UnpackPipe(), step=step_index, timestep=500, kwargs={'latents': latents, 'noise_pred': noise_pred})
        shared.state.__class__ = state_cls
        published_latent, published_noise_pred = shared.state.current_latent, shared.state.current_noise_pred
        shared.state.end()
    finally:
        shared.state.__class__ = state_cls
        processing_callbacks.set_callbacks_p(None)
    assert writes == [('current_latent', (1, 4, 8, 8)), ('current_noise_pred', (1, 4, 8, 8))], f'writes {writes}'
    assert torch.equal(published_latent, latents) and torch.equal(published_noise_pred, noise_pred), 'published values are not the middle frame'
    return True


class LTXPipe:
    """The LTX 2 pipeline's packing contract, with diffusers' own pack and unpack."""
    vae_spatial_compression_ratio = 32
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1
    num_timesteps = steps

    def __init__(self):
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline
        self._unpack_latents = LTX2Pipeline._unpack_latents  # pylint: disable=protected-access
        self.pack = LTX2Pipeline._pack_latents  # pylint: disable=protected-access


def publish_packed(video, extra_tokens=None):
    """Pack a [B, C, F, H, W] latent as LTX 2 does, run the callback at the latent's pixel size, return the published latent."""
    pipe = LTXPipe()
    packed = pipe.pack(video)
    if extra_tokens is not None:
        packed = torch.cat([packed, extra_tokens], dim=1)
    p = processing()
    p.width, p.height = video.shape[-1] * pipe.vae_spatial_compression_ratio, video.shape[-2] * pipe.vae_spatial_compression_ratio
    try:
        processing_callbacks.set_callbacks_p(p)
        shared.state.begin('Inference')
        processing_callbacks.diffusers_callback(pipe, step=step_index, timestep=500, kwargs={'latents': packed})
        published = shared.state.current_latent
        shared.state.end()
    finally:
        processing_callbacks.set_callbacks_p(None)
    return published


def test_ltx_packed_latent_published_as_middle_frame():
    """Red before the fix: the packed [B, frames*h*w, C] tokens were published as they came, and the preview decoders took them for an image."""
    video = torch.randn(1, 128, 4, 5, 8, generator=torch.Generator().manual_seed(42))
    published = publish_packed(video)
    assert published is not None and published.shape == (1, 128, 5, 8), f'published {None if published is None else tuple(published.shape)}'
    assert torch.equal(published, video[:, :, 2]), 'published latent is not the middle frame'
    return True


def test_ltx_keyframe_tokens_left_out():
    """The condition pipeline appends keyframe tokens after the video tokens; the grid comes from the pass size, not from the token count."""
    gen = torch.Generator().manual_seed(42)
    video = torch.randn(1, 128, 4, 5, 8, generator=gen)
    published = publish_packed(video, extra_tokens=torch.randn(1, 5 * 8, 128, generator=gen))
    assert published is not None and published.shape == (1, 128, 5, 8), f'published {None if published is None else tuple(published.shape)}'
    assert torch.equal(published, video[:, :, 2]), 'published latent is not a video frame'
    return True


def test_preview_reads_published_values_once():
    """Red before the fix: the preview read the latent twice and the model output four times, each read a point where the callback can swap it."""
    reads = {'current_latent': 0, 'current_noise_pred': 0}

    class CountingState(type(shared.state)):
        def __getattribute__(self, name):
            if name in reads:
                reads[name] += 1
            return super().__getattribute__(name)

    decoded = []
    gen = torch.Generator().manual_seed(42)
    state_cls = type(shared.state)
    saved_decode = sd_samplers_common.samples_decode
    saved_opts = {k: shared.opts.data.get(k, None) for k in ('live_preview_force', 'show_progress_type')}
    try:
        sd_samplers_common.samples_decode = lambda latents, *args, **kwargs: latents
        shared.opts.data['live_preview_force'] = False
        shared.opts.data['show_progress_type'] = 'Approximate'
        shared.state.begin('Inference')
        shared.state.assign_current_image = decoded.append
        shared.state.current_latent, shared.state.current_noise_pred = torch.randn(1, 4, 8, 8, generator=gen), torch.randn(1, 4, 8, 8, generator=gen)
        shared.state.current_sigma, shared.state.current_sigma_next = torch.tensor(2.0), torch.tensor(1.5)
        shared.state.prediction_type = 'epsilon'
        shared.state.preview_job = -1
        shared.state.__class__ = CountingState
        shared.state.do_set_current_image()
        shared.state.__class__ = state_cls
        del shared.state.assign_current_image
        shared.state.end()
    finally:
        shared.state.__class__ = state_cls
        shared.state.__dict__.pop('assign_current_image', None)
        sd_samplers_common.samples_decode = saved_decode
        shared.opts.data.update(saved_opts)
    assert len(decoded) == 1, f'expected one preview, got {len(decoded)}'
    assert reads == {'current_latent': 1, 'current_noise_pred': 1}, f'reads {reads}'
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

    log.warning('=== Live preview publishing ===')
    cat = category('publish')
    for fn in [
        test_video_latent_published_once_as_4d,
        test_preview_reads_published_values_once,
        test_ltx_packed_latent_published_as_middle_frame,
        test_ltx_keyframe_tokens_left_out,
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
