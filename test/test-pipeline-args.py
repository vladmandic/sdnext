#!/usr/bin/env python
"""
Offline tests for the call arguments modules.processing_args builds for a pipeline.

Covers:

- a resized hires pass on a modular pipeline that takes condition images, called at the hires size
- a negative prompt the pipeline takes no input for, warned once per job, through process_images on stub pipelines
- ``get_guidance_params`` on standard and modular pipelines, with and without a guider, and ``set_guider`` building the same guider
- the FLUX.3 live-preview unpack and the standard preview callback sized for the pass that runs, hires or base

No running server required.

Usage:
    python test/test-pipeline-args.py
"""

import os
import sys
import time
import logging
import contextlib
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
from PIL import Image                                                # pylint: disable=wrong-import-position
from modules import shared, devices                                  # pylint: disable=wrong-import-position
from modules.errors import log                                       # pylint: disable=wrong-import-position
from modules.modeldata import model_data                             # pylint: disable=wrong-import-position

model_data.locked = False  # webui.py unlocks it at startup
devices.torch_gc = lambda *args, **kwargs: None  # begin/end call it; nothing here holds device memory
shared.opts.data['sd_model_checkpoint'] = 'None'  # an empty slot must not trigger a model load
shared.opts.data['sd_model_refiner'] = 'None'
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


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):  # pylint: disable=redefined-outer-name
        self.records.append(record)

    def messages(self, level: int = logging.DEBUG) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


@contextlib.contextmanager
def capture_log():
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


class NoNegativePipeline(diffusers.DiffusionPipeline):
    """A guidance-distilled pipeline: no negative prompt among its call parameters."""

    def __init__(self):
        super().__init__()
        self.register_to_config(_name_or_path='stub')
        self.sd_checkpoint_info = SimpleNamespace(name='stub', title='stub', filename='stub.safetensors', hash=None, sha256=None, shorthash=None, model_name='stub', type='diffusers')
        self.sd_model_hash = None
        self.calls: list[dict] = []

    def __call__(self, prompt=None, num_inference_steps=4, width=64, height=64, generator=None, output_type='pil', num_images_per_prompt=1, **kwargs):  # pylint: disable=unused-argument
        self.calls.append({'prompt': prompt, **kwargs})
        return SimpleNamespace(images=[Image.new('RGB', (64, 64))])


class NegativePipeline(NoNegativePipeline):
    def __call__(self, prompt=None, negative_prompt=None, num_inference_steps=4, width=64, height=64, generator=None, output_type='pil', num_images_per_prompt=1, **kwargs):  # pylint: disable=unused-argument
        self.calls.append({'prompt': prompt, 'negative_prompt': negative_prompt, **kwargs})
        return SimpleNamespace(images=[Image.new('RGB', (64, 64))])


def generate(pipe, negative_prompt: str) -> list[str]:
    """One txt2img job on a stub pipeline; the warnings it logged."""
    from modules.processing import StableDiffusionProcessingTxt2Img, process_images
    shared.sd_model = pipe
    p = StableDiffusionProcessingTxt2Img(prompt='a cat', negative_prompt=negative_prompt, steps=4, width=64, height=64, n_iter=2, batch_size=1, seed=1)
    p.scripts = None
    p.do_not_save_samples = True
    p.do_not_save_grid = True
    jobid = shared.state.begin('API-TXT', api=True)
    try:
        with capture_log() as captured:
            process_images(p)
    finally:
        shared.state.end(jobid)
    return captured.messages(logging.WARNING)


class GuidancePipeline(NoNegativePipeline):
    def __call__(self, prompt=None, negative_prompt=None, guidance_scale=4.0, guidance_rescale=0.0, num_inference_steps=4, **kwargs):  # pylint: disable=unused-argument
        return SimpleNamespace(images=[Image.new('RGB', (64, 64))])


class StubModularPipeline:
    """A modular pipeline as the accessors read it: block inputs, an optional guider and its component spec."""

    def __init__(self, inputs: list[str], guider_cls=None):
        self.blocks = SimpleNamespace(inputs=[SimpleNamespace(name=name) for name in inputs])
        self.guider = guider_cls() if guider_cls is not None else None
        self.guider_cls = guider_cls
        self.updates: list[dict] = []

    def get_component_spec(self, name: str):
        assert name == 'guider', f'component spec asked for {name}'
        return SimpleNamespace(type_hint=self.guider_cls)

    def update_components(self, **kwargs):
        self.updates.append(kwargs)


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


# ============================================================
# negative prompt
# ============================================================

def test_dropped_negative_prompt_warns_once():
    pipe = NoNegativePipeline()
    warnings = [m for m in generate(pipe, 'blurry, watermark') if 'negative prompt' in m]
    assert len(pipe.calls) == 2, f'pipeline called {len(pipe.calls)} times for two images'
    assert len(warnings) == 1 and 'NoNegativePipeline' in warnings[0], f'warnings for a dropped negative prompt over two images: {warnings}'


def test_used_or_empty_negative_prompt_is_quiet():
    """Controls: a pipeline that takes the negative prompt, and an empty negative prompt on one that does not."""
    pipe = NegativePipeline()
    warnings = [m for m in generate(pipe, 'blurry, watermark') if 'negative prompt' in m]
    assert pipe.calls and pipe.calls[0].get('negative_prompt'), f'negative prompt not passed: {pipe.calls[:1]}'
    assert not warnings, f'warned with the negative prompt passed: {warnings}'
    warnings = [m for m in generate(NoNegativePipeline(), '') if 'negative prompt' in m]
    assert not warnings, f'warned for an empty negative prompt: {warnings}'


# ============================================================
# guidance
# ============================================================

def test_guidance_params_standard_pipeline():
    from modules import processing_args
    assert processing_args.get_guidance_params(GuidancePipeline()) == {'guidance_scale', 'guidance_rescale', 'negative_prompt'}
    assert processing_args.get_guidance_params(NoNegativePipeline()) == set()


def test_guidance_params_modular_with_guider():
    """A modular pipeline's block inputs carry no guidance settings; the guider set_guider builds takes them."""
    from modules import processing_args
    model = StubModularPipeline(['prompt', 'negative_prompt', 'num_inference_steps'], diffusers.ClassifierFreeGuidance)
    got = processing_args.get_guidance_params(model)
    assert got == {'guidance_scale', 'guidance_rescale', 'start', 'stop', 'negative_prompt'}, f'default guider: {sorted(got)}'
    got = processing_args.get_guidance_params(model, 'FDG: FrequencyDecoupledGuidance')
    assert got == {'guidance_rescale', 'start', 'stop', 'negative_prompt'}, f'FDG takes guidance_scales, not guidance_scale: {sorted(got)}'


def test_guidance_params_modular_without_guider():
    from modules import processing_args
    model = StubModularPipeline(['prompt', 'num_inference_steps'])
    assert processing_args.get_guidance_params(model) == set()


def test_set_guider_builds_default_guider():
    """Guard for the shared guider_class: set_guider still builds the pipeline's own guider with the job's settings."""
    from modules import modular_guiders
    model = StubModularPipeline(['prompt'], diffusers.ClassifierFreeGuidance)
    shared.sd_model = model
    p = SimpleNamespace(cfg_name='Default', cfg_scale=4.0, cfg_image=-1.0, cfg_rescale=0.25, cfg_start=0.1, cfg_stop=0.9)
    modular_guiders.set_guider(p)
    assert len(model.updates) == 1, f'update_components calls: {model.updates}'
    guider = model.updates[0]['guider']
    assert isinstance(guider, diffusers.ClassifierFreeGuidance), f'guider {type(guider).__name__}'
    got = (guider.guidance_scale, guider.guidance_rescale, guider._start, guider._stop) # pylint: disable=protected-access
    assert got == (4.0, 0.25, 0.1, 0.9), f'guider settings {got}'


# ============================================================
# preview
# ============================================================

def test_flux3_preview_unpack_at_pass_size():
    """Packed latents carry no size; a hires pass on FLUX.3 runs at the hires size, so the preview unpack must not use the request's."""
    from modules import processing_callbacks
    from pipelines.flux3 import flux3_latents
    components = SimpleNamespace(vae_scale_factor=8)
    previous = processing_callbacks.p
    shapes = {}
    try:
        for name, p, tokens in (('hires', hires_p(width=512, height=512, hr_upscale_to_x=768, hr_upscale_to_y=768), 48 * 48), ('base', hires_p(width=512, height=512, is_hr_pass=False), 32 * 32)):
            processing_callbacks.p = p
            latents = torch.zeros(1, tokens, 128)
            shapes[name] = tuple(flux3_latents.unpack_latents(latents, components, SimpleNamespace(latents=latents)).shape)
    finally:
        processing_callbacks.p = previous
    assert shapes == {'hires': (1, 32, 96, 96), 'base': (1, 32, 64, 64)}, f'unpacked {shapes}'


def test_preview_callback_at_pass_size():
    """Guard for the shared pass_size: the standard callback unpacks FLUX.1-style packed latents at the hires size on a hires pass."""
    from modules import processing_callbacks

    class Flux1Shaped:
        vae_scale_factor = 8
        num_timesteps = 4

        def __init__(self):
            self.sizes = []

        def _unpack_latents(self, latents, height, width, vae_scale_factor):  # pylint: disable=unused-argument
            self.sizes.append((width, height))
            return latents

    previous = processing_callbacks.p
    pipe = Flux1Shaped()
    try:
        for p in (hires_p(width=512, height=512, hr_upscale_to_x=768, hr_upscale_to_y=768, cfg_stop=1.0), hires_p(width=512, height=512, is_hr_pass=False, cfg_stop=1.0)):
            processing_callbacks.p = p
            processing_callbacks.diffusers_callback(pipe, step=4, timestep=1, kwargs={'latents': torch.zeros(1, 16, 64)}) # the last step, where the correction pass is skipped
    finally:
        processing_callbacks.p = previous
    assert pipe.sizes == [(768, 768), (512, 512)], f'unpacked at {pipe.sizes}'


def run_all():
    log.warning('=== Pipeline args ===')
    cat = category('size')
    for fn in [
        test_modular_hires_pass_at_hires_size,
        test_modular_base_pass_at_request_size,
    ]:
        run_test(cat, fn)
    cat = category('negative')
    for fn in [
        test_dropped_negative_prompt_warns_once,
        test_used_or_empty_negative_prompt_is_quiet,
    ]:
        run_test(cat, fn)
    cat = category('guidance')
    for fn in [
        test_guidance_params_standard_pipeline,
        test_guidance_params_modular_with_guider,
        test_guidance_params_modular_without_guider,
        test_set_guider_builds_default_guider,
    ]:
        run_test(cat, fn)
    cat = category('preview')
    run_test(cat, test_flux3_preview_unpack_at_pass_size)
    run_test(cat, test_preview_callback_at_pass_size)

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
