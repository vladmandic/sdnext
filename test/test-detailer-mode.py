#!/usr/bin/env python
"""
Offline unit tests for the detailer's model verdict in modules.detailer.detailer.

The detailer runs a model in one of four ways:

- ``inpaint``: set_diffuser_pipe switches it to a class registered for inpainting, and the mask reaches the pipeline.
- ``custom``: set_diffuser_pipe never switches the class (``pipe_switch_task_exclude``), so it runs as loaded.
- ``edit``: the model declares ``max_condition_images``, each crop is its condition image and the mask applies on paste.
- ``None``: not compatible.

Covers:

- ``sd_models.get_task_class`` against the class set_diffuser_pipe switches to, including the PAG reset, loader
  registrations that keep one class for every task, and registrations diffusers never resolves
- ``get_mode`` and ``is_compatible`` per loaded model, with each loader's auto-pipeline registrations and
  ``max_condition_images`` declaration applied and the registry restored afterwards
- every model without a declaration keeps the verdict the detailer reached before edit mode existed
- ``get_diffusers_task`` delegating to ``get_class_task``

Pipelines are real diffusers classes created without ``__init__``: no weights load and ``shared.sd_model`` is not touched.

No running server required.

Usage:
    python test/test-detailer-mode.py
"""

import os
import sys
from collections import namedtuple
from contextlib import contextmanager

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

import diffusers                                  # pylint: disable=wrong-import-position
from modules.errors import log                    # pylint: disable=wrong-import-position
from modules import shared                        # pylint: disable=wrong-import-position,unused-import
from modules import sd_models                     # pylint: disable=wrong-import-position
from modules.detailer import detailer             # pylint: disable=wrong-import-position


# ============================================================
# Test infrastructure
# ============================================================

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
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


# ============================================================
# Loaded models
# ============================================================

D = diffusers
T = sd_models.DiffusersTaskType
AUTO = diffusers.pipelines.auto_pipeline
MAPPINGS = (AUTO.AUTO_TEXT2IMAGE_PIPELINES_MAPPING, AUTO.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, AUTO.AUTO_INPAINT_PIPELINES_MAPPING)
PRISTINE = [list(m.items()) for m in MAPPINGS]

# Cloud pipeline outside diffusers; the verdict reads only its class name and declaration.
GoogleNanoBananaPipeline = type('GoogleNanoBananaPipeline', (), {'__call__': lambda self, prompt=None, images=None: None})

Case = namedtuple('Case', 'name cls registrations declared target mode ran_before')
CASES = [ # registrations and declarations as pipelines/model_*.py apply them; ran_before is the verdict before edit mode
    Case('Qwen-Image 2.1', D.QwenImage21Pipeline, {'qwen-image-21': (D.QwenImage21Pipeline, D.QwenImage21Pipeline, D.QwenImage21Pipeline)}, 10, 'QwenImage21Pipeline', 'edit', False),
    Case('Qwen-Image-Edit', D.QwenImageEditPipeline, {'qwen-image': (D.QwenImageEditPipeline, D.QwenImageEditPipeline, D.QwenImageEditPipeline)}, 1, 'QwenImageEditPipeline', 'edit', False),
    Case('Qwen-Image-Edit Plus', D.QwenImageEditPlusPipeline, {'qwen-image': (D.QwenImageEditPlusPipeline, D.QwenImageEditPlusPipeline, D.QwenImageEditPlusPipeline)}, 4, 'QwenImageEditPlusPipeline', 'edit', False),
    Case('Qwen-Image', D.QwenImagePipeline, {'qwen-image': (D.QwenImagePipeline, D.QwenImageImg2ImgPipeline, D.QwenImageInpaintPipeline)}, None, 'QwenImageInpaintPipeline', 'inpaint', True),
    Case('Qwen-Image Layered', D.QwenImageLayeredPipeline, {'qwen-layered': (D.QwenImageLayeredPipeline, D.QwenImageLayeredPipeline, D.QwenImageLayeredPipeline)}, None, 'QwenImageLayeredPipeline', None, False),
    Case('FLUX.2 dev', D.Flux2Pipeline, {'flux2': (D.Flux2Pipeline, D.Flux2Pipeline, D.Flux2Pipeline)}, 10, 'Flux2Pipeline', 'edit', False),
    Case('FLUX.2 Klein', D.Flux2KleinPipeline, {'flux2klein': (D.Flux2KleinPipeline, D.Flux2KleinPipeline, D.Flux2KleinPipeline)}, 4, 'Flux2KleinPipeline', 'edit', False),
    Case('FLUX.1 Kontext', D.FluxKontextPipeline, {'flux1kontext': (D.FluxKontextPipeline, D.FluxKontextPipeline, D.FluxKontextInpaintPipeline)}, 1, 'FluxKontextPipeline', 'edit', False),
    Case('FLUX.1', D.FluxPipeline, {}, None, 'FluxInpaintPipeline', 'inpaint', True),
    Case('JoyImage Edit Plus', D.JoyImageEditPlusPipeline, {'joy-image-edit': (D.JoyImageEditPlusPipeline, D.JoyImageEditPlusPipeline, None)}, 4, 'JoyImageEditPlusPipeline', 'edit', False),
    Case('GLM-Image', D.GlmImagePipeline, {}, 10, 'GlmImagePipeline', 'edit', False),
    Case('Chroma', D.ChromaPipeline, {'chroma': (D.ChromaPipeline, D.ChromaImg2ImgPipeline, D.ChromaInpaintPipeline)}, None, 'ChromaInpaintPipeline', 'inpaint', True),
    Case('SDXL', D.StableDiffusionXLPipeline, {}, None, 'StableDiffusionXLInpaintPipeline', 'inpaint', True),
    Case('SDXL PAG', D.StableDiffusionXLPAGPipeline, {}, None, 'StableDiffusionXLInpaintPipeline', 'inpaint', True),
    Case('AuraFlow', D.AuraFlowPipeline, {}, None, 'AuraFlowPipeline', 'custom', True),
    Case('Nano Banana', GoogleNanoBananaPipeline, {}, 14, 'GoogleNanoBananaPipeline', 'custom', True),
    Case('Kolors', D.KolorsPipeline, {}, None, 'KolorsPipeline', None, False),
]


def restore_registry():
    for mapping, items in zip(MAPPINGS, PRISTINE):
        mapping.clear()
        mapping.update(items)


@contextmanager
def loaded(case: Case):
    """The pipeline as its loader leaves it: registrations applied and declaration set; the registry is restored on exit."""
    try:
        for key, classes in case.registrations.items():
            for mapping, cls in zip(MAPPINGS, classes):
                if cls is not None:
                    mapping[key] = cls
        pipe = object.__new__(case.cls)
        if case.declared is not None:
            pipe.max_condition_images = case.declared
        yield pipe
    finally:
        restore_registry()


def check_cases(fn):
    failures = []
    for case in CASES:
        with loaded(case) as pipe:
            message = fn(case, pipe)
        if message:
            failures.append(f'{case.name}: {message}')
    assert not failures, '; '.join(failures)


def case(name: str) -> Case:
    return next(c for c in CASES if c.name == name)


# ============================================================
# get_task_class
# ============================================================

def test_inpaint_task_class_per_model():
    def check(c, pipe):
        target = sd_models.get_task_class(pipe, T.INPAINTING).__name__
        return None if target == c.target else f'expected {c.target}, got {target}'
    check_cases(check)


def test_registered_class_kept_for_every_task():
    with loaded(case('Qwen-Image 2.1')) as pipe:
        for task in (T.TEXT_2_IMAGE, T.IMAGE_2_IMAGE, T.INPAINTING):
            assert sd_models.get_task_class(pipe, task) is D.QwenImage21Pipeline, task


def test_pag_resets_to_parent_before_switching():
    with loaded(case('SDXL PAG')) as pipe:
        assert sd_models.get_task_class(pipe, T.IMAGE_2_IMAGE) is D.StableDiffusionXLImg2ImgPipeline
    pipe = object.__new__(D.StableDiffusionPAGPipeline)
    assert sd_models.get_task_class(pipe, T.INPAINTING) is D.StableDiffusionInpaintPipeline


def test_switch_back_to_text2image():
    with loaded(case('Chroma')):
        assert sd_models.get_task_class(object.__new__(D.ChromaImg2ImgPipeline), T.TEXT_2_IMAGE) is D.ChromaPipeline
    with loaded(case('Qwen-Image')):
        assert sd_models.get_task_class(object.__new__(D.QwenImageInpaintPipeline), T.TEXT_2_IMAGE) is D.QwenImagePipeline


def test_unresolved_registration_keeps_own_class():
    with loaded(case('FLUX.1 Kontext')) as pipe: # diffusers resolves Kontext to its own key, which has no inpaint entry
        assert AUTO._get_model('FluxKontextPipeline') == 'flux-kontext' # pylint: disable=protected-access
        assert sd_models.get_task_class(pipe, T.INPAINTING) is D.FluxKontextPipeline


def test_excluded_and_modular_keep_own_class():
    with loaded(case('AuraFlow')) as pipe:
        assert sd_models.get_task_class(pipe, T.INPAINTING) is D.AuraFlowPipeline
    modular = type('FakeModularPipeline', (), {})
    assert sd_models.get_task_class(object.__new__(modular), T.INPAINTING) is modular


def test_diffusers_task_delegates_to_class_task():
    def check(c, pipe):
        return None if sd_models.get_diffusers_task(pipe) == sd_models.get_class_task(c.cls) else 'differs'
    check_cases(check)


# ============================================================
# get_mode / is_compatible
# ============================================================

def test_mode_per_model():
    def check(c, pipe):
        mode = detailer.get_mode(pipe)
        return None if mode == c.mode else f'expected {c.mode}, got {mode}'
    check_cases(check)


def test_undeclared_models_keep_their_verdict():
    def check(c, pipe):
        if c.declared is not None:
            return None
        compatible = detailer.is_compatible(pipe)
        return None if compatible == c.ran_before else f'compatible={compatible}, before={c.ran_before}'
    check_cases(check)


def test_declared_models_are_compatible():
    def check(c, pipe):
        if c.declared is None:
            return None
        return None if detailer.is_compatible(pipe) else 'not compatible'
    check_cases(check)


def test_excluded_class_stays_custom_when_declared():
    with loaded(case('Nano Banana')) as pipe:
        assert sd_models.get_max_condition_images(pipe) == 14
        assert detailer.get_mode(pipe) == 'custom'


def test_registry_restored_after_cases():
    for c in CASES:
        with loaded(c):
            pass
    for mapping, items in zip(MAPPINGS, PRISTINE):
        assert list(mapping.items()) == items, 'auto-pipeline registry changed'


# ============================================================
# Runner
# ============================================================

def run_all():
    log.warning('=== get_task_class ===')
    cat = category('task_class')
    for fn in [
        test_inpaint_task_class_per_model,
        test_registered_class_kept_for_every_task,
        test_pag_resets_to_parent_before_switching,
        test_switch_back_to_text2image,
        test_unresolved_registration_keeps_own_class,
        test_excluded_and_modular_keep_own_class,
        test_diffusers_task_delegates_to_class_task,
    ]:
        run_test(cat, fn)

    log.warning('=== get_mode ===')
    cat = category('mode')
    for fn in [
        test_mode_per_model,
        test_undeclared_models_keep_their_verdict,
        test_declared_models_are_compatible,
        test_excluded_class_stays_custom_when_declared,
        test_registry_restored_after_cases,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
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
