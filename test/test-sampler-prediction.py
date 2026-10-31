#!/usr/bin/env python
"""
Offline tests for the prediction type a sampler preset leaves in place (``sd_samplers_diffusers.DiffusionSampler``).

A sampler whose scheduler handles several prediction types takes the model's own; a preset sets the type only for a
scheduler that handles one. Samplers are built through ``DiffusionSampler`` against a stub model whose default
scheduler declares the type, then run on an oracle that knows the clean target ``x0`` and answers each step in that
type, so a sampler that reads the model output the way the model means it ends at ``x0``.

Covers:

- Euler SGM and DC Solver following an epsilon and a v-prediction model
- DPM++ Cosine running on a v-prediction model
- controls: Euler follows the model, PeRFlow keeps its own type

No running server required.

Usage:
    python test/test-sampler-prediction.py
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

import torch                                                    # pylint: disable=wrong-import-position
from diffusers import EulerAncestralDiscreteScheduler           # pylint: disable=wrong-import-position
from modules import sd_samplers_diffusers as sdd                # pylint: disable=wrong-import-position
from modules.errors import log                                  # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
steps = 20


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


def build(name: str, model_type: str):
    """The scheduler sdnext builds for a sampler on a model whose default scheduler declares model_type.

    The default is the one sd_models gives single-file SDXL loads; a default that takes sigmas makes sdnext refuse
    samplers that do not (DC Solver, PeRFlow), which is not what this file tests.
    """
    default = EulerAncestralDiscreteScheduler(beta_schedule='scaled_linear', beta_start=0.00085, beta_end=0.012, prediction_type=model_type)
    model = SimpleNamespace(default_scheduler=default, scheduler=default)
    data = next(s for s in sdd.samplers_data_diffusers if s.name == name)
    sampler = data.constructor(model)
    assert sampler.sampler is not None, f'{name} was not built'
    return sampler.sampler


def oracle_error(scheduler, model_type: str) -> float:
    """Relative error of the final sample against x0, with the model answering exactly in model_type."""
    gen = torch.Generator().manual_seed(1)
    x0 = torch.randn(1, 4, 16, 16, generator=gen)
    noise = torch.randn(1, 4, 16, 16, generator=gen)
    scheduler.set_timesteps(steps)
    x = noise * scheduler.init_noise_sigma
    for i, t in enumerate(scheduler.timesteps):
        xin = scheduler.scale_model_input(x, t)
        s = float(scheduler.sigmas[i])
        alpha = 1.0 / (1.0 + s * s) ** 0.5
        sig = s * alpha
        eps = (xin - alpha * x0) / sig if sig > 0 else torch.zeros_like(x0)
        out = {'epsilon': eps, 'v_prediction': alpha * eps - sig * x0, 'sample': x0}[model_type]
        x = scheduler.step(out.to(x.dtype), t, x).prev_sample
    return float((x.float() - x0).abs().mean() / x0.abs().mean())


def check_follows(name: str):
    for model_type in ('epsilon', 'v_prediction'):
        scheduler = build(name, model_type)
        used = scheduler.config.prediction_type
        assert used == model_type, f'{name} on a {model_type} model uses {used}'
        err = oracle_error(scheduler, model_type)
        assert err < 0.05, f'{name} on a {model_type} model ends {err:.4f} from x0'
    return True


def test_euler_sgm_follows_model():
    """Red before: the preset forced sample, so an epsilon model's output was read as the clean image."""
    return check_follows('Euler SGM')


def test_dc_solver_follows_model():
    """Red before: the preset forced epsilon, so a v-prediction model's output was read as noise."""
    return check_follows('DC Solver')


def test_dpm_cosine_runs_on_v_prediction():
    """Red before: the preset wrote v-prediction, which the scheduler rejects on its first step."""
    scheduler = build('DPM++ Cosine', 'v_prediction')
    assert scheduler.config.prediction_type == 'v_prediction', f'uses {scheduler.config.prediction_type}'
    err = oracle_error(scheduler, 'v_prediction')
    assert err < 0.05, f'ends {err:.4f} from x0'
    return True


def test_euler_follows_model():
    return check_follows('Euler')


def test_perflow_keeps_own_type():
    scheduler = build('PeRFlow', 'v_prediction')
    assert scheduler.config.prediction_type == 'ddim_eps', f'uses {scheduler.config.prediction_type}'
    return True


def run_all():
    log.warning('=== Sampler prediction types ===')
    cat = category('presets')
    for fn in [test_euler_sgm_follows_model, test_dc_solver_follows_model, test_dpm_cosine_runs_on_v_prediction]:
        run_test(cat, fn)
    cat = category('controls')
    for fn in [test_euler_follows_model, test_perflow_keeps_own_type]:
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
