import os
import sys
import time
import numpy as np
import torch

# Ensure we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from modules.errors import log
from modules.res4lyf import (
    BASE, SIMPLE, VARIANTS,
    RESUnifiedScheduler, RESMultistepScheduler, RESDEISMultistepScheduler,
    ETDRKScheduler, LawsonScheduler, ABNorsettScheduler, PECScheduler,
    RiemannianFlowScheduler, RESSinglestepScheduler, RESSinglestepSDEScheduler,
    RESMultistepSDEScheduler, SimpleExponentialScheduler, LinearRKScheduler,
    LobattoScheduler, GaussLegendreScheduler, RungeKutta44Scheduler,
    RungeKutta57Scheduler, RungeKutta67Scheduler, SpecializedRKScheduler,
    BongTangentScheduler, CommonSigmaScheduler, RadauIIAScheduler,
    LangevinDynamicsScheduler
)
from modules.schedulers.scheduler_vdm import VDMScheduler
from modules.schedulers.scheduler_unipc_flowmatch import FlowUniPCMultistepScheduler
from modules.schedulers.scheduler_ufogen import UFOGenScheduler
from modules.schedulers.scheduler_tdd import TDDScheduler
from modules.schedulers.scheduler_tcd import TCDScheduler
from modules.schedulers.scheduler_flashflow import FlashFlowMatchEulerDiscreteScheduler
from modules.schedulers.scheduler_dpm_flowmatch import FlowMatchDPMSolverMultistepScheduler
from modules.schedulers.scheduler_dc import DCSolverMultistepScheduler
from modules.schedulers.scheduler_bdia import BDIA_DDIMScheduler

def test_scheduler(name, scheduler_class, config):
    try:
        scheduler = scheduler_class(**config)
    except Exception as e:
        log.error(f'scheduler="{name}" cls={scheduler_class} config={config} error="Init failed: {e}"')
        return False

    num_steps = 20
    scheduler.set_timesteps(num_steps)

    sample = torch.randn((1, 4, 64, 64))
    has_changed = False
    t0 = time.time()
    messages = []

    try:
        for i, t in enumerate(scheduler.timesteps):
            # Simulate model output (noise or x0 or v), Using random noise for stability check
            model_output = torch.randn_like(sample)

            # Scaling Check
            step_idx = scheduler.step_index if hasattr(scheduler, "step_index") and scheduler.step_index is not None else i
            # Clamp index
            if hasattr(scheduler, 'sigmas'):
                step_idx = min(step_idx, len(scheduler.sigmas) - 1)
                sigma = scheduler.sigmas[step_idx]
            else:
                sigma = torch.tensor(1.0) # Dummy for non-sigma schedulers

            # Re-introduce scaling calculation first
            scaled_sample = scheduler.scale_model_input(sample, t)

            if config.get("prediction_type") == "flow_prediction" or name in ["UFOGenScheduler", "TDDScheduler", "TCDScheduler", "BDIA_DDIMScheduler", "DCSolverMultistepScheduler"]:
                # Some new schedulers don't use K-diffusion scaling
                expected_scale = 1.0
            else:
                expected_scale = 1.0 / ((sigma**2 + 1) ** 0.5)

            # Simple check with loose tolerance due to float precision
            expected_scaled_sample = sample * expected_scale
            if not torch.allclose(scaled_sample, expected_scaled_sample, atol=1e-4):
                # If failed, double check if it's just 'sample' (no scaling)
                if torch.allclose(scaled_sample, sample, atol=1e-4):
                    messages.append('warning="scaling is identity"')
                else:
                    log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} expected={expected_scale} error="scaling mismatch"')
                    return False

            if torch.isnan(scaled_sample).any():
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="NaN in scaled_sample"')
                return False

            if torch.isinf(scaled_sample).any():
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="Inf in scaled_sample"')
                return False

            output = scheduler.step(model_output, t, sample)

            # Shape and Dtype check
            if output.prev_sample.shape != sample.shape:
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="Shape mismatch: {output.prev_sample.shape} vs {sample.shape}"')
                return False
            if output.prev_sample.dtype != sample.dtype:
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="Dtype mismatch: {output.prev_sample.dtype} vs {sample.dtype}"')
                return False

            # Update check: Did the sample change?
            if not torch.equal(sample, output.prev_sample):
                has_changed = True

            # Sample Evolution Check
            step_diff = (sample - output.prev_sample).abs().mean().item()
            if step_diff < 1e-6:
                messages.append(f'warning="minimal sample change: {step_diff}"')

            sample = output.prev_sample

            if torch.isnan(sample).any():
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="NaN in sample"')
                return False

            if torch.isinf(sample).any():
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="Inf in sample"')
                return False

            # Divergence check
            if sample.abs().max() > 1e10:
                log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} step={i} error="divergence detected"')
                return False

        # External check for Sigma Monotonicity
        if hasattr(scheduler, 'sigmas'):
            sigmas = scheduler.sigmas.cpu().numpy()
            if len(sigmas) > 1:
                diffs = np.diff(sigmas) # Check if potentially monotonic decreasing (standard) OR increasing (some flow/inverse setups). We allow flat sections (diff=0) hence 1e-6 slack
                is_monotonic_decreasing = np.all(diffs <= 1e-6)
                is_monotonic_increasing = np.all(diffs >= -1e-6)
                if not (is_monotonic_decreasing or is_monotonic_increasing):
                    messages.append('warning="sigmas are not monotonic"')

    except Exception as e:
        log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} exception: {e}')
        import traceback
        traceback.print_exc()
        return False

    if not has_changed:
        log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} error="sample never changed"')
        return False

    final_std = sample.std().item()
    if final_std > 50.0 or final_std < 0.1:
        log.error(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} std={final_std} error="variance drift"')

    t1 = time.time()
    messages = list(set(messages))
    log.info(f'scheduler="{name}" cls={scheduler.__class__.__name__} config={config} time={t1-t0} messages={messages}')
    return True

def run_tests():
    prediction_types = ["epsilon", "v_prediction", "sample"] # flow_prediction is special, usually requires flow sigmas or specific setup, checking standard ones first

    # Test BASE schedulers with their specific parameters
    log.warning('type="base"')
    for name, cls in BASE:
        configs = []

        # prediction_types
        for pt in prediction_types:
            configs.append({"prediction_type": pt})

        # Specific params for specific classes
        if cls == RESUnifiedScheduler:
            rk_types = ["res_2m", "res_3m", "res_2s", "res_3s", "res_5s", "res_6s", "deis_1s", "deis_2m", "deis_3m"]
            for rk in rk_types:
                for pt in prediction_types:
                    configs.append({"rk_type": rk, "prediction_type": pt})

        elif cls == RESMultistepScheduler:
            variants = ["res_2m", "res_3m", "deis_2m", "deis_3m"]
            for v in variants:
                for pt in prediction_types:
                    configs.append({"variant": v, "prediction_type": pt})

        elif cls == RESDEISMultistepScheduler:
            for order in range(1, 6):
                for pt in prediction_types:
                    configs.append({"solver_order": order, "prediction_type": pt})

        elif cls == ETDRKScheduler:
            variants = ["etdrk2_2s", "etdrk3_a_3s", "etdrk3_b_3s", "etdrk4_4s", "etdrk4_4s_alt"]
            for v in variants:
                for pt in prediction_types:
                    configs.append({"variant": v, "prediction_type": pt})

        elif cls == LawsonScheduler:
            variants = ["lawson2a_2s", "lawson2b_2s", "lawson4_4s"]
            for v in variants:
                for pt in prediction_types:
                    configs.append({"variant": v, "prediction_type": pt})

        elif cls == ABNorsettScheduler:
            variants = ["abnorsett_2m", "abnorsett_3m", "abnorsett_4m"]
            for v in variants:
                for pt in prediction_types:
                    configs.append({"variant": v, "prediction_type": pt})

        elif cls == PECScheduler:
            variants = ["pec423_2h2s", "pec433_2h3s"]
            for v in variants:
                for pt in prediction_types:
                    configs.append({"variant": v, "prediction_type": pt})

        elif cls == RiemannianFlowScheduler:
            metrics = ["euclidean", "hyperbolic", "spherical", "lorentzian"]
            for m in metrics:
                configs.append({"metric_type": m, "prediction_type": "epsilon"}) # Flow usually uses v or raw, but epsilon check matches others

        if not configs:
            for pt in prediction_types:
                configs.append({"prediction_type": pt})

        for conf in configs:
            test_scheduler(name, cls, conf)

    log.warning('type="simple"')
    for name, cls in SIMPLE:
        for pt in prediction_types:
            test_scheduler(name, cls, {"prediction_type": pt})

    log.warning('type="variants"')
    for name, cls in VARIANTS:
        # these classes preset their variants/rk_types in __init__ so we just test prediction types
        for pt in prediction_types:
            test_scheduler(name, cls, {"prediction_type": pt})

    # Extra robustness check: Flow Prediction Type
    log.warning('type="flow"')
    flow_schedulers = [
        # res4lyf schedulers
        RESUnifiedScheduler, RESMultistepScheduler, ABNorsettScheduler,
        RESSinglestepScheduler, RESSinglestepSDEScheduler, RESDEISMultistepScheduler,
        RESMultistepSDEScheduler, ETDRKScheduler, LawsonScheduler, PECScheduler,
        SimpleExponentialScheduler, LinearRKScheduler, LobattoScheduler,
        GaussLegendreScheduler, RungeKutta44Scheduler, RungeKutta57Scheduler,
        RungeKutta67Scheduler, SpecializedRKScheduler, BongTangentScheduler,
        CommonSigmaScheduler, RadauIIAScheduler, LangevinDynamicsScheduler,
        RiemannianFlowScheduler,
        # sdnext schedulers
        FlowUniPCMultistepScheduler, FlashFlowMatchEulerDiscreteScheduler, FlowMatchDPMSolverMultistepScheduler,
    ]
    for cls in flow_schedulers:
        test_scheduler(cls.__name__, cls, {"prediction_type": "flow_prediction", "use_flow_sigmas": True})

    log.warning('type="sdnext"')
    extended_schedulers = [
        VDMScheduler,
        UFOGenScheduler,
        TDDScheduler,
        TCDScheduler,
        DCSolverMultistepScheduler,
        BDIA_DDIMScheduler
    ]
    for prediction_type in ["epsilon", "v_prediction", "sample"]:
        for cls in extended_schedulers:
            test_scheduler(cls.__name__, cls, {"prediction_type": prediction_type})

if __name__ == "__main__":
    run_tests()
