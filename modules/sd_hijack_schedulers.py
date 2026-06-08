from __future__ import annotations
import inspect
import torch
import diffusers
import numpy as np


_scheduled_classes = [
    "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverMultistepInverseScheduler",
    "DPMSolverSinglestepScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "SASolverScheduler",
    "UniPCMultistepScheduler",
]
_patched_schedulers = set()
_orig_unipc_set_timesteps = None


def init_hijack():
    for class_name in _scheduled_classes:
        scheduler_cls = getattr(diffusers, class_name, None)
        if scheduler_cls is None:
            continue
        _patch_scheduler_set_timesteps(scheduler_cls)


def _patch_scheduler_set_timesteps(scheduler_cls):
    if scheduler_cls in _patched_schedulers:
        return

    scheduler_cls.original_set_timesteps = scheduler_cls.set_timesteps

    # @wraps(scheduler_cls.original_set_timesteps)
    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None, mu=None, **kwargs):
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

        if timesteps is None and sigmas is None:
            return _call_original_set_timesteps(
                scheduler_cls.original_set_timesteps,
                self,
                num_inference_steps=num_inference_steps,
                device=device,
                mu=mu,
                **kwargs,
            )

        if timesteps is not None:
            if "timesteps" in set(inspect.signature(scheduler_cls.original_set_timesteps).parameters.keys()):
                return _call_original_set_timesteps(
                    scheduler_cls.original_set_timesteps,
                    self,
                    num_inference_steps=None,
                    device=device,
                    mu=mu,
                    timesteps=timesteps,
                    **kwargs,
                )
            if "sigmas" in set(inspect.signature(scheduler_cls.original_set_timesteps).parameters.keys()) and getattr(self.config, "use_flow_sigmas", False):
                sigmas_values = _invert_unipc_timesteps(self, timesteps)
                return _call_original_set_timesteps(
                    scheduler_cls.original_set_timesteps,
                    self,
                    num_inference_steps=None,
                    device=device,
                    mu=mu,
                    sigmas=sigmas_values,
                    **kwargs,
                )
            num_inference_steps, timesteps_array, sigmas_array = _prepare_custom_schedule_from_timesteps(
                self,
                timesteps,
            )
            return _assign_custom_schedule(self, num_inference_steps, timesteps_array, sigmas_array, device)

        if sigmas is not None:
            if "sigmas" in set(inspect.signature(scheduler_cls.original_set_timesteps).parameters.keys()):
                return _call_original_set_timesteps(
                    scheduler_cls.original_set_timesteps,
                    self,
                    num_inference_steps=None,
                    device=device,
                    mu=mu,
                    sigmas=sigmas,
                    **kwargs,
                )
            num_inference_steps, timesteps_array, sigmas_array = _prepare_custom_schedule_from_sigmas(self, sigmas, mu=mu)
            return _assign_custom_schedule(self, num_inference_steps, timesteps_array, sigmas_array, device)

        return None

    scheduler_cls.set_timesteps = set_timesteps
    _patched_schedulers.add(scheduler_cls)


def _call_original_set_timesteps(original, self, num_inference_steps=None, device=None, mu=None, **kwargs):
    signature = inspect.signature(original)
    call_args = {}

    if "num_inference_steps" in signature.parameters and num_inference_steps is not None:
        call_args["num_inference_steps"] = num_inference_steps
    if "device" in signature.parameters:
        call_args["device"] = device
    if "mu" in signature.parameters and mu is not None:
        call_args["mu"] = mu

    if "sigmas" in signature.parameters and "sigmas" in kwargs:
        sigmas_value = kwargs["sigmas"]
        if not isinstance(sigmas_value, (np.ndarray, torch.Tensor)):
            kwargs["sigmas"] = np.array(sigmas_value, dtype=np.float32)
    if "timesteps" in signature.parameters and "timesteps" in kwargs:
        timesteps_value = kwargs["timesteps"]
        if not isinstance(timesteps_value, (np.ndarray, torch.Tensor)):
            kwargs["timesteps"] = np.array(timesteps_value, dtype=np.int64)
    call_args.update(kwargs)
    return original(self, **call_args)


def _get_base_sigmas(scheduler) -> np.ndarray:
    if hasattr(scheduler, "alphas_cumprod"):
        alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy()
        return np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5, dtype=np.float32)
    if hasattr(scheduler, "sigmas"):
        return np.array(scheduler.sigmas.cpu().numpy(), dtype=np.float32)
    raise ValueError("Scheduler does not expose alphas_cumprod or sigmas for custom schedule conversion.")


def _get_final_sigma(scheduler) -> float:
    if getattr(scheduler.config, "final_sigmas_type", None) == "zero":
        return 0.0
    base_sigmas = _get_base_sigmas(scheduler)
    return float(base_sigmas[0])


def _sigma_to_t(sigma: np.ndarray, log_sigmas: np.ndarray) -> np.ndarray:
    sigma = np.array(sigma, dtype=np.float32)
    log_sigma = np.log(np.maximum(sigma, 1e-10))
    dists = log_sigma - log_sigmas[:, np.newaxis]
    low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
    high_idx = low_idx + 1
    low = log_sigmas[low_idx]
    high = log_sigmas[high_idx]
    w = (low - log_sigma) / (low - high)
    w = np.clip(w, 0, 1)
    t = (1 - w) * low_idx + w * high_idx
    return t.reshape(sigma.shape).astype(np.float32)


def _compute_timesteps_from_sigmas(scheduler, sigmas_array: np.ndarray) -> np.ndarray:
    base_sigmas = _get_base_sigmas(scheduler)
    log_sigmas = np.log(base_sigmas)
    if hasattr(scheduler, "_sigma_to_t"):
        sigma_to_t = scheduler._sigma_to_t # pylint: disable=protected-access
        parameters = list(inspect.signature(sigma_to_t).parameters)
        if len(parameters) == 2:
            results = []
            for sigma in sigmas_array:
                value = sigma_to_t(np.array([sigma], dtype=np.float32), log_sigmas)
                value = np.array(value, dtype=np.float32)
                results.append(float(value.reshape(-1)[0]))
            timesteps = np.array(results, dtype=np.float32)
        else:
            timesteps = np.array(
                [sigma_to_t(float(sigma)) for sigma in sigmas_array],
                dtype=np.float32,
            )
    else:
        timesteps = _sigma_to_t(sigmas_array, log_sigmas)
    return np.round(timesteps).astype(np.int64)


def _prepare_custom_schedule_from_sigmas(scheduler, sigmas, mu=None):
    sigmas_array = np.array(sigmas, dtype=np.float32)
    if sigmas_array.ndim != 1:
        raise ValueError("`sigmas` must be a 1D sequence.")
    if sigmas_array.size == 0:
        raise ValueError("`sigmas` cannot be empty.")

    final_sigma = _get_final_sigma(scheduler)
    if sigmas_array.size > 1 and np.isclose(sigmas_array[-1], final_sigma, atol=1e-6):
        sigmas_values = sigmas_array[:-1]
        appended_sigma = float(sigmas_array[-1])
    else:
        sigmas_values = sigmas_array
        appended_sigma = final_sigma

    if sigmas_values.size == 0:
        raise ValueError("`sigmas` must contain at least one non-final sigma value.")

    if getattr(scheduler.config, "use_flow_sigmas", False):
        flow_sigmas = sigmas_values.astype(np.float32)
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            if mu is None:
                raise ValueError("`mu` is required for flow sigmas when use_dynamic_shifting is enabled.")
            if not hasattr(scheduler, "time_shift"):
                raise ValueError("Scheduler does not support dynamic shifting for custom sigmas.")
            flow_sigmas = scheduler.time_shift(mu, 1.0, flow_sigmas)
        else:
            flow_shift = float(getattr(scheduler.config, "flow_shift", 1.0))
            flow_sigmas = flow_shift * flow_sigmas / (1 + (flow_shift - 1.0) * flow_sigmas)

        if getattr(scheduler.config, "shift_terminal", False) and hasattr(scheduler, "stretch_shift_to_terminal"):
            flow_sigmas = scheduler.stretch_shift_to_terminal(flow_sigmas)

        eps = 1e-6
        if np.fabs(flow_sigmas[0] - 1) < eps:
            flow_sigmas[0] -= eps

        timesteps = np.round(flow_sigmas * float(scheduler.config.num_train_timesteps)).astype(np.int64)
    elif getattr(scheduler.config, "use_karras_sigmas", False) or getattr(scheduler.config, "use_exponential_sigmas", False) or getattr(scheduler.config, "use_beta_sigmas", False) or getattr(scheduler.config, "use_lu_lambdas", False):
        raise ValueError("Custom sigmas are not supported when the scheduler uses a specialized sigma schedule configuration.")
    else:
        timesteps = _compute_timesteps_from_sigmas(scheduler, sigmas_values)

    return int(timesteps.shape[0]), timesteps, np.concatenate([sigmas_values, [appended_sigma]]).astype(np.float32)


def _prepare_custom_schedule_from_timesteps(scheduler, timesteps):
    timesteps_array = np.array(timesteps, dtype=np.float32)
    if timesteps_array.ndim != 1:
        raise ValueError("`timesteps` must be a 1D sequence.")
    if timesteps_array.size == 0:
        raise ValueError("`timesteps` cannot be empty.")

    if getattr(scheduler.config, "use_karras_sigmas", False) or getattr(scheduler.config, "use_exponential_sigmas", False) or getattr(scheduler.config, "use_beta_sigmas", False) or getattr(scheduler.config, "use_flow_sigmas", False) or getattr(scheduler.config, "use_lu_lambdas", False):
        raise ValueError("Cannot set custom timesteps when the scheduler uses a specialized sigma schedule configuration.")

    base_sigmas = _get_base_sigmas(scheduler)
    sigma_values = np.interp(timesteps_array, np.arange(base_sigmas.shape[0], dtype=np.float32), base_sigmas).astype(np.float32)
    final_sigma = _get_final_sigma(scheduler)
    return int(timesteps_array.shape[0]), timesteps_array.astype(np.int64), np.concatenate([sigma_values, [final_sigma]]).astype(np.float32)


def _invert_unipc_timesteps(scheduler, timesteps):
    timesteps_array = np.array(timesteps, dtype=np.float32)
    num_train_timesteps = float(scheduler.config.num_train_timesteps)
    transformed_sigmas = timesteps_array / num_train_timesteps
    if np.any(transformed_sigmas <= 0) or np.any(transformed_sigmas >= 1):
        raise ValueError("Custom timesteps for UniPCMultistepScheduler must be within the valid flow sigma range (0, num_train_timesteps).")
    if getattr(scheduler.config, "use_dynamic_shifting", False):
        raise ValueError("Cannot convert custom timesteps to sigmas for UniPCMultistepScheduler when use_dynamic_shifting is enabled.")
    if getattr(scheduler.config, "shift_terminal", False):
        raise ValueError("Cannot convert custom timesteps to sigmas for UniPCMultistepScheduler when shift_terminal is enabled.")
    flow_shift = float(getattr(scheduler.config, "flow_shift", 1.0))
    if flow_shift != 1.0:
        sigmas = transformed_sigmas / (flow_shift - (flow_shift - 1.0) * transformed_sigmas)
    else:
        sigmas = transformed_sigmas
    return sigmas.astype(np.float32)


def _assign_custom_schedule(scheduler, num_inference_steps, timesteps, sigmas, device):
    scheduler.timesteps = torch.from_numpy(np.array(timesteps, dtype=np.int64)).to(device=device, dtype=torch.int64)
    scheduler.sigmas = torch.from_numpy(np.array(sigmas, dtype=np.float32))
    scheduler.num_inference_steps = int(num_inference_steps)

    if hasattr(scheduler, "config") and hasattr(scheduler.config, "solver_order"):
        scheduler.model_outputs = [None] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0

    scheduler._step_index = None # pylint: disable=protected-access
    scheduler._begin_index = None # pylint: disable=protected-access
    scheduler.sigmas = scheduler.sigmas.to("cpu")


def hijack_unipc():
    global _orig_unipc_set_timesteps # pylint: disable=global-statement

    from diffusers import UniPCMultistepScheduler
    _orig_unipc_set_timesteps = UniPCMultistepScheduler.set_timesteps

    def _unipc_set_timesteps_device_fix(self, num_inference_steps=None, device=None, **kwargs):
        _orig_unipc_set_timesteps(self, num_inference_steps=num_inference_steps, device=device, **kwargs)
        if device is not None:
            self.sigmas = self.sigmas.to(device)

    UniPCMultistepScheduler.set_timesteps = _unipc_set_timesteps_device_fix


def attach_scale_noise_if_missing(sampler):
    def scale_noise(sample, timestep, noise=None):
        if noise is None:
            raise ValueError("`scale_noise` requires a `noise` tensor")

        sigmas = sampler.sigmas.to(device=sample.device, dtype=sample.dtype)
        schedule_timesteps = sampler.timesteps.to(device=sample.device, dtype=sample.dtype)

        if isinstance(timestep, torch.Tensor):
            timestep_tensor = timestep.to(device=sample.device, dtype=sample.dtype)
        else:
            timestep_tensor = torch.tensor([timestep], device=sample.device, dtype=sample.dtype)

        if timestep_tensor.ndim == 0:
            timestep_tensor = timestep_tensor.unsqueeze(0)

        if schedule_timesteps.ndim == 0:
            sigma = sigmas
        else:
            step_indices = []
            if getattr(sampler, "begin_index", None) is None:
                for t in timestep_tensor:
                    indices = (schedule_timesteps == t).nonzero(as_tuple=False)
                    if len(indices) > 1:
                        step_indices.append(indices[1].item())
                    elif len(indices) == 1:
                        step_indices.append(indices[0].item())
                    else:
                        step_indices.append(torch.argmin(torch.abs(schedule_timesteps - t)).item())
            elif getattr(sampler, "step_index", None) is not None:
                step_indices = [sampler.step_index] * timestep_tensor.shape[0]
            else:
                step_indices = [sampler.begin_index] * timestep_tensor.shape[0]

            step_indices = torch.tensor(step_indices, device=schedule_timesteps.device, dtype=torch.int64)
            sigma = sigmas[step_indices]

        while sigma.ndim < noise.ndim:
            sigma = sigma.unsqueeze(-1)

        return sigma * noise + (1.0 - sigma) * sample

    sampler.scale_noise = scale_noise
