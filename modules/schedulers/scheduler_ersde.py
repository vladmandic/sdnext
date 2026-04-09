# Copyright 2024 Qinpeng Cui and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ER-SDE Solver: Extended Reverse-time SDE for diffusion sampling
# Based on https://github.com/QinpengCui/ER-SDE-Solver (WACV 2025)
# Ported to diffusers-compatible scheduler for SD.Next

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class ERSDESchedulerOutput(BaseOutput):
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999, alpha_transform_type="cosine"):
    if alpha_transform_type == "cosine":
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


def customized_func(x, func_type=7):
    """Noise scaling function for ER-SDE. Controls the stochasticity of sampling.
    func_type=1 gives ODE (deterministic), func_type=7 is the paper's recommended default."""
    if func_type == 1:  # ODE
        return x
    elif func_type == 2:  # Original SDE
        return x ** 2
    elif func_type == 3:  # SDE_1
        eta = 0.0
        return x * (torch.exp(x ** (eta - 1) / (eta - 1)) + 10)
    elif func_type == 4:  # SDE_2
        return x ** 1.5
    elif func_type == 5:  # SDE_3
        return x ** 2.5
    elif func_type == 6:  # SDE_4
        return x ** 0.9 * torch.log10(1 + 100 * x ** 1.5)
    elif func_type == 7:  # SDE_5 (default)
        return x * (torch.exp(x ** 0.3) + 10)
    else:
        raise ValueError(f"func_type must be 1-7, got {func_type}")


class ERSDEScheduler(SchedulerMixin, ConfigMixin):
    """ER-SDE (Extended Reverse-time SDE) Solver for diffusion models.

    Supports VP-type models (SD 1.5, SDXL) and flow-matching models (SD3, Flux)
    with 1st, 2nd, and 3rd order Taylor expansion methods.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        solver_order: int = 1,
        func_type: int = 7,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        rescale_betas_zero_snr: bool = False,
        set_alpha_to_one: bool = True,
        lower_order_final: bool = True,
        use_karras_sigmas: bool = False,
        num_integration_steps: int = 100,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        if solver_order not in (1, 2, 3):
            raise ValueError(f"solver_order must be 1, 2, or 3, got {solver_order}")
        if func_type < 1 or func_type > 7:
            raise ValueError(f"func_type must be 1-7, got {func_type}")

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.init_noise_sigma = 1.0
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

        # ER-SDE state
        self._step_index = None
        self._begin_index = None
        self._is_flow = False
        self.old_x0 = None
        self.old_d_x0 = None
        self.prev_lambda = None
        self.prev_prev_lambda = None
        self.lower_order_nums = 0
        # Flow-matching arrays (populated in set_timesteps for flow mode)
        self.sigmas = None
        self._flow_alphas = None
        self._flow_sigmas = None
        self._flow_lambdas = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index=0):
        self._begin_index = begin_index

    def _init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        index_candidates = (self.timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        self._step_index = step_index

    def _reset_state(self):
        self._step_index = None
        self._begin_index = None
        self.old_x0 = None
        self.old_d_x0 = None
        self.prev_lambda = None
        self.prev_prev_lambda = None
        self.lower_order_nums = 0

    def _setup_flow(self, sigmas, device, mu=None):
        """Common flow-matching schedule setup from a sigmas tensor."""
        self._is_flow = True
        if self.config.use_dynamic_shifting:
            if mu is None:
                raise ValueError("mu must be provided when use_dynamic_shifting=True")
            sigmas = self._time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        flow_sigmas = sigmas.clamp(min=1e-8, max=1.0 - 1e-8)
        flow_alphas = 1.0 - flow_sigmas
        flow_lambdas = flow_sigmas / flow_alphas
        self.timesteps = (sigmas * self.config.num_train_timesteps).to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device, dtype=sigmas.dtype)])
        self._flow_alphas = torch.cat([flow_alphas, torch.ones(1, device=device, dtype=torch.float64)])
        self._flow_sigmas = torch.cat([flow_sigmas, torch.zeros(1, device=device, dtype=torch.float64)])
        self._flow_lambdas = torch.cat([flow_lambdas, torch.zeros(1, device=device, dtype=torch.float64)])

    def set_timesteps(self, num_inference_steps: Optional[int] = None, device: Union[str, torch.device] = None, timesteps: Optional[List[int]] = None, sigmas: Optional[List[float]] = None, mu: Optional[float] = None):
        if sigmas is not None:
            # Flow-matching path: sigmas provided externally
            sigmas = np.array(sigmas, dtype=np.float64) if not isinstance(sigmas, np.ndarray) else sigmas.astype(np.float64)
            self.num_inference_steps = len(sigmas)
            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float64, device=device)
            self._setup_flow(sigmas, device, mu)
        elif self.config.prediction_type == "flow_prediction":
            # Flow-matching path: compute flow schedule internally
            self.num_inference_steps = num_inference_steps
            sigmas = np.linspace(1.0, 1.0 / self.config.num_train_timesteps, num_inference_steps)
            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float64, device=device)
            self._setup_flow(sigmas, device, mu)
        elif timesteps is not None:
            # Custom timesteps path
            self._is_flow = False
            self.num_inference_steps = len(timesteps)
            timesteps = np.array(timesteps, dtype=np.int64)
            self.timesteps = torch.from_numpy(timesteps).to(device)
            sigmas_arr = []
            for t in timesteps:
                acp = self.alphas_cumprod[t].item()
                sigmas_arr.append(((1 - acp) / max(acp, 1e-8)) ** 0.5)
            sigmas_arr.append(0.0)
            self.sigmas = torch.tensor(sigmas_arr, dtype=torch.float64, device=device)
            self._flow_alphas = None
            self._flow_sigmas = None
            self._flow_lambdas = None
        else:
            # VP path: standard diffusion models
            self._is_flow = False
            self.num_inference_steps = num_inference_steps

            if self.config.timestep_spacing == "linspace":
                timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // num_inference_steps
                timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / num_inference_steps
                timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(f"{self.config.timestep_spacing} is not supported")

            self.timesteps = torch.from_numpy(timesteps).to(device)
            sigmas_arr = []
            for t in timesteps:
                acp = self.alphas_cumprod[t].item()
                sigmas_arr.append(((1 - acp) / max(acp, 1e-8)) ** 0.5)
            sigmas_arr.append(0.0)
            self.sigmas = torch.tensor(sigmas_arr, dtype=torch.float64, device=device)
            self._flow_alphas = None
            self._flow_sigmas = None
            self._flow_lambdas = None

        self._reset_state()

    def _time_shift(self, mu, sigma, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _get_alpha_sigma_lambda(self, step_idx):
        """Get alpha, sigma, lambda values for a given step index."""
        if self._is_flow:
            alpha = self._flow_alphas[step_idx]
            sigma = self._flow_sigmas[step_idx]
            lam = self._flow_lambdas[step_idx]
        else:
            t = self.timesteps[step_idx] if step_idx < len(self.timesteps) else 0
            if step_idx >= len(self.timesteps):
                # Past the last timestep: fully denoised
                alpha = torch.tensor(1.0, dtype=torch.float64)
                sigma = torch.tensor(0.0, dtype=torch.float64)
                lam = torch.tensor(0.0, dtype=torch.float64)
            else:
                acp = self.alphas_cumprod[t].to(torch.float64)
                alpha = acp.sqrt()
                sigma = (1.0 - acp).sqrt()
                lam = sigma / alpha.clamp(min=1e-8)
        return alpha, sigma, lam

    def _numerical_clip(self, x, eps=1e-6):
        """Clip near-zero values to prevent negative variance from float errors."""
        if isinstance(x, torch.Tensor):
            return torch.where(torch.abs(x) < eps, torch.zeros_like(x), x)
        return 0.0 if abs(x) < eps else x

    @staticmethod
    def _safe_div(numerator, denominator, eps=1e-10):
        """Division with sign-preserving zero protection for lambda differences."""
        if denominator.abs() < eps:
            return torch.zeros_like(numerator)
        return numerator / denominator

    def _compute_fn(self, x):
        """Evaluate the noise scaling function on a scalar or tensor."""
        if isinstance(x, (int, float)):
            x = torch.tensor(x, dtype=torch.float64)
        return customized_func(x, self.config.func_type)

    def _compute_integral(self, lambda_next, lambda_curr):
        """Numerical integration of 1/fn(lambda) over [lambda_next, lambda_curr] for 2nd-order correction."""
        N = self.config.num_integration_steps
        delta = (lambda_curr - lambda_next)
        if abs(delta.item()) < 1e-10:
            return torch.tensor(0.0, dtype=torch.float64, device=lambda_curr.device)
        indices = lambda_next + torch.arange(1, N + 1, dtype=torch.float64, device=lambda_curr.device) / N * delta
        fn_vals = torch.stack([self._compute_fn(idx) for idx in indices])
        fn_vals = fn_vals.clamp(min=1e-10)
        s_int = (1.0 / fn_vals * delta / N).sum()
        return s_int

    def _compute_derivative_integral(self, lambda_next, lambda_curr):
        """Numerical integration of (lambda - lambda_curr)/fn(lambda) over [lambda_next, lambda_curr] for 3rd-order correction."""
        N = self.config.num_integration_steps
        delta = (lambda_curr - lambda_next)
        if abs(delta.item()) < 1e-10:
            return torch.tensor(0.0, dtype=torch.float64, device=lambda_curr.device)
        indices = lambda_next + torch.arange(1, N + 1, dtype=torch.float64, device=lambda_curr.device) / N * delta
        fn_vals = torch.stack([self._compute_fn(idx) for idx in indices])
        fn_vals = fn_vals.clamp(min=1e-10)
        s_d_int = ((indices - lambda_curr) / fn_vals * delta / N).sum()
        return s_d_int

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)
        return sample

    def _predict_x0(self, model_output, sample, alpha, sigma):
        """Convert model output to x0 prediction based on prediction_type."""
        prediction_type = self.config.prediction_type
        if prediction_type == "epsilon":
            x0 = (sample - sigma * model_output) / alpha.clamp(min=1e-8)
        elif prediction_type == "sample":
            x0 = model_output
        elif prediction_type == "v_prediction":
            x0 = alpha * sample - sigma * model_output
        elif prediction_type == "flow_prediction":
            x0 = sample - sigma * model_output
        else:
            raise ValueError(f"prediction_type {prediction_type} must be one of epsilon, sample, v_prediction, flow_prediction")
        return x0

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    def scale_noise(self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor], noise: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        """Forward process for flow-matching models (img2img noise addition)."""
        if self.sigmas is None:
            return sample
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)
        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)
        if self._begin_index is None:
            step_indices = [self._index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self._step_index is not None:
            step_indices = [self._step_index] * timestep.shape[0]
        else:
            step_indices = [self._begin_index] * timestep.shape[0]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)
        sample = sigma * noise + (1.0 - sigma) * sample
        return sample

    def _index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        index_candidates = (schedule_timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            return len(schedule_timesteps) - 1
        return index_candidates[0].item()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[ERSDESchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is None, run set_timesteps first")

        if self._step_index is None:
            self._init_step_index(timestep)

        dtype = sample.dtype
        device = sample.device

        # Get alpha, sigma, lambda for current and next step
        alpha_curr, sigma_curr, lambda_curr = self._get_alpha_sigma_lambda(self._step_index)
        alpha_next, sigma_next, lambda_next = self._get_alpha_sigma_lambda(self._step_index + 1)

        alpha_curr = alpha_curr.to(device)
        sigma_curr = sigma_curr.to(device)
        lambda_curr = lambda_curr.to(device)
        alpha_next = alpha_next.to(device)
        sigma_next = sigma_next.to(device)
        lambda_next = lambda_next.to(device)

        # Predict x0 from model output
        x0 = self._predict_x0(model_output.to(torch.float64), sample.to(torch.float64), alpha_curr, sigma_curr)

        # Apply thresholding or clipping
        if self.config.thresholding:
            x0 = self._threshold_sample(x0)
        elif self.config.clip_sample:
            x0 = x0.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        # Determine effective solver order for this step
        max_order = self.config.solver_order
        effective_order = min(max_order, self.lower_order_nums + 1)

        # Lower order for final step
        is_last_step = (self._step_index == len(self.timesteps) - 1)
        if self.config.lower_order_final and is_last_step:
            effective_order = 1

        # When next sigma is 0 (fully denoised), use 1st order and skip noise
        at_final_sigma = sigma_next.abs() < 1e-8
        if at_final_sigma:
            effective_order = 1

        # Compute noise scaling function ratio
        fn_lambda_curr = self._compute_fn(lambda_curr)
        fn_lambda_next = self._compute_fn(lambda_next)
        # Avoid division by zero
        if fn_lambda_curr.abs() < 1e-10:
            r_fn = torch.tensor(1.0, dtype=torch.float64, device=device)
        else:
            r_fn = fn_lambda_next / fn_lambda_curr

        r_alpha = alpha_next / alpha_curr.clamp(min=1e-8)

        # Compute stochastic noise standard deviation
        noise_var = self._numerical_clip(lambda_next ** 2 - lambda_curr ** 2 * r_fn ** 2)
        if isinstance(noise_var, torch.Tensor):
            noise_std = alpha_next * torch.sqrt(noise_var.clamp(min=0))
        else:
            noise_std = alpha_next * math.sqrt(max(noise_var, 0))

        # Generate noise
        if at_final_sigma or noise_std.abs() < 1e-10:
            noise_term = torch.zeros_like(sample, dtype=torch.float64)
        else:
            noise = randn_tensor(sample.shape, generator=generator, device=device, dtype=torch.float64)
            noise_term = noise_std * noise

        sample_f64 = sample.to(torch.float64)

        # ER-SDE VP update
        if effective_order == 1:
            # 1st order: x_next = r_alpha * r_fn * x + alpha_next * (1 - r_fn) * x0 + noise
            prev_sample = r_alpha * r_fn * sample_f64 + alpha_next * (1.0 - r_fn) * x0 + noise_term

        elif effective_order == 2:
            # 2nd order: add first derivative correction
            d_x0 = self._safe_div(x0 - self.old_x0, lambda_curr - self.prev_lambda)
            s_int = self._compute_integral(lambda_next, lambda_curr)
            s_int = s_int.to(device)
            delta_lambda = lambda_next - lambda_curr
            correction = alpha_next * (delta_lambda + s_int * fn_lambda_next) * d_x0
            prev_sample = r_alpha * r_fn * sample_f64 + alpha_next * (1.0 - r_fn) * x0 + correction + noise_term

        elif effective_order == 3:
            # 3rd order: add second derivative correction
            d_x0 = self._safe_div(x0 - self.old_x0, lambda_curr - self.prev_lambda)
            dd_x0 = self._safe_div(2.0 * (d_x0 - self.old_d_x0), lambda_curr - self.prev_prev_lambda)
            s_int = self._compute_integral(lambda_next, lambda_curr).to(device)
            s_d_int = self._compute_derivative_integral(lambda_next, lambda_curr).to(device)
            delta_lambda = lambda_next - lambda_curr
            correction_1 = alpha_next * (delta_lambda + s_int * fn_lambda_next) * d_x0
            correction_2 = alpha_next * (delta_lambda ** 2 / 2.0 + s_d_int * fn_lambda_next) * dd_x0
            prev_sample = r_alpha * r_fn * sample_f64 + alpha_next * (1.0 - r_fn) * x0 + correction_1 + correction_2 + noise_term

        # Update state for next step
        # Compute d_x0 for buffer (needed by next step's 3rd order)
        if self.old_x0 is not None and (lambda_curr - self.prev_lambda).abs() > 1e-10:
            current_d_x0 = (x0 - self.old_x0) / (lambda_curr - self.prev_lambda)
        else:
            current_d_x0 = None

        self.prev_prev_lambda = self.prev_lambda
        self.prev_lambda = lambda_curr
        self.old_d_x0 = current_d_x0
        self.old_x0 = x0
        self.lower_order_nums = min(self.lower_order_nums + 1, self.config.solver_order)

        self._step_index += 1

        if not return_dict:
            return (prev_sample.to(dtype),)
        return ERSDESchedulerOutput(prev_sample=prev_sample.to(dtype), pred_original_sample=x0.to(dtype))

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
