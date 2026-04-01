# Copyright 2025 The RES4LYF Team (Clybius) and The HuggingFace Team. All rights reserved.
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

from typing import ClassVar, Literal

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

from diffusers.utils import logging

logger = logging.get_logger(__name__)


class RESMultistepSDEScheduler(SchedulerMixin, ConfigMixin):
    """
    RESMultistepSDEScheduler (Stochastic Exponential Integrator) ported from RES4LYF.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        variant (`str`, defaults to "res_2m"):
            The specific RES/DEIS variant to use. Supported: "res_2m", "res_3m".
        eta (`float`, defaults to 1.0):
            The amount of noise to add during sampling (0.0 for ODE, 1.0 for full SDE).
    """

    _compatibles: ClassVar[list[str]] = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        variant: Literal["res_2m", "res_3m"] = "res_2m",
        eta: float = 1.0,
        use_analytic_solution: bool = True,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        use_flow_sigmas: bool = False,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
    ):
        from .scheduler_utils import betas_for_alpha_bar, rescale_zero_terminal_snr

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Buffer for multistep
        self.model_outputs = []
        self.x0_outputs = []
        self.prev_sigmas = []
        self._step_index = None
        self._begin_index = None
        self.init_noise_sigma = 1.0

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        if self._step_index is None:
            self._init_step_index(timestep)
        if self.config.prediction_type == "flow_prediction":
            return sample
        sigma = self.sigmas[self._step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, mu: float | None = None, dtype: torch.dtype = torch.float32):
        from .scheduler_utils import (
            apply_shift,
            get_dynamic_shift,
            get_sigmas_beta,
            get_sigmas_exponential,
            get_sigmas_flow,
            get_sigmas_karras,
        )

        self.num_inference_steps = num_inference_steps

        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy()
            timesteps -= 1
        else:
            raise ValueError(f"timestep_spacing {self.config.timestep_spacing} is not supported.")

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.use_karras_sigmas:
            sigmas = get_sigmas_karras(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()
        elif self.config.use_exponential_sigmas:
            sigmas = get_sigmas_exponential(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()
        elif self.config.use_beta_sigmas:
            sigmas = get_sigmas_beta(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()
        elif self.config.use_flow_sigmas:
            sigmas = get_sigmas_flow(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()

        if self.config.shift != 1.0 or self.config.use_dynamic_shifting:
            shift = self.config.shift
            if self.config.use_dynamic_shifting and mu is not None:
                shift = get_dynamic_shift(
                    mu,
                    self.config.base_shift,
                    self.config.max_shift,
                    self.config.base_image_seq_len,
                    self.config.max_image_seq_len,
                )
            sigmas = apply_shift(torch.from_numpy(sigmas), shift).numpy()

        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]])).to(device=device, dtype=dtype)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=dtype)
        self.init_noise_sigma = self.sigmas.max().item() if self.sigmas.numel() > 0 else 1.0

        self._step_index = None
        self._begin_index = None
        self.model_outputs = []
        self.x0_outputs = []
        self.prev_sigmas = []

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        from .scheduler_utils import index_for_timestep
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        return index_for_timestep(timestep, schedule_timesteps)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        from .scheduler_utils import add_noise_to_sample
        return add_noise_to_sample(original_samples, noise, self.sigmas, timesteps, self.timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        if self._step_index is None:
            self._init_step_index(timestep)

        step = self._step_index
        sigma = self.sigmas[step]
        sigma_next = self.sigmas[step + 1]

        h = -torch.log(sigma_next / sigma) if sigma > 0 and sigma_next > 0 else torch.zeros_like(sigma)

        # RECONSTRUCT X0
        if self.config.prediction_type == "epsilon":
            x0 = sample - sigma * model_output
        elif self.config.prediction_type == "sample":
            x0 = model_output
        elif self.config.prediction_type == "v_prediction":
            alpha_t = 1.0 / (sigma**2 + 1) ** 0.5
            sigma_t = sigma * alpha_t
            x0 = alpha_t * sample - sigma_t * model_output
        elif self.config.prediction_type == "flow_prediction":
            x0 = sample - sigma * model_output
        else:
            x0 = model_output

        self.model_outputs.append(model_output)
        self.x0_outputs.append(x0)
        self.prev_sigmas.append(sigma)

        # Order logic
        variant = self.config.variant
        order = int(variant[-2]) if variant.endswith("m") else 1

        # Effective order for current step
        curr_order = min(len(self.prev_sigmas), order)

        # REiS Multistep logic
        c2, c3 = 0.5, 1.0
        if curr_order == 2:
            h_prev = -torch.log(self.prev_sigmas[-1] / self.prev_sigmas[-2])
            c2 = (-h_prev / h).item() if h > 0 else 0.5
            rk_type = "res_2s"
        elif curr_order == 3:
            h_prev1 = -torch.log(self.prev_sigmas[-1] / self.prev_sigmas[-2])
            h_prev2 = -torch.log(self.prev_sigmas[-1] / self.prev_sigmas[-3])
            c2 = (-h_prev1 / h).item() if h > 0 else 0.5
            c3 = (-h_prev2 / h).item() if h > 0 else 1.0
            rk_type = "res_3s"
        else:
            rk_type = "res_1s"

        if curr_order == 1:
            rk_type = "res_1s"
        _a, b, _ci = self._get_res_coefficients(rk_type, h, c2, c3)

        # Apply coefficients to get multistep x_0
        res = torch.zeros_like(sample)
        for i, b_val in enumerate(b[0]):
            idx = len(self.x0_outputs) - 1 - i
            if idx >= 0:
                res += b_val * self.x0_outputs[idx]

        # SDE stochastic step
        eta = self.config.eta
        if sigma_next == 0:
            x_next = x0
        else:
            # Ancestral SDE logic:
            # 1. Calculate sigma_up and sigma_down to preserve variance
            # sigma_up = eta * sigma_next * sqrt(1 - (sigma_next/sigma)^2)
            # sigma_down = sqrt(sigma_next^2 - sigma_up^2)

            sigma_up = eta * (sigma_next**2 * (sigma**2 - sigma_next**2) / (sigma**2 + 1e-9))**0.5
            sigma_down = (sigma_next**2 - sigma_up**2)**0.5

            # 2. Take deterministic step to sigma_down
            h_det = -torch.log(sigma_down / sigma) if sigma > 0 and sigma_down > 0 else h

            # Re-calculate coefficients for h_det
            _a, b_det, _ci = self._get_res_coefficients(rk_type, h_det, c2, c3)
            res_det = torch.zeros_like(sample)
            for i, b_val in enumerate(b_det[0]):
                idx = len(self.x0_outputs) - 1 - i
                if idx >= 0:
                    res_det += b_val * self.x0_outputs[idx]

            x_det = torch.exp(-h_det) * sample + h_det * res_det

            # 3. Add noise scaled by sigma_up
            if eta > 0:
                noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
                x_next = x_det + sigma_up * noise
            else:
                x_next = x_det

        self._step_index += 1

        if len(self.x0_outputs) > order:
            self.x0_outputs.pop(0)
            self.model_outputs.pop(0)
            self.prev_sigmas.pop(0)

        if not return_dict:
            return (x_next,)

        return SchedulerOutput(prev_sample=x_next)

    def _get_res_coefficients(self, rk_type, h, c2, c3):
        from .phi_functions import Phi, calculate_gamma
        ci = [0, c2, c3]
        phi = Phi(h, ci, self.config.use_analytic_solution)

        if rk_type == "res_2s":
            b2 = phi(2) / (c2 + 1e-9)
            b = [[phi(1) - b2, b2]]
            a = [[0, 0], [c2 * phi(1, 2), 0]]
        elif rk_type == "res_3s":
            gamma_val = calculate_gamma(c2, c3)
            b3 = phi(2) / (gamma_val * c2 + c3 + 1e-9)
            b2 = gamma_val * b3
            b = [[phi(1) - (b2 + b3), b2, b3]]
            a = []
        else:
            b = [[phi(1)]]
            a = [[0]]
        return a, b, ci

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def __len__(self):
        return self.config.num_train_timesteps
