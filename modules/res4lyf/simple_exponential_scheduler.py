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

from typing import ClassVar

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

from diffusers.utils import logging

logger = logging.get_logger(__name__)


class SimpleExponentialScheduler(SchedulerMixin, ConfigMixin):
    """
    Simple Exponential sigma scheduler using Exponential Integrator step.
    """

    _compatibles: ClassVar[list[str]] = [e.name for e in KarrasDiffusionSchedulers]
    order: ClassVar[int] = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        sigma_max: float = 1.0,
        sigma_min: float = 0.01,
        gain: float = 1.0,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
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
            raise NotImplementedError(f"{beta_schedule} does not exist.")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # Setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.sigmas = torch.zeros((num_train_timesteps,), dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

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

        sigmas = np.exp(np.linspace(np.log(self.config.sigma_max), np.log(self.config.sigma_min), num_inference_steps))

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
        self.timesteps = torch.from_numpy(np.linspace(1000, 0, num_inference_steps)).to(device=device, dtype=dtype)
        self.init_noise_sigma = self.sigmas.max().item() if self.sigmas.numel() > 0 else 1.0

        self._step_index = None
        self._begin_index = None

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

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        if self._step_index is None:
            self._init_step_index(timestep)
        if self.config.prediction_type == "flow_prediction":
            return sample
        sigma = self.sigmas[self._step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        if self._step_index is None:
            self._init_step_index(timestep)

        step_index = self._step_index
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        # Determine denoised (x_0 prediction)
        if self.config.prediction_type == "epsilon":
            x0 = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            alpha_t = 1.0 / (sigma**2 + 1)**0.5
            sigma_t = sigma * alpha_t
            x0 = alpha_t * sample - sigma_t * model_output
        elif self.config.prediction_type == "sample":
            x0 = model_output
        elif self.config.prediction_type == "flow_prediction":
            x0 = sample - sigma * model_output
        else:
            x0 = model_output

        # Exponential Integrator Update (1st order)
        if sigma_next == 0:
            x_next = x0
        else:
            h = -torch.log(sigma_next / sigma) if sigma > 0 and sigma_next > 0 else torch.zeros_like(sigma)
            x_next = torch.exp(-h) * sample + (1 - torch.exp(-h)) * x0

        self._step_index += 1

        if not return_dict:
            return (x_next,)
        return SchedulerOutput(prev_sample=x_next)

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def __len__(self):
        return self.config.num_train_timesteps
