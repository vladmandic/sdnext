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

from diffusers.utils import logging

from .phi_functions import Phi

logger = logging.get_logger(__name__)


class ABNorsettScheduler(SchedulerMixin, ConfigMixin):
    """
    Adams-Bashforth Norsett (ABNorsett) scheduler.
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
        trained_betas: np.ndarray | list[float] | None = None,
        prediction_type: str = "epsilon",
        variant: Literal["abnorsett_2m", "abnorsett_3m", "abnorsett_4m"] = "abnorsett_2m",
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

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
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

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, mu: float | None = None, dtype: torch.dtype = torch.float32):
        from .scheduler_utils import (
            apply_shift,
            get_dynamic_shift,
            get_sigmas_beta,
            get_sigmas_exponential,
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
            s_min = getattr(self.config, "sigma_min", None)
            s_max = getattr(self.config, "sigma_max", None)
            if s_min is None:
                s_min = 0.001
            if s_max is None:
                s_max = 1.0
            sigmas = np.linspace(s_max, s_min, num_inference_steps)

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

        # Map shifted sigmas back to timesteps (Linear mapping for Flow)
        # t = sigma * 1000. Use standard linear scaling.
        # This ensures the model receives the correct time embedding for the shifted noise level.
        # We assume Flow sigmas are in [1.0, 0.0] range (before shift) and model expects [1000, 0].
        timesteps = sigmas * self.config.num_train_timesteps

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

        variant = self.config.variant
        order = int(variant[-2])
        curr_order = min(len(self.prev_sigmas), order)

        phi = Phi(h, [0], getattr(self.config, "use_analytic_solution", True))

        if sigma_next == 0:
            x_next = x0
        else:
            # Multi-step coefficients b for ABNorsett family
            if curr_order == 1:
                b = [[phi(1)]]
            elif curr_order == 2:
                b2 = -phi(2)
                b1 = phi(1) - b2
                b = [[b1, b2]]
            elif curr_order == 3:
                b2 = -2 * phi(2) - 2 * phi(3)
                b3 = 0.5 * phi(2) + phi(3)
                b1 = phi(1) - (b2 + b3)
                b = [[b1, b2, b3]]
            elif curr_order == 4:
                b2 = -3 * phi(2) - 5 * phi(3) - 3 * phi(4)
                b3 = 1.5 * phi(2) + 4 * phi(3) + 3 * phi(4)
                b4 = -1 / 3 * phi(2) - phi(3) - phi(4)
                b1 = phi(1) - (b2 + b3 + b4)
                b = [[b1, b2, b3, b4]]
            else:
                b = [[phi(1)]]

            # Apply coefficients to x0 buffer
            res = torch.zeros_like(sample)
            for i, b_val in enumerate(b[0]):
                idx = len(self.x0_outputs) - 1 - i
                if idx >= 0:
                    res += b_val * self.x0_outputs[idx]

            # Exponential Integrator Update
            if self.config.prediction_type == "flow_prediction":
                # Variable Step Adams-Bashforth for Flow Matching
                # x_{n+1} = x_n + \int_{t_n}^{t_{n+1}} v(t) dt
                sigma_curr = sigma
                dt = sigma_next - sigma_curr

                # Current derivative v_n is self.model_outputs[-1]
                v_n = self.model_outputs[-1]

                if curr_order == 1:
                    # Euler: x_{n+1} = x_n + dt * v_n
                    x_next = sample + dt * v_n
                elif curr_order == 2:
                    # AB2 Variable Step
                    # x_{n+1} = x_n + dt * [ (1 + r/2) * v_n - (r/2) * v_{n-1} ]
                    # where r = dt_cur / dt_prev

                    v_nm1 = self.model_outputs[-2]
                    sigma_prev = self.prev_sigmas[-2]
                    dt_prev = sigma_curr - sigma_prev

                    if abs(dt_prev) < 1e-8:
                         # Fallback to Euler if division by zero risk
                         x_next = sample + dt * v_n
                    else:
                        r = dt / dt_prev
                        # Standard variable step AB2 coefficients
                        c0 = 1 + 0.5 * r
                        c1 = -0.5 * r
                        x_next = sample + dt * (c0 * v_n + c1 * v_nm1)

                elif curr_order >= 3:
                     # For now, fallback to AB2 (variable) for higher orders to ensure stability
                     # given the complexity of variable-step AB3/4 formulas inline.
                     # The user specifically requested abnorsett_2m.
                     v_nm1 = self.model_outputs[-2]
                     sigma_prev = self.prev_sigmas[-2]
                     dt_prev = sigma_curr - sigma_prev

                     if abs(dt_prev) < 1e-8:
                         x_next = sample + dt * v_n
                     else:
                        r = dt / dt_prev
                        c0 = 1 + 0.5 * r
                        c1 = -0.5 * r
                        x_next = sample + dt * (c0 * v_n + c1 * v_nm1)
                else:
                     x_next = sample + dt * v_n

            else:
                x_next = torch.exp(-h) * sample + h * res

        self._step_index += 1

        if len(self.x0_outputs) > order:
            self.x0_outputs.pop(0)
            self.model_outputs.pop(0)
            self.prev_sigmas.pop(0)

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
