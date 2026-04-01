from typing import ClassVar

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

from .phi_functions import Phi


class RESUnifiedScheduler(SchedulerMixin, ConfigMixin):
    """
    RESUnifiedScheduler (Exponential Integrator) ported from RES4LYF.
    Supports RES 2M, 3M, 2S, 3S, 5S, 6S
    Supports DEIS 1S, 2M, 3M
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
        prediction_type: str = "epsilon",
        rk_type: str = "res_2m",
        use_analytic_solution: bool = True,
        rescale_betas_zero_snr: bool = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
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

        self.sigmas = torch.Tensor([])
        self.timesteps = torch.Tensor([])
        self.model_outputs = []
        self.x0_outputs = []
        self.prev_sigmas = []

        self._step_index = None
        self._begin_index = None
        self.init_noise_sigma = 1.0

    def set_sigmas(self, sigmas: torch.Tensor):
        self.sigmas = sigmas
        self._step_index = None

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
        return sample / ((sigma**2 + 1) ** 0.5)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, mu: float | None = None, dtype: torch.dtype = torch.float32):
        from .scheduler_utils import (
            apply_shift,
            get_dynamic_shift,
            get_sigmas_beta,
            get_sigmas_exponential,
            get_sigmas_karras,
        )

        self.num_inference_steps = num_inference_steps
        timestep_spacing = getattr(self.config, "timestep_spacing", "linspace")
        steps_offset = getattr(self.config, "steps_offset", 0)

        if timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        elif timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
            timesteps += steps_offset
        elif timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy()
            timesteps -= 1
        else:
            raise ValueError(f"timestep_spacing {timestep_spacing} is not supported.")

        # Derived sigma range from alphas_cumprod
        base_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = base_sigmas[::-1].copy() # Ensure high to low

        if getattr(self.config, "use_karras_sigmas", False):
            sigmas = get_sigmas_karras(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()
        elif getattr(self.config, "use_exponential_sigmas", False):
            sigmas = get_sigmas_exponential(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()
        elif getattr(self.config, "use_beta_sigmas", False):
            sigmas = get_sigmas_beta(num_inference_steps, sigmas[-1], sigmas[0], device=device, dtype=dtype).cpu().numpy()
        elif getattr(self.config, "use_flow_sigmas", False):
            sigmas = np.linspace(1.0, 1 / 1000, num_inference_steps)
        else:
             if self.config.use_flow_sigmas:
                  sigmas = np.linspace(1.0, 1 / 1000, num_inference_steps)
             else:
                  # Re-sample the base sigmas at the requested steps
                  idx = np.linspace(0, len(base_sigmas) - 1, num_inference_steps)
                  sigmas = np.interp(idx, np.arange(len(base_sigmas)), base_sigmas)[::-1].copy()

        shift = getattr(self.config, "shift", 1.0)
        use_dynamic_shifting = getattr(self.config, "use_dynamic_shifting", False)
        if shift != 1.0 or use_dynamic_shifting:
            if use_dynamic_shifting and mu is not None:
                shift = get_dynamic_shift(
                    mu,
                    getattr(self.config, "base_shift", 0.5),
                    getattr(self.config, "max_shift", 1.5),
                    getattr(self.config, "base_image_seq_len", 256),
                    getattr(self.config, "max_image_seq_len", 4096),
                )
            sigmas = apply_shift(torch.from_numpy(sigmas), shift).numpy()

        if getattr(self.config, "use_flow_sigmas", False):
             timesteps = sigmas * self.config.num_train_timesteps

        self.sigmas = torch.from_numpy(np.concatenate([sigmas, [0.0]])).to(device=device, dtype=dtype)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=dtype)
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

    def _get_coefficients(self, sigma, sigma_next):
        h = -torch.log(sigma_next / sigma) if sigma > 0 else torch.zeros_like(sigma)
        phi = Phi(h, [], getattr(self.config, "use_analytic_solution", True))
        phi_1 = phi(1)
        phi_2 = phi(2)
        # phi_2 = phi(2) # Moved inside conditional blocks as needed

        history_len = len(self.x0_outputs)

        # Stability: Force Order 1 for final few steps to prevent degradation at low noise levels
        if self.num_inference_steps is not None and self._step_index >= self.num_inference_steps - 3:
            return [phi_1], h

        if self.config.rk_type in ["res_2m", "deis_2m"] and history_len >= 2:
            h_prev = -torch.log(self.prev_sigmas[-1] / (self.prev_sigmas[-2] + 1e-9))
            r = h_prev / (h + 1e-9)

            h_prev = -torch.log(self.prev_sigmas[-1] / (self.prev_sigmas[-2] + 1e-9))
            r = h_prev / (h + 1e-9)

            # Hard Restart: if step sizes vary too wildly, fallback to order 1
            if r < 0.5 or r > 2.0:
                 return [phi_1], h

            phi_2 = phi(2)
            # Correct Adams-Bashforth-like coefficients for Exponential Integrators
            b2 = -phi_2 / (r + 1e-9)
            b1 = phi_1 - b2
            return [b1, b2], h
        elif self.config.rk_type in ["res_3m", "deis_3m"] and history_len >= 3:
            h_prev1 = -torch.log(self.prev_sigmas[-1] / (self.prev_sigmas[-2] + 1e-9))
            h_prev2 = -torch.log(self.prev_sigmas[-1] / (self.prev_sigmas[-3] + 1e-9))
            r1 = h_prev1 / (h + 1e-9)
            r2 = h_prev2 / (h + 1e-9)

            h_prev1 = -torch.log(self.prev_sigmas[-1] / (self.prev_sigmas[-2] + 1e-9))
            h_prev2 = -torch.log(self.prev_sigmas[-1] / (self.prev_sigmas[-3] + 1e-9))
            r1 = h_prev1 / (h + 1e-9)
            r2 = h_prev2 / (h + 1e-9)

            # Hard Restart check
            if r1 < 0.5 or r1 > 2.0 or r2 < 0.5 or r2 > 2.0:
                 return [phi_1], h

            phi_2 = phi(2)
            phi_3 = phi(3)

            # Generalized AB3 for Exponential Integrators (Varying steps)
            denom = r2 - r1 + 1e-9
            b3 = (phi_3 + r1 * phi_2) / (r2 * denom)
            b2 = -(phi_3 + r2 * phi_2) / (r1 * denom)
            b1 = phi_1 - (b2 + b3)
            return [b1, b2, b3], h

        return [phi_1], h

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        if self._step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        h = -torch.log(sigma_next / sigma) if sigma > 0 and sigma_next > 0 else torch.zeros_like(sigma)

        # RECONSTRUCT X0 (Matching PEC pattern)
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

        self.x0_outputs.append(x0)
        self.model_outputs.append(model_output) # Added for AB support
        self.prev_sigmas.append(sigma)

        if len(self.x0_outputs) > 3:
            self.x0_outputs.pop(0)
            self.model_outputs.pop(0)
            self.prev_sigmas.pop(0)

        if self.config.prediction_type == "flow_prediction":
            # Variable Step Adams-Bashforth for Flow Matching
            dt = sigma_next - sigma
            v_n = model_output

            curr_order = min(len(self.prev_sigmas), 3) # Max order 3 here

            if curr_order == 1:
                 x_next = sample + dt * v_n
            elif curr_order == 2:
                 sigma_prev = self.prev_sigmas[-2]
                 dt_prev = sigma - sigma_prev
                 r = dt / dt_prev if abs(dt_prev) > 1e-8 else 0.0
                 if dt_prev == 0 or r < -0.9 or r > 2.0:
                     x_next = sample + dt * v_n
                 else:
                     c0 = 1 + 0.5 * r
                     c1 = -0.5 * r
                     x_next = sample + dt * (c0 * v_n + c1 * self.model_outputs[-2])
            else:
                 # AB2 fallback for robustness
                 sigma_prev = self.prev_sigmas[-2]
                 dt_prev = sigma - sigma_prev
                 r = dt / dt_prev if abs(dt_prev) > 1e-8 else 0.0
                 c0 = 1 + 0.5 * r
                 c1 = -0.5 * r
                 x_next = sample + dt * (c0 * v_n + c1 * self.model_outputs[-2])

            self._step_index += 1
            if not return_dict:
                return (x_next,)
            return SchedulerOutput(prev_sample=x_next)

        # GET COEFFICIENTS
        b, h_val = self._get_coefficients(sigma, sigma_next)

        if len(b) == 1:
            res = b[0] * x0
        elif len(b) == 2:
            res = b[0] * self.x0_outputs[-1] + b[1] * self.x0_outputs[-2]
        elif len(b) == 3:
            res = b[0] * self.x0_outputs[-1] + b[1] * self.x0_outputs[-2] + b[2] * self.x0_outputs[-3]
        else:
            res = b[0] * x0

        # UPDATE
        if sigma_next == 0:
            x_next = x0
        else:
            # Propagate in x-space (unnormalized)
            x_next = torch.exp(-h) * sample + h * res

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
