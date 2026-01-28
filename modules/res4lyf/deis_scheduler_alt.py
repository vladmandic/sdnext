from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


def get_def_integral_2(a, b, start, end, c):
    coeff = (end**3 - start**3) / 3 - (end**2 - start**2) * (a + b) / 2 + (end - start) * a * b
    return coeff / ((c - a) * (c - b))


def get_def_integral_3(a, b, c, start, end, d):
    coeff = (end**4 - start**4) / 4 - (end**3 - start**3) * (a + b + c) / 3 + (end**2 - start**2) * (a * b + a * c + b * c) / 2 - (end - start) * a * b * c
    return coeff / ((d - a) * (d - b) * (d - c))


class DEISMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    DEISMultistepScheduler: Diffusion Explicit Iterative Sampler with high-order multistep.
    Adapted from the RES4LYF repository.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        use_flow_sigmas: bool = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7.0,
        shift: Optional[float] = None,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        use_dynamic_shifting: bool = False,
        timestep_spacing: str = "linspace",
        solver_order: int = 2,
        clip_sample: bool = False,
        sample_max_value: float = 1.0,
        set_alpha_to_one: bool = False,
        skip_prk_steps: bool = False,
        interpolation_type: str = "linear",
        steps_offset: int = 0,
        timestep_type: str = "discrete",
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.sigmas = None

        # Internal state
        self.model_outputs = []
        self.hist_samples = []
        self._step_index = None
        self._sigmas_cpu = None
        self.all_coeffs = []

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        mu: Optional[float] = None,
    ):
        self.num_inference_steps = num_inference_steps

        # 1. Spacing
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=float).copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(float)
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = (np.arange(num_inference_steps, 0, -step_ratio)).round().copy().astype(float)
            timesteps -= step_ratio
        else:
            raise ValueError(f"timestep_spacing must be one of 'linspace', 'leading', or 'trailing', got {self.config.timestep_spacing}")

        if self.config.timestep_spacing == "trailing":
            timesteps = np.maximum(timesteps, 0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas_all = np.log(sigmas)
        if self.config.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(len(sigmas)), sigmas)
        elif self.config.interpolation_type == "log_linear":
            sigmas = np.exp(np.interp(timesteps, np.arange(len(sigmas)), np.log(sigmas)))
        else:
            raise ValueError(f"interpolation_type must be one of 'linear' or 'log_linear', got {self.config.interpolation_type}")

        # 2. Sigma Schedule
        if self.config.use_karras_sigmas:
            sigma_min = self.config.sigma_min if self.config.sigma_min is not None else sigmas[-1]
            sigma_max = self.config.sigma_max if self.config.sigma_max is not None else sigmas[0]
            rho = self.config.rho
            ramp = np.linspace(0, 1, num_inference_steps)
            sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        elif self.config.use_exponential_sigmas:
            sigma_min = self.config.sigma_min if self.config.sigma_min is not None else sigmas[-1]
            sigma_max = self.config.sigma_max if self.config.sigma_max is not None else sigmas[0]
            sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_inference_steps))
        elif self.config.use_beta_sigmas:
            sigma_min = self.config.sigma_min if self.config.sigma_min is not None else sigmas[-1]
            sigma_max = self.config.sigma_max if self.config.sigma_max is not None else sigmas[0]
            alpha, beta = 0.6, 0.6
            ramp = np.linspace(0, 1, num_inference_steps)
            try:
                import torch.distributions as dist

                b = dist.Beta(alpha, beta)
                ramp = b.sample((num_inference_steps,)).sort().values.numpy()
            except Exception:
                pass
            sigmas = sigma_max * (1 - ramp) + sigma_min * ramp
        elif self.config.use_flow_sigmas:
            sigmas = np.linspace(1.0, 1 / 1000, num_inference_steps)

        # 3. Shifting
        if self.config.use_dynamic_shifting and mu is not None:
            sigmas = mu * sigmas / (1 + (mu - 1) * sigmas)
        elif self.config.shift is not None:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        # Map back to timesteps
        timesteps = np.interp(np.log(np.maximum(sigmas, 1e-10)), log_sigmas_all, np.arange(len(log_sigmas_all)))

        self.sigmas = torch.from_numpy(np.append(sigmas, 0.0)).to(device=device, dtype=torch.float32)
        self.timesteps = torch.from_numpy(timesteps + self.config.steps_offset).to(device=device, dtype=torch.float32)

        self._sigmas_cpu = self.sigmas.detach().cpu().numpy()

        # Precompute coefficients
        self.all_coeffs = []
        num_steps = len(timesteps)
        for i in range(num_steps):
            sigma_t = self._sigmas_cpu[i]
            sigma_next = self._sigmas_cpu[i + 1]

            if sigma_next <= 0:
                coeffs = None
            else:
                current_order = min(i + 1, self.config.solver_order)
                if current_order == 1:
                    coeffs = [sigma_next - sigma_t]
                else:
                    ts = [self._sigmas_cpu[i - j] for j in range(current_order)]
                    t_next = sigma_next
                    if current_order == 2:
                        t_cur, t_prev1 = ts[0], ts[1]
                        coeff_cur = ((t_next - t_prev1) ** 2 - (t_cur - t_prev1) ** 2) / (2 * (t_cur - t_prev1))
                        coeff_prev1 = (t_next - t_cur) ** 2 / (2 * (t_prev1 - t_cur))
                        coeffs = [coeff_cur, coeff_prev1]
                    elif current_order == 3:
                        t_cur, t_prev1, t_prev2 = ts[0], ts[1], ts[2]
                        coeffs = [
                            get_def_integral_2(t_prev1, t_prev2, t_cur, t_next, t_cur),
                            get_def_integral_2(t_cur, t_prev2, t_cur, t_next, t_prev1),
                            get_def_integral_2(t_cur, t_prev1, t_cur, t_next, t_prev2),
                        ]
                    elif current_order == 4:
                        t_cur, t_prev1, t_prev2, t_prev3 = ts[0], ts[1], ts[2], ts[3]
                        coeffs = [
                            get_def_integral_3(t_prev1, t_prev2, t_prev3, t_cur, t_next, t_cur),
                            get_def_integral_3(t_cur, t_prev2, t_prev3, t_cur, t_next, t_prev1),
                            get_def_integral_3(t_cur, t_prev1, t_prev3, t_cur, t_next, t_prev2),
                            get_def_integral_3(t_cur, t_prev1, t_prev2, t_cur, t_next, t_prev3),
                        ]
                    else:
                        coeffs = [(sigma_next - sigma_t) / sigma_t]  # Fallback to Euler
            self.all_coeffs.append(coeffs)

        # Reset history
        self.model_outputs = []
        self.hist_samples = []
        self._step_index = None

    @property
    def step_index(self):
        """
        The index counter for the current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if self._step_index is not None:
            return self._step_index

        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        if isinstance(schedule_timesteps, torch.Tensor):
            schedule_timesteps = schedule_timesteps.detach().cpu().numpy()

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.detach().cpu().numpy()

        return np.abs(schedule_timesteps - timestep).argmin().item()

    def _init_step_index(self, timestep):
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep)

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        if self._step_index is None:
            self._init_step_index(timestep)

        step_index = self._step_index
        sigma_t = self.sigmas[step_index]
        # Calculate alpha_t and sigma_t (NSR)
        alpha_t = 1 / (sigma_t**2 + 1) ** 0.5
        sigma_actual = sigma_t * alpha_t

        if self.config.prediction_type == "epsilon":
            denoised = (sample / alpha_t) - sigma_t * model_output
        elif self.config.prediction_type == "v_prediction":
            denoised = alpha_t * sample - sigma_actual * model_output
        elif self.config.prediction_type == "flow_prediction":
            alpha_t = 1.0
            denoised = sample - sigma_t * model_output
        elif self.config.prediction_type == "sample":
            denoised = model_output
        else:
            raise ValueError(f"prediction_type error: {self.config.prediction_type}")

        if self.config.clip_sample:
            denoised = denoised.clamp(-self.config.sample_max_value, self.config.sample_max_value)

        # DEIS coefficients are precomputed in set_timesteps
        coeffs = self.all_coeffs[step_index]

        sigma_next = self.sigmas[step_index + 1]
        alpha_next = 1 / (sigma_next**2 + 1) ** 0.5 if sigma_next > 0 else 1.0

        if coeffs is None:
            prev_sample = denoised
        else:
            current_order = len(coeffs)
            if current_order == 1:
                # 1st order step (Euler) in normalized space
                prev_sample_norm = (sigma_next / sigma_t) * (sample / alpha_t) + (1 - sigma_next / sigma_t) * denoised
                prev_sample = prev_sample_norm * alpha_next
            else:
                # Xs: [x0_curr, x0_prev1, x0_prev2, ...]
                x0s = [denoised, *self.model_outputs[::-1][: current_order - 1]]

                # Normalize DEIS coefficients to get weights for x0 interpolation
                # sum(coeffs) = sigma_next - sigma_t
                delta_sigma = sigma_next - sigma_t
                if abs(delta_sigma) > 1e-8:
                    weights = [c / delta_sigma for c in coeffs]
                else:
                    weights = [1.0] + [0.0] * (current_order - 1)

                mixed_x0 = 0
                for i in range(current_order):
                    mixed_x0 = mixed_x0 + weights[i] * x0s[i]

                # Stable update in normalized space
                prev_sample_norm = (sigma_next / sigma_t) * (sample / alpha_t) + (1 - sigma_next / sigma_t) * mixed_x0
                prev_sample = prev_sample_norm * alpha_next

        # Store state (always store x0)
        self.model_outputs.append(denoised)
        self.hist_samples.append(sample)

        if len(self.model_outputs) > 4:
            self.model_outputs.pop(0)
            self.hist_samples.pop(0)

        if self._step_index is not None:
            self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        step_indices = [self.index_for_timestep(t) for t in timesteps]
        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        return original_samples + noise * sigma

    @property
    def init_noise_sigma(self):
        return 1.0

    def __len__(self):
        return self.config.num_train_timesteps
