
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class RungeKutta67Scheduler(SchedulerMixin, ConfigMixin):
    """
    RK6_7S: 6th-order Runge-Kutta scheduler with 7 stages.
    Adapted from the RES4LYF repository.
    (Note: Defined as 5th order in some contexts, but follows the 7-stage tableau).
    """

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
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        use_flow_sigmas: bool = False,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        rho: float = 7.0,
        shift: float | None = None,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        use_dynamic_shifting: bool = False,
        timestep_spacing: str = "linspace",
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
        self.init_noise_sigma = 1.0

        # internal state
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.sigmas = None
        self.model_outputs = []
        self.sample_at_start_of_step = None
        self._sigmas_cpu = None
        self._timesteps_cpu = None
        self._step_index = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device = None,
        mu: float | None = None, dtype: torch.dtype = torch.float32):
        self.num_inference_steps = num_inference_steps

        # 1. Spacing
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=float).copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(float)
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio).round().copy().astype(float)
            timesteps -= step_ratio
        else:
            raise ValueError(f"timestep_spacing must be one of 'linspace', 'leading', or 'trailing', got {self.config.timestep_spacing}")

        # Ensure trailing ends at 0
        if self.config.timestep_spacing == "trailing":
            timesteps = np.maximum(timesteps, 0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        if self.config.interpolation_type == "linear":
            sigmas = np.interp(timesteps, np.arange(len(sigmas)), sigmas)
        elif self.config.interpolation_type == "log_linear":
            sigmas = np.exp(np.interp(timesteps, np.arange(len(sigmas)), np.log(sigmas)))
        else:
            raise ValueError(f"interpolation_type must be one of 'linear' or 'log_linear', got {self.config.interpolation_type}")

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

        # RK6_7s c values: [0, 1/3, 2/3, 1/3, 1/2, 1/2, 1]
        c_values = [0, 1 / 3, 2 / 3, 1 / 3, 1 / 2, 1 / 2, 1]

        sigmas_expanded = []
        for i in range(len(sigmas) - 1):
            s_curr = sigmas[i]
            s_next = sigmas[i + 1]
            for c in c_values:
                sigmas_expanded.append(s_curr + c * (s_next - s_curr))
        sigmas_expanded.append(0.0)

        sigmas_interpolated = np.array(sigmas_expanded)
        # Linear remapping for Flow Matching
        timesteps_expanded = sigmas_interpolated * self.config.num_train_timesteps
        self.sigmas = torch.from_numpy(sigmas_interpolated).to(device=device, dtype=dtype)
        self.timesteps = torch.from_numpy(timesteps_expanded + self.config.steps_offset).to(device=device, dtype=dtype)

        self.init_noise_sigma = self.sigmas.max().item() if self.sigmas.numel() > 0 else 1.0
        self._sigmas_cpu = self.sigmas.detach().cpu().numpy()
        self._timesteps_cpu = self.timesteps.detach().cpu().numpy()
        self._step_index = None

        self.model_outputs = []
        self.sample_at_start_of_step = None

    @property
    def step_index(self):
        """
        The index counter for the current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        from .scheduler_utils import index_for_timestep
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        return index_for_timestep(timestep, schedule_timesteps)

    def _init_step_index(self, timestep):
        if self._step_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        if self._step_index is None:
            self._init_step_index(timestep)
        if self.config.prediction_type == "flow_prediction":
            return sample
        sigma = self._sigmas_cpu[self._step_index]
        return sample / ((sigma**2 + 1) ** 0.5)

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
        stage_index = step_index % 7

        base_step_index = (step_index // 7) * 7
        sigma_curr = self._sigmas_cpu[base_step_index]
        sigma_next_idx = min(base_step_index + 7, len(self._sigmas_cpu) - 1)
        sigma_next = self._sigmas_cpu[sigma_next_idx]
        h = sigma_next - sigma_curr

        sigma_t = self._sigmas_cpu[step_index]
        alpha_t = 1 / (sigma_t**2 + 1) ** 0.5
        sigma_actual = sigma_t * alpha_t

        prediction_type = getattr(self.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            denoised = sample - sigma_t * model_output
        elif prediction_type == "v_prediction":
            alpha_t = 1 / (sigma_t**2 + 1) ** 0.5
            sigma_actual = sigma_t * alpha_t
            denoised = alpha_t * sample - sigma_actual * model_output
        elif prediction_type == "flow_prediction":
            denoised = sample - sigma_t * model_output
        elif prediction_type == "sample":
            denoised = model_output
        else:
            raise ValueError(f"prediction_type error: {prediction_type}")

        if self.config.clip_sample:
            denoised = denoised.clamp(-self.config.sample_max_value, self.config.sample_max_value)

        # derivative = (x - x0) / sigma
        derivative = (sample - denoised) / sigma_t if sigma_t > 1e-6 else torch.zeros_like(sample)

        if self.sample_at_start_of_step is None:
            if stage_index > 0:
                # Mid-step fallback for Img2Img/Inpainting
                sigma_next_t = self._sigmas_cpu[self._step_index + 1]
                dt = sigma_next_t - sigma_t
                prev_sample = sample + dt * derivative
                self._step_index += 1
                if not return_dict:
                    return (prev_sample,)
                return SchedulerOutput(prev_sample=prev_sample)

            self.sample_at_start_of_step = sample
            self.model_outputs = [derivative] * stage_index

        # Butcher Tableau A matrix for rk6_7s
        a = [
            [],
            [1 / 3],
            [0, 2 / 3],
            [1 / 12, 1 / 3, -1 / 12],
            [-1 / 16, 9 / 8, -3 / 16, -3 / 8],
            [0, 9 / 8, -3 / 8, -3 / 4, 1 / 2],
            [9 / 44, -9 / 11, 63 / 44, 18 / 11, 0, -16 / 11],
        ]

        # Butcher Tableau B weights for rk6_7s
        b = [11 / 120, 0, 27 / 40, 27 / 40, -4 / 15, -4 / 15, 11 / 120]

        if stage_index == 0:
            self.model_outputs = [derivative]
            self.sample_at_start_of_step = sample
        else:
            self.model_outputs.append(derivative)

        if stage_index < 6:
            # Predict next stage sample: y_next_stage = y_start + h * sum(a[stage_index+1][j] * k[j])
            next_a_row = a[stage_index + 1]
            sum_ak = torch.zeros_like(derivative)
            for j, weight in enumerate(next_a_row):
                sum_ak += weight * self.model_outputs[j]

            prev_sample = self.sample_at_start_of_step + h * sum_ak
        else:
            # Final 7th stage complete, calculate final step
            sum_bk = torch.zeros_like(derivative)
            for j, weight in enumerate(b):
                sum_bk += weight * self.model_outputs[j]

            prev_sample = self.sample_at_start_of_step + h * sum_bk

            # Clear state
            self.model_outputs = []
            self.sample_at_start_of_step = None

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
        from .scheduler_utils import add_noise_to_sample
        return add_noise_to_sample(original_samples, noise, self.sigmas, timesteps, self.timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
