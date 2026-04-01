
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class LinearRKScheduler(SchedulerMixin, ConfigMixin):
    """
    LinearRKScheduler: Standard explicit Runge-Kutta integrators.
    Supports Ralston, Midpoint, Heun, Kutta, and standard RK4.
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
        trained_betas: np.ndarray | list[float] | None = None,
        prediction_type: str = "epsilon",
        variant: str = "rk4",  # euler, heun, rk2, rk3, rk4, ralston, midpoint
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

        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.sigmas = None
        self.init_noise_sigma = 1.0

        # Internal state
        self.model_outputs = []
        self.sample_at_start_of_step = None
        self._step_index = None

    def _get_tableau(self):
        v = str(self.config.variant).lower().strip()
        if v in ["ralston", "ralston_2s"]:
            a, b, c = [[2 / 3]], [1 / 4, 3 / 4], [0, 2 / 3]
        elif v in ["midpoint", "midpoint_2s"]:
            a, b, c = [[1 / 2]], [0, 1], [0, 1 / 2]
        elif v in ["heun", "heun_2s"]:
            a, b, c = [[1]], [1 / 2, 1 / 2], [0, 1]
        elif v == "heun_3s":
            a, b, c = [[1 / 3], [0, 2 / 3]], [1 / 4, 0, 3 / 4], [0, 1 / 3, 2 / 3]
        elif v in ["kutta", "kutta_3s"]:
            a, b, c = [[1 / 2], [-1, 2]], [1 / 6, 2 / 3, 1 / 6], [0, 1 / 2, 1]
        elif v in ["rk4", "rk4_4s"]:
            a, b, c = [[1 / 2], [0, 1 / 2], [0, 0, 1]], [1 / 6, 1 / 3, 1 / 3, 1 / 6], [0, 1 / 2, 1 / 2, 1]
        elif v in ["rk2", "heun"]:
            a, b, c = [[1]], [1 / 2, 1 / 2], [0, 1]
        elif v == "rk3":
            a, b, c = [[1 / 2], [-1, 2]], [1 / 6, 2 / 3, 1 / 6], [0, 1 / 2, 1]
        elif v == "euler":
            a, b, c = [], [1], [0]
        else:
            raise ValueError(f"Unknown variant: {v}")

        # Expand 'a' to full matrix
        stages = len(c)
        full_a = np.zeros((stages, stages))
        for i, row in enumerate(a):
            full_a[i + 1, : len(row)] = row

        return full_a, np.array(b), np.array(c)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device = None,
        mu: float | None = None, dtype: torch.dtype = torch.float32):
        self.num_inference_steps = num_inference_steps

        # 1. Spacing
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(float)
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(float)
            timesteps -= 1
        else:
            raise ValueError(f"timestep_spacing must be one of 'linspace', 'leading', or 'trailing', got {self.config.timestep_spacing}")

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
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

        # We handle multi-history expansion
        _a_mat, _b_vec, c_vec = self._get_tableau()
        len(c_vec)

        sigmas_expanded = []
        for i in range(len(sigmas) - 1):
            s_curr = sigmas[i]
            s_next = sigmas[i + 1]
            for c_val in c_vec:
                sigmas_expanded.append(s_curr + c_val * (s_next - s_curr))
        sigmas_expanded.append(0.0)

        sigmas_interpolated = np.array(sigmas_expanded)
        # Linear remapping for Flow Matching
        timesteps_expanded = sigmas_interpolated * self.config.num_train_timesteps

        self.sigmas = torch.from_numpy(sigmas_interpolated).to(device=device, dtype=dtype)
        self.timesteps = torch.from_numpy(timesteps_expanded + self.config.steps_offset).to(device=device, dtype=dtype)
        self.init_noise_sigma = self.sigmas.max().item() if self.sigmas.numel() > 0 else 1.0

        self.model_outputs = []
        self.sample_at_start_of_step = None
        self._step_index = None

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
        sigma = self.sigmas[self._step_index]
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
        a_mat, b_vec, c_vec = self._get_tableau()
        num_stages = len(c_vec)

        stage_index = self._step_index % num_stages
        base_step_index = (self._step_index // num_stages) * num_stages

        sigma_curr = self.sigmas[base_step_index]
        sigma_next_idx = min(base_step_index + num_stages, len(self.sigmas) - 1)
        sigma_next = self.sigmas[sigma_next_idx]

        if sigma_next <= 0:
            sigma_t = self.sigmas[self._step_index]
            denoised = sample - sigma_t * model_output if self.config.prediction_type == "epsilon" else model_output
            prev_sample = denoised
            self._step_index += 1
            if not return_dict:
                return (prev_sample,)
            return SchedulerOutput(prev_sample=prev_sample)

        h = sigma_next - sigma_curr
        sigma_t = self.sigmas[self._step_index]

        if self.config.prediction_type == "epsilon":
            denoised = sample - sigma_t * model_output
        elif self.config.prediction_type == "v_prediction":
            alpha_t = 1 / (sigma_t**2 + 1) ** 0.5
            sigma_actual = sigma_t * alpha_t
            denoised = alpha_t * sample - sigma_actual * model_output
        elif self.config.prediction_type == "flow_prediction":
            denoised = sample - sigma_t * model_output
        elif self.config.prediction_type == "sample":
            denoised = model_output
        else:
            raise ValueError(f"prediction_type {self.config.prediction_type} is not supported.")

        if self.config.clip_sample:
            denoised = denoised.clamp(-self.config.sample_max_value, self.config.sample_max_value)

        # derivative = (x - x0) / sigma
        derivative = (sample - denoised) / sigma_t if sigma_t > 1e-6 else torch.zeros_like(sample)

        if self.sample_at_start_of_step is None:
            if stage_index > 0:
                # Mid-step fallback for Img2Img/Inpainting
                sigma_next_t = self.sigmas[self._step_index + 1]
                dt = sigma_next_t - sigma_t
                prev_sample = sample + dt * derivative
                self._step_index += 1
                if not return_dict:
                    return (prev_sample,)
                return SchedulerOutput(prev_sample=prev_sample)

            self.sample_at_start_of_step = sample
            self.model_outputs = [derivative] * stage_index

        if stage_index == 0:
            self.model_outputs = [derivative]
            self.sample_at_start_of_step = sample
        else:
            self.model_outputs.append(derivative)

        next_stage_idx = stage_index + 1
        if next_stage_idx < num_stages:
            sum_ak = 0
            for j in range(len(self.model_outputs)):
                sum_ak = sum_ak + a_mat[next_stage_idx][j] * self.model_outputs[j]

            sigma_next_stage = self.sigmas[self._step_index + 1]

            # Update x (unnormalized sample)
            prev_sample = self.sample_at_start_of_step + (sigma_next_stage - sigma_curr) * sum_ak
        else:
            sum_bk = 0
            for j in range(len(self.model_outputs)):
                sum_bk = sum_bk + b_vec[j] * self.model_outputs[j]

            prev_sample = self.sample_at_start_of_step + h * sum_bk

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
