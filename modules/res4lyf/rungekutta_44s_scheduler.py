
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


class RungeKutta44Scheduler(SchedulerMixin, ConfigMixin):
    """
    RK4: Classical 4th-order Runge-Kutta scheduler.
    Adapted from the RES4LYF repository.

    This scheduler uses 4 stages per step.
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
            raise NotImplementedError(f"{beta_schedule} is not implemented for RungeKutta44Scheduler")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sigmas = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        self.init_noise_sigma = 1.0

        # Internal state for multi-stage
        self.model_outputs = []
        self.sample_at_start_of_step = None
        self._sigmas_cpu = None
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None, mu: float | None = None, dtype: torch.dtype = torch.float32):
        self.num_inference_steps = num_inference_steps

        # 1. Base sigmas
        timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
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
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        # 2. Add sub-step sigmas for multi-stage RK
        # RK4 has c = [0, 1/2, 1/2, 1]
        c_values = [0.0, 0.5, 0.5, 1.0]

        sigmas_expanded = []
        for i in range(len(sigmas) - 1):
            s_curr = sigmas[i]
            s_next = sigmas[i + 1]
            # Intermediate sigmas: s_curr + c * (s_next - s_curr)
            for c in c_values:
                # Add a tiny epsilon to duplicate sigmas to allow distinct indexing if needed,
                # but better to rely on internal counter.
                sigmas_expanded.append(s_curr + c * (s_next - s_curr))
        sigmas_expanded.append(0.0)  # terminal sigma

        # 3. Map back to timesteps
        sigmas_interpolated = np.array(sigmas_expanded)
        # Linear remapping for Flow Matching
        timesteps_expanded = sigmas_interpolated * self.config.num_train_timesteps

        self.sigmas = torch.from_numpy(sigmas_interpolated).to(device=device, dtype=dtype)
        self.timesteps = torch.from_numpy(timesteps_expanded + self.config.steps_offset).to(device=device, dtype=dtype)
        self.init_noise_sigma = self.sigmas.max().item() if self.sigmas.numel() > 0 else 1.0

        self._sigmas_cpu = self.sigmas.detach().cpu().numpy()

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
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # Use argmin for robust float matching
        index = torch.abs(schedule_timesteps - timestep).argmin().item()
        return index

    def _init_step_index(self, timestep):
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
        stage_index = step_index % 4

        # Current and next step interval sigmas
        base_step_index = (step_index // 4) * 4
        sigma_curr = self._sigmas_cpu[base_step_index]
        sigma_next_idx = min(base_step_index + 4, len(self._sigmas_cpu) - 1)
        sigma_next = self._sigmas_cpu[sigma_next_idx]  # The sigma at the end of this 4-stage step

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

        if stage_index == 0:
            self.model_outputs = [derivative]
            self.sample_at_start_of_step = sample
            # Stage 2 input: y + 0.5 * h * k1
            prev_sample = self.sample_at_start_of_step + 0.5 * h * derivative
        elif stage_index == 1:
            self.model_outputs.append(derivative)
            # Stage 3 input: y + 0.5 * h * k2
            prev_sample = self.sample_at_start_of_step + 0.5 * h * derivative
        elif stage_index == 2:
            self.model_outputs.append(derivative)
            # Stage 4 input: y + h * k3
            prev_sample = self.sample_at_start_of_step + h * derivative
        elif stage_index == 3:
            self.model_outputs.append(derivative)
            # Final result: y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            k1, k2, k3, k4 = self.model_outputs
            prev_sample = self.sample_at_start_of_step + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # Clear state
            self.model_outputs = []
            self.sample_at_start_of_step = None

        # Increment step index
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
