
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

from .phi_functions import Phi


def get_def_integral_2(a, b, start, end, c):
    coeff = (end**3 - start**3) / 3 - (end**2 - start**2) * (a + b) / 2 + (end - start) * a * b
    return coeff / ((c - a) * (c - b))


def get_def_integral_3(a, b, c, start, end, d):
    coeff = (end**4 - start**4) / 4 - (end**3 - start**3) * (a + b + c) / 3 + (end**2 - start**2) * (a * b + a * c + b * c) / 2 - (end - start) * a * b * c
    return coeff / ((d - a) * (d - b) * (d - c))


class RESDEISMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    RESDEISMultistepScheduler: Diffusion Explicit Iterative Sampler with high-order multistep.
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
        solver_order: int = 2,
        use_analytic_solution: bool = True,
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
        self.hist_samples = []
        self._step_index = None
        self._sigmas_cpu = None
        self.all_coeffs = []
        self.prev_sigmas = []

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device = None,
        mu: float | None = None,
        dtype: torch.dtype = torch.float32):

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
        log_sigmas_all = np.log(np.maximum(sigmas, 1e-10))
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
        if self.config.use_flow_sigmas:
             timesteps = sigmas * self.config.num_train_timesteps
        else:
             timesteps = np.interp(np.log(np.maximum(sigmas, 1e-10)), log_sigmas_all, np.arange(len(log_sigmas_all)))

        self.sigmas = torch.from_numpy(np.append(sigmas, 0.0)).to(device=device, dtype=dtype)
        self.timesteps = torch.from_numpy(timesteps + self.config.steps_offset).to(device=device, dtype=dtype)
        self.init_noise_sigma = self.sigmas.max().item() if self.sigmas.numel() > 0 else 1.0

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
        from .scheduler_utils import index_for_timestep
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        return index_for_timestep(timestep, schedule_timesteps)

    def _init_step_index(self, timestep):
        if self._step_index is None:
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

        step_index = self._step_index
        sigma_t = self.sigmas[step_index]

        # RECONSTRUCT X0 (Matching PEC pattern)
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
            raise ValueError(f"prediction_type error: {self.config.prediction_type}")

        if self.config.clip_sample:
            denoised = denoised.clamp(-self.config.sample_max_value, self.config.sample_max_value)

        if self.config.prediction_type == "flow_prediction":
            # Variable Step Adams-Bashforth for Flow Matching
            self.model_outputs.append(model_output)
            self.prev_sigmas.append(sigma_t)
            # Note: deis uses hist_samples for x0? I'll use model_outputs for v.
            if len(self.model_outputs) > 4:
                 self.model_outputs.pop(0)
                 self.prev_sigmas.pop(0)

            dt = self.sigmas[step_index + 1] - sigma_t
            v_n = model_output

            curr_order = min(len(self.prev_sigmas), 3)

            if curr_order == 1:
                 x_next = sample + dt * v_n
            elif curr_order == 2:
                 sigma_prev = self.prev_sigmas[-2]
                 dt_prev = sigma_t - sigma_prev
                 r = dt / dt_prev if abs(dt_prev) > 1e-8 else 0.0
                 if dt_prev == 0 or r < -0.9 or r > 2.0:
                     x_next = sample + dt * v_n
                 else:
                     c0 = 1 + 0.5 * r
                     c1 = -0.5 * r
                     x_next = sample + dt * (c0 * v_n + c1 * self.model_outputs[-2])
            else:
                 # AB2 fallback
                 sigma_prev = self.prev_sigmas[-2]
                 dt_prev = sigma_t - sigma_prev
                 r = dt / dt_prev if abs(dt_prev) > 1e-8 else 0.0
                 c0 = 1 + 0.5 * r
                 c1 = -0.5 * r
                 x_next = sample + dt * (c0 * v_n + c1 * self.model_outputs[-2])

            self._step_index += 1
            if not return_dict:
                return (x_next,)
            return SchedulerOutput(prev_sample=x_next)

        sigma_next = self.sigmas[step_index + 1]

        if self.config.solver_order == 1:
            # 1st order step (Euler) in x-space
            x_next = (sigma_next / sigma_t) * sample + (1 - sigma_next / sigma_t) * denoised
            prev_sample = x_next
        else:
            # Multistep weights based on phi functions (consistent with RESMultistep)
            h = -torch.log(sigma_next / sigma_t) if sigma_t > 0 and sigma_next > 0 else torch.zeros_like(sigma_t)
            phi = Phi(h, [0], getattr(self.config, "use_analytic_solution", True))
            phi_1 = phi(1)

            # History of denoised samples
            x0s = [denoised] + self.model_outputs[::-1]
            orders = min(len(x0s), self.config.solver_order)

            # Force Order 1 at the end of schedule
            if self.num_inference_steps is not None and step_index >= self.num_inference_steps - 3:
                res = phi_1 * denoised
            elif orders == 1:
                res = phi_1 * denoised
            elif orders == 2:
                # Use phi(2) for 2nd order interpolation
                h_prev = -np.log(self._sigmas_cpu[step_index] / (self._sigmas_cpu[step_index - 1] + 1e-9))
                h_prev_t = torch.tensor(h_prev, device=sample.device, dtype=sample.dtype)
                r = h_prev_t / (h + 1e-9)
                h_prev = -np.log(self._sigmas_cpu[step_index] / (self._sigmas_cpu[step_index - 1] + 1e-9))
                h_prev_t = torch.tensor(h_prev, device=sample.device, dtype=sample.dtype)
                r = h_prev_t / (h + 1e-9)

                # Hard Restart
                if r < 0.5 or r > 2.0:
                    res = phi_1 * denoised
                else:
                    phi_2 = phi(2)
                    # Correct Adams-Bashforth-like coefficients: b2 = -phi_2 / r
                    b2 = -phi_2 / (r + 1e-9)
                    b1 = phi_1 - b2
                    res = b1 * x0s[0] + b2 * x0s[1]
            elif orders == 3:
                # 3rd order with varying step sizes
                # 3rd order with varying step sizes
                h_p1 = -np.log(self._sigmas_cpu[step_index] / (self._sigmas_cpu[step_index - 1] + 1e-9))
                h_p2 = -np.log(self._sigmas_cpu[step_index] / (self._sigmas_cpu[step_index - 2] + 1e-9))
                r1 = torch.tensor(h_p1, device=sample.device, dtype=sample.dtype) / (h + 1e-9)
                r2 = torch.tensor(h_p2, device=sample.device, dtype=sample.dtype) / (h + 1e-9)
                h_p1 = -np.log(self._sigmas_cpu[step_index] / (self._sigmas_cpu[step_index - 1] + 1e-9))
                h_p2 = -np.log(self._sigmas_cpu[step_index] / (self._sigmas_cpu[step_index - 2] + 1e-9))
                r1 = torch.tensor(h_p1, device=sample.device, dtype=sample.dtype) / (h + 1e-9)
                r2 = torch.tensor(h_p2, device=sample.device, dtype=sample.dtype) / (h + 1e-9)

                # Hard Restart
                if r1 < 0.5 or r1 > 2.0 or r2 < 0.5 or r2 > 2.0:
                    res = phi_1 * denoised
                else:
                    phi_2, phi_3 = phi(2), phi(3)
                    denom = r2 - r1 + 1e-9
                    b3 = (phi_3 + r1 * phi_2) / (r2 * denom)
                    b2 = -(phi_3 + r2 * phi_2) / (r1 * denom)
                    b1 = phi_1 - b2 - b3
                    res = b1 * x0s[0] + b2 * x0s[1] + b3 * x0s[2]
            else:
                # Fallback to Euler or lower order
                res = phi_1 * denoised

            # Stable update in x-space
            if sigma_next == 0:
                x_next = denoised
            else:
                x_next = torch.exp(-h) * sample + h * res
            prev_sample = x_next

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
        from .scheduler_utils import add_noise_to_sample
        return add_noise_to_sample(original_samples, noise, self.sigmas, timesteps, self.timesteps)

    def __len__(self):
        return self.config.num_train_timesteps
