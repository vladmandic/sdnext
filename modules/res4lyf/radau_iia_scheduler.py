
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput


# pylint: disable=no-member
class RadauIIAScheduler(SchedulerMixin, ConfigMixin):
    """
    RadauIIAScheduler: Fully implicit Runge-Kutta integrators.
    Supports variants with 2, 3, 5, 7, 9, 11 stages.
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
        variant: str = "radau_iia_3s",  # 2s to 11s variants
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
        v = self.config.variant
        if v == "radau_iia_2s":
            a, b, c = [[5 / 12, -1 / 12], [3 / 4, 1 / 4]], [3 / 4, 1 / 4], [1 / 3, 1]
        elif v == "radau_iia_3s":
            r6 = 6**0.5
            a = [[11 / 45 - 7 * r6 / 360, 37 / 225 - 169 * r6 / 1800, -2 / 225 + r6 / 75], [37 / 225 + 169 * r6 / 1800, 11 / 45 + 7 * r6 / 360, -2 / 225 - r6 / 75], [4 / 9 - r6 / 36, 4 / 9 + r6 / 36, 1 / 9]]
            b, c = [4 / 9 - r6 / 36, 4 / 9 + r6 / 36, 1 / 9], [2 / 5 - r6 / 10, 2 / 5 + r6 / 10, 1]
        elif v == "radau_iia_5s":
            a = [
                [0.07299886, -0.02673533, 0.01867693, -0.01287911, 0.00504284],
                [0.15377523, 0.14621487, -0.03644457, 0.02123306, -0.00793558],
                [0.14006305, 0.29896713, 0.16758507, -0.03396910, 0.01094429],
                [0.14489431, 0.27650007, 0.32579792, 0.12875675, -0.01570892],
                [0.14371356, 0.28135602, 0.31182652, 0.22310390, 0.04000000],
            ]
            b = [0.14371356, 0.28135602, 0.31182652, 0.22310390, 0.04]
            c = [0.05710420, 0.27684301, 0.58359043, 0.86024014, 1.0]
        elif v == "radau_iia_7s":
            a = [
                [0.03754626, -0.01403933, 0.01035279, -0.00815832, 0.00638841, -0.00460233, 0.00182894],
                [0.08014760, 0.08106206, -0.02123799, 0.01400029, -0.01023419, 0.00715347, -0.00281264],
                [0.07206385, 0.17106835, 0.10961456, -0.02461987, 0.01476038, -0.00957526, 0.00367268],
                [0.07570513, 0.15409016, 0.22710774, 0.11747819, -0.02381083, 0.01270999, -0.00460884],
                [0.07391234, 0.16135561, 0.20686724, 0.23700712, 0.10308679, -0.01885414, 0.00585890],
                [0.07470556, 0.15830722, 0.21415342, 0.21987785, 0.19875212, 0.06926550, -0.00811601],
                [0.07449424, 0.15910212, 0.21235189, 0.22355491, 0.19047494, 0.11961374, 0.02040816],
            ]
            b = [0.07449424, 0.15910212, 0.21235189, 0.22355491, 0.19047494, 0.11961374, 0.02040816]
            c = [0.02931643, 0.14807860, 0.33698469, 0.55867152, 0.76923386, 0.92694567, 1.0]
        elif v == "radau_iia_9s":
            a = [
                [0.02278838, -0.00858964, 0.00645103, -0.00525753, 0.00438883, -0.00365122, 0.00294049, -0.00214927, 0.00085884],
                [0.04890795, 0.05070205, -0.01352381, 0.00920937, -0.00715571, 0.00574725, -0.00454258, 0.00328816, -0.00130907],
                [0.04374276, 0.10830189, 0.07291957, -0.01687988, 0.01070455, -0.00790195, 0.00599141, -0.00424802, 0.00167815],
                [0.04624924, 0.09656073, 0.15429877, 0.08671937, -0.01845164, 0.01103666, -0.00767328, 0.00522822, -0.00203591],
                [0.04483444, 0.10230685, 0.13821763, 0.18126393, 0.09043360, -0.01808506, 0.01019339, -0.00640527, 0.00242717],
                [0.04565876, 0.09914547, 0.14574704, 0.16364828, 0.18594459, 0.08361326, -0.01580994, 0.00813825, -0.00291047],
                [0.04520060, 0.10085371, 0.14194224, 0.17118947, 0.16978339, 0.16776829, 0.06707903, -0.01179223, 0.00360925],
                [0.04541652, 0.10006040, 0.14365284, 0.16801908, 0.17556077, 0.15588627, 0.12889391, 0.04281083, -0.00493457],
                [0.04535725, 0.10027665, 0.14319335, 0.16884698, 0.17413650, 0.15842189, 0.12359469, 0.07382701, 0.01234568],
            ]
            b = [0.04535725, 0.10027665, 0.14319335, 0.16884698, 0.17413650, 0.15842189, 0.12359469, 0.07382701, 0.01234568]
            c = [0.01777992, 0.09132361, 0.21430848, 0.37193216, 0.54518668, 0.71317524, 0.85563374, 0.95536604, 1.0]
        elif v == "radau_iia_11s":
            a = [
                [0.01528052, -0.00578250, 0.00438010, -0.00362104, 0.00309298, -0.00267283, 0.00230509, -0.00195565, 0.00159387, -0.00117286, 0.00046993],
                [0.03288398, 0.03451351, -0.00928542, 0.00641325, -0.00509546, 0.00424609, -0.00358767, 0.00300683, -0.00243267, 0.00178278, -0.00071315],
                [0.02933250, 0.07416243, 0.05114868, -0.01200502, 0.00777795, -0.00594470, 0.00480266, -0.00392360, 0.00312733, -0.00227314, 0.00090638],
                [0.03111455, 0.06578995, 0.10929963, 0.06381052, -0.01385359, 0.00855744, -0.00630764, 0.00491336, -0.00381400, 0.00273343, -0.00108397],
                [0.03005269, 0.07011285, 0.09714692, 0.13539160, 0.07147108, -0.01471024, 0.00873319, -0.00619941, 0.00459164, -0.00321333, 0.00126286],
                [0.03072807, 0.06751926, 0.10334060, 0.12083526, 0.15032679, 0.07350932, -0.01451288, 0.00829665, -0.00561283, 0.00376623, -0.00145771],
                [0.03029202, 0.06914472, 0.09972096, 0.12801064, 0.13493180, 0.15289670, 0.06975993, -0.01327455, 0.00725877, -0.00448439, 0.00168785],
                [0.03056654, 0.06813851, 0.10188107, 0.12403361, 0.14211432, 0.13829395, 0.14289135, 0.06052636, -0.01107774, 0.00559867, -0.00198773],
                [0.03040663, 0.06871881, 0.10066096, 0.12619527, 0.13848876, 0.14450774, 0.13065189, 0.12111401, 0.04655548, -0.00802620, 0.00243764],
                [0.03048412, 0.06843925, 0.10124185, 0.12518732, 0.14011843, 0.14190387, 0.13500343, 0.11262870, 0.08930604, 0.02896966, -0.00331170],
                [0.03046255, 0.06851684, 0.10108155, 0.12546269, 0.13968067, 0.14258278, 0.13393354, 0.11443306, 0.08565881, 0.04992304, 0.00826446],
            ]
            b = [0.03046255, 0.06851684, 0.10108155, 0.12546269, 0.13968067, 0.14258278, 0.13393354, 0.11443306, 0.08565881, 0.04992304, 0.00826446]
            c = [0.01191761, 0.06173207, 0.14711145, 0.26115968, 0.39463985, 0.53673877, 0.67594446, 0.80097892, 0.90171099, 0.96997097, 1.0]
        else:
            raise ValueError(f"Unknown variant: {v}")
        return np.array(a), np.array(b), np.array(c)

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
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        if isinstance(schedule_timesteps, torch.Tensor):
            schedule_timesteps = schedule_timesteps.detach().cpu().numpy()

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.detach().cpu().numpy()

        return np.abs(schedule_timesteps - timestep).argmin().item()

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        if self._step_index is None:
            self._init_step_index(timestep)
        if self.config.prediction_type == "flow_prediction":
            return sample
        sigma = self.sigmas[self._step_index]
        return sample / ((sigma**2 + 1) ** 0.5)



    def _init_step_index(self, timestep):
        if self._step_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)

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
            denoised = sample - sigma_t * model_output if getattr(self.config, "prediction_type", "epsilon") == "epsilon" else model_output
            prev_sample = denoised
            self._step_index += 1
            if not return_dict:
                return (prev_sample,)
            return SchedulerOutput(prev_sample=prev_sample)

        h = sigma_next - sigma_curr
        sigma_t = self.sigmas[self._step_index]

        prediction_type = getattr(self.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            denoised = sample - sigma_t * model_output
        elif self.config.prediction_type == "v_prediction":
            alpha_t = 1 / (sigma_t**2 + 1) ** 0.5
            sigma_actual = sigma_t * alpha_t
            denoised = alpha_t * sample - sigma_actual * model_output
        elif prediction_type == "flow_prediction":
            denoised = sample - sigma_t * model_output
        elif prediction_type == "sample":
            denoised = model_output
        else:
            raise ValueError(f"prediction_type error: {getattr(self.config, 'prediction_type', 'epsilon')}")

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
