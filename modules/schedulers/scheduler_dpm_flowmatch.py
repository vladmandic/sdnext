# Credits: @ukaprch <https://github.com/huggingface/diffusers/issues/9924>

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        import torchsde
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = [torch.randint(0, 2**63 - 1, []).item()]
        seed = [s.initial_seed() if isinstance(s, torch.Generator) else s for s in seed]
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [
            torchsde.BrownianInterval(
                t0=t0,
                t1=t1,
                size=w0.shape,
                dtype=w0.dtype,
                device=w0.device,
                entropy=s,
                tol=1e-6,
                pool_size=24,
                halfway_tree=True,
            )
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]

class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will use one BrownianTree per batch item, each
            with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

@dataclass
class FlowMatchDPMSolverMultistepSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor

class FlowMatchDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `DPMSolverMultistepScheduler` is a fast dedicated high-order solver for diffusion ODEs.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"scaled linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `2` or `3`. It is recommended to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling.
        algorithm_type (`str`, defaults to `dpmsolver++2M`):
            Algorithm type for the solver; can be `dpmsolver2`, `dpmsolver2A`, `dpmsolver++2M`, `dpmsolver++2S`, `dpmsolver++sde`, `dpmsolver++2Msde`,
            or `dpmsolver++3Msde`.
        solver_type (`str`, defaults to `midpoint`):
            Solver type for the second-order solver; can be `midpoint` or `heun`. The solver type slightly affects the
            sample quality, especially for a small number of steps. It is recommended to use `midpoint` solvers.
        sigma_schedule (`str`, *optional*, defaults to None (beta)): Sigma schedule to compute the `sigmas`. Optionally, we use
            the schedule "karras" introduced in the EDM paper (https://arxiv.org/abs/2206.00364). Other acceptable values are
            "exponential". The exponential schedule was incorporated in this model: https://huggingface.co/stabilityai/cosxl.
            Other acceptable values are "lambdas". The uniform-logSNR for step sizes proposed by Lu's DPM-Solver in the
            noise schedule during the sampling process. The sigmas and time steps are determined according to a sequence of `lambda(t)`.
            "betas" for step sizes in the noise schedule during the sampling process. Refer to [Beta
            Sampling is All You Need](https://huggingface.co/papers/2407.12173) for more information.
        use_noise_sampler for BrownianTreeNoiseSampler (only valid for `dpmsolver++2S`, `dpmsolver++sde`, `dpmsolver++2Msde`, or `dpmsolver++3Msde`.
            A noise sampler backed by a torchsde increasing the stability of convergence. Default strategy
            (random noise) has it jumping all over the place, but Brownian sampling is more stable. Utilizes the model generation seed provided.
        midpoint_ratio (`float`, *optional*, range: 0.4 to 0.6, default=0.5): Only valid for (`dpmsolver++sde`, `dpmsolver++2S`).
            Higher values may result in smoothing, more vivid colors and less noise at the expense of more detail and effect.
        s_noise (`float`, *optional*, defaults to 1.0): Sigma noise strength: range 0 - 1.1 (only valid for `dpmsolver++2S`, `dpmsolver++sde`,
            `dpmsolver++2Msde`, or `dpmsolver++3Msde`). The amount of additional noise to counteract loss of detail during sampling. A
            reasonable range is [1.000, 1.011]. Defaults to 1.0 from the original implementation.
        use_beta_sigmas: (`bool` defaults to False for FLUX and True for SD3). Based on original interpretation of using beta values for determining sigmas.
        use_dynamic_shifting (`bool` defaults to False for SD3 and True for FLUX). When `True`, shift is ignored.
        shift (`float`, defaults to 3.0): The shift value for the timestep schedule for SD3 when not using dynamic shifting.
        The remaining args are specific to Flux's dynamic shifting based on resolution.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        algorithm_type: str = "dpmsolver++2M",
        solver_type: str = "midpoint",
        sigma_schedule: Optional[str] = None,
        prediction_type: str = "flow_prediction",
        use_flow_sigmas: bool = True,
        shift: float = 3.0,
        midpoint_ratio: Optional[float] = 0.5,
        s_noise: Optional[float] = 1.0,
        use_noise_sampler: Optional[bool] = True,
        use_beta_sigmas: Optional[bool] = False,
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
    ):
        from installer import install
        install('torchsde==0.2.6', 'torchsde', quiet=True)
        try:
            import torchsde
        except Exception as e:
            raise ImportError("Failed to import torchsde. Please make sure it is installed correctly.") from e

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver2", "dpmsolver2A", "dpmsolver++2M", "dpmsolver++2S", "dpmsolver++sde", "dpmsolver++2Msde", "dpmsolver++3Msde"]:
            raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        if sigma_schedule not in [None, "karras", "exponential", "lambdas", "betas"]:
            raise NotImplementedError(f"{sigma_schedule} is not implemented for {self.__class__}")

        if beta_schedule not in ["linear", "scaled linear"]:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # setable values
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.h_last = None
        self.h_1 = None
        self.h_2 = None
        self.noise_sampler = None
        self._step_index = None
        self._begin_index = None
        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.model_outputs = [None] * solver_order

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def time_shift(self, mu: float, sigma: float, t: torch.FloatTensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[torch.Tensor] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.use_beta_sigmas = True
            self.num_inference_steps = num_inference_steps
            beta_start = self.config.beta_start
            beta_end = self.config.beta_end
            if self.config.trained_betas is not None:
                betas = torch.tensor(self.config.trained_betas, dtype=torch.float64)
            elif self.config.beta_schedule == "linear":
                betas = torch.linspace(beta_start, beta_end, self.config.num_train_timesteps, dtype=torch.float64)
            elif self.config.beta_schedule == "scaled linear":
                # this schedule is very specific to the latent diffusion model.
                betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.config.num_train_timesteps, dtype=torch.float64) ** 2
            else:
                raise NotImplementedError(f"{self.config.beta_schedule} is not implemented for {self.__class__}")
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
            del alphas_cumprod
            del alphas
            del betas
        elif self.use_beta_sigmas:
            num_inference_steps = len(sigmas)
            self.num_inference_steps = num_inference_steps
            beta_start = self.config.beta_start
            beta_end = self.config.beta_end
            if self.config.trained_betas is not None:
                betas = torch.tensor(self.config.trained_betas, dtype=torch.float64)
            elif self.config.beta_schedule == "linear":
                betas = torch.linspace(beta_start, beta_end, self.config.num_train_timesteps, dtype=torch.float64)
            elif self.config.beta_schedule == "scaled linear":
                # this schedule is very specific to the latent diffusion model.
                betas = torch.linspace(beta_start**0.5, beta_end**0.5, self.config.num_train_timesteps, dtype=torch.float64) ** 2
            else:
                raise NotImplementedError(f"{self.config.beta_schedule} is not implemented for {self.__class__}")
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
            del alphas_cumprod
            del alphas
            del betas
        else:
            num_inference_steps = len(sigmas)
            self.num_inference_steps = num_inference_steps

        if self.config.sigma_schedule == "exponential":
            if self.use_beta_sigmas:
                sigmas = np.flip(sigmas).copy()
                sigma_min = sigmas[-1]
                sigma_max = sigmas[0]
                sigmas = self._convert_to_exponential(sigma_min, sigma_max, num_inference_steps=num_inference_steps)
                OldRange = sigma_max - sigma_min
                NewRange = 1.0 - sigma_min
                sigmas = (((sigmas - sigma_min) * NewRange) / OldRange) + sigma_min
            else:
                sigma_min = sigmas[-1]
                sigma_max = sigmas[0]
                sigmas = self._convert_to_exponential(sigma_min, sigma_max, num_inference_steps=num_inference_steps)
        elif self.config.sigma_schedule == "karras":
            if self.use_beta_sigmas:
                sigmas = np.flip(sigmas).copy()
                sigma_min = sigmas[-1]
                sigma_max = sigmas[0]
                sigmas = self._convert_to_karras(sigma_min, sigma_max, num_inference_steps=num_inference_steps)
                OldRange = sigma_max - sigma_min
                NewRange = 1.0 - sigma_min
                sigmas = (((sigmas - sigma_min) * NewRange) / OldRange) + sigma_min
            else:
                sigma_min = sigmas[-1]
                sigma_max = sigmas[0]
                sigmas = self._convert_to_karras(sigma_min, sigma_max, num_inference_steps=num_inference_steps)
            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float64, device=device)
        elif self.config.sigma_schedule == "lambdas":
            if self.use_beta_sigmas:
                log_sigmas = np.log(sigmas)
                lambdas = np.flip(log_sigmas.copy())
                lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
                sigmas = np.exp(lambdas)
                sigma_min = sigmas[-1]
                sigma_max = sigmas[0]
                OldRange = sigma_max - sigma_min
                NewRange = 1.0 - sigma_min
                sigmas = (((sigmas - sigma_min) * NewRange) / OldRange) + sigma_min
                del lambdas
                del log_sigmas
            else:
                log_sigmas = np.log(sigmas)
                lambdas = log_sigmas.copy()
                lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
                sigmas = np.exp(lambdas)
                del lambdas
                del log_sigmas
            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float64, device=device)
        elif self.config.sigma_schedule == "betas":
            if self.use_beta_sigmas:
                sigmas = np.flip(sigmas).copy()
                sigma_min = sigmas[-1]
                sigma_max = sigmas[0]
                sigmas = self._convert_to_beta(sigma_min, sigma_max, num_inference_steps=num_inference_steps, device=device)
                OldRange = sigma_max - sigma_min
                NewRange = 1.0 - sigma_min
                sigmas = (((sigmas - sigma_min) * NewRange) / OldRange) + sigma_min
            else:
                sigmas = np.flip(sigmas).copy()
                sigma_min = sigmas[-1]
                sigmas = np.linspace(1.0, sigma_min, num_inference_steps)
                sigmas = torch.from_numpy(sigmas).to(dtype=torch.float64, device=device)
        else:
            if self.use_beta_sigmas:
                sigmas = np.flip(sigmas).copy()
                sigma_min = sigmas[-1]
                sigmas = np.linspace(1.0, sigma_min, num_inference_steps)
            sigmas = torch.from_numpy(sigmas).to(dtype=torch.float64, device=device)

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.h_last = None
        self.h_1 = None
        self.h_2 = None
        self.noise_sampler = None
        self.model_outputs = [None] * self.config.solver_order
        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(self, sigma_min, sigma_max, num_inference_steps, device: Union[str, torch.device] = None, alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""
        import scipy.stats
        sigmas = torch.Tensor(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps).astype(np.float64)
                ]
            ]
        ).to(dtype=torch.float64, device=device)
        return sigmas

    def _convert_to_lu(self, in_lambdas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """Constructs the noise schedule of Lu et al. (2022)."""

        lambda_min: float = in_lambdas[-1].item()
        lambda_max: float = in_lambdas[0].item()

        rho = 1.0  # 1.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = lambda_min ** (1 / rho)
        max_inv_rho = lambda_max ** (1 / rho)
        lambdas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return lambdas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, sigma_min, sigma_max, num_inference_steps) -> torch.Tensor:
        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    def _convert_to_exponential(self, sigma_min, sigma_max, num_inference_steps) -> torch.Tensor:
        sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps).exp()
        return sigmas

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchDPMSolverMultistepSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        if self.config.algorithm_type in ["dpmsolver2", "dpmsolver2A"]:
            pass
        else:
            # Flow Match needs to solve an integral of the data prediction model.
            sigma = self.sigmas[self.step_index]
            model_output = sample - sigma * model_output
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
            self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        if sample.dtype != model_output.dtype:
            sample = sample.to(model_output.dtype)

        if self.config.algorithm_type in ["dpmsolver2A", "dpmsolver++2S", "dpmsolver++sde", "dpmsolver++2Msde", "dpmsolver++3Msde"] and variance_noise is None:
            # Create a noise sampler if it hasn't been created yet
            if self.config.use_noise_sampler:
                if self.noise_sampler is None:
                    min_sigma, max_sigma = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
                    self.noise_sampler = BrownianTreeNoiseSampler(sample, min_sigma, max_sigma, generator)
            else:
                noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        elif self.config.algorithm_type in ["dpmsolver2A", "dpmsolver++2S", "dpmsolver++sde", "dpmsolver++2Msde", "dpmsolver++3Msde"]:
            noise = variance_noise.to(device=model_output.device, dtype=model_output.dtype)
        else:
            noise = None

        def sigma_fn(_t: torch.Tensor) -> torch.Tensor:
            return _t.neg().exp()
        def t_fn(_sigma: torch.Tensor) -> torch.Tensor:
            return _sigma.log().neg()
        sigma = self.sigmas[self.step_index]
        try:
            sigma_next = self.sigmas[self.step_index + 1]
        except Exception:
            sigma_next = self.sigmas[-1]
        sigma_prev = self.sigmas[self.step_index - 1]
        if self.config.algorithm_type == "dpmsolver2":
            if self.config.solver_order == 2:
                if sigma_next == 0:
                    # Euler method
                    model_output = sample - sigma * model_output
                    d = (sample - model_output) / sigma
                    dt = sigma_next - sigma
                    sample = sample + d * dt
                else:
                    # DPM-Solver2
                    sigma_mid = sigma.log().lerp(sigma_next.log(), 0.5).exp()

                    #using epsilon for new model output:
                    pred_original_sample = sample - sigma * model_output
                    # 2. Convert to an ODE derivative for 1st order
                    d = (sample - pred_original_sample) / sigma
                    # 3. delta timestep
                    dt = sigma_mid - sigma
                    x_2 = sample + d * dt

                    #using epsilon for new model output:
                    denoised_2 = x_2 - sigma_mid * model_output
                    # 2. Convert to an ODE derivative for 2nd order
                    d = (x_2 - denoised_2) / sigma_mid

                    # 3. delta timestep
                    dt = sigma_next - sigma
                    sample = sample + d * dt

                    del pred_original_sample
                    del denoised_2
                    del x_2
                    del d
        elif self.config.algorithm_type == "dpmsolver2A":
            if self.config.solver_order == 2:
                # get ancestral step
                sigma_from = sigma
                sigma_to = sigma_next
                su = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
                sd = (sigma_to**2 - su**2) ** 0.5
                if sd == 0:
                    # Euler method
                    model_output = sample - sigma * model_output
                    d = (sample - model_output) / sigma
                    dt = sd - sigma
                    sample = sample + d * dt
                else:
                    # DPM-Solver2A
                    sigma_mid = sigma.log().lerp(sd.log(), 0.5).exp()

                    #using epsilon for new model output:
                    model_output = sample - sigma * model_output
                    # 2. Convert to an ODE derivative for 1st order
                    d = (sample - model_output) / sigma
                    dt = sd - sigma
                    sample = sample + d * dt

                    #using epsilon for new model output:
                    pred_original_sample = sample - sigma * model_output
                    # 2. Convert to an ODE derivative for 1st order
                    d = (sample - pred_original_sample) / sigma
                    # 3. delta timestep
                    dt_1 = sigma_mid - sigma
                    x_2 = sample + d * dt_1

                    #using epsilon for new model output:
                    denoised_2 = x_2 - sigma_mid * model_output
                    # 2. Convert to an ODE derivative for 2nd order
                    d_2 = (x_2 - denoised_2) / sigma_mid

                    # 3. delta timestep
                    dt_2 = sd - sigma_mid
                    sample = sample + d_2 * dt_2

                    if self.config.use_noise_sampler:
                        sample = sample + self.noise_sampler(sigma, sigma_next) * self.config.s_noise * su
                    else:
                        sample = sample + noise * self.config.s_noise * su

                    del pred_original_sample
                    del denoised_2
                    del x_2
                    del d
        elif self.config.algorithm_type == "dpmsolver++2M":
            if self.config.solver_order == 2:
                t, t_next = t_fn(sigma), t_fn(sigma_next)
                h = t_next - t
                if self.model_outputs[-2] is None or sigma_next == 0:
                    sample = (sigma_fn(t_next) / sigma_fn(t)) * sample - (-h).expm1() * model_output
                else:
                    # DPM-Solver++(2M)
                    h_last = t - t_fn(sigma_prev)
                    r = h_last / h
                    denoised_d = (1 + 1 / (2 * r)) * model_output - (1 / (2 * r)) * self.model_outputs[-2]
                    sample = (sigma_fn(t_next) / sigma_fn(t)) * sample - (-h).expm1() * denoised_d
                    del denoised_d
        elif self.config.algorithm_type == "dpmsolver++2S":
            if self.config.solver_order == 2:
                # get ancestral step
                sigma_from = sigma
                sigma_to = sigma_next
                su = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
                sd = (sigma_to**2 - su**2) ** 0.5
                if sd == 0:
                    # Euler method
                    d = (sample - model_output) / sigma
                    dt = sd - sigma
                    sample = sample + d * dt
                else:
                    # DPM-Solver++(2S)
                    t, t_next = t_fn(sigma), t_fn(sd)
                    r = self.config.midpoint_ratio
                    h = t_next - t
                    s = t + r * h

                    # Euler method
                    d = (sample - model_output) / sigma
                    dt = sd - sigma
                    sample = sample + d * dt

                    x_2 = (sigma_fn(s) / sigma_fn(t)) * sample - (-h * r).expm1() * model_output

                    #using epsilon for new model output:
                    denoised_2 = x_2 - sigma_fn(s) * model_output
                    # 2. Convert to an ODE derivative for 2nd order
                    d = (x_2 - denoised_2) / sigma_fn(s)
                    dt = sd - sigma_next
                    sample = sample + d * dt

                    del x_2
                    del denoised_2
                    del d
                # Noise addition
                if sigma_next > 0:
                    if self.config.use_noise_sampler:
                        sample = sample + self.noise_sampler(sigma, sigma_next) * self.config.s_noise * su
                    else:
                        sample = sample + noise * self.config.s_noise * su
        elif self.config.algorithm_type == "dpmsolver++sde":
            if self.config.solver_order == 2:
                if sigma_next == 0:
                    # Euler method
                    d = (sample - model_output) / sigma
                    dt = sigma_next - sigma
                    sample = sample + d * dt
                else:
                    # DPM-Solver++(SDE)
                    t, t_next = t_fn(sigma), t_fn(sigma_next)
                    r = self.config.midpoint_ratio
                    h = t_next - t
                    s = t + r * h

                    # Euler method
                    d = (sample - model_output) / sigma
                    dt = sigma_next - sigma
                    sample = sample + d * dt

                    # Step 1
                    # get ancestral step
                    sigma_from = sigma_fn(t)
                    sigma_to = sigma_fn(s)
                    su = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
                    sd = (sigma_to**2 - su**2) ** 0.5

                    # Euler method
                    d = (sample - model_output) / sigma
                    dt = sd - sigma
                    sample = sample + d * dt

                    s_ = t_fn(sd)
                    x_2 = (sigma_fn(s_) / sigma_fn(t)) * sample - (t - s_).expm1() * model_output
                    if self.config.use_noise_sampler:
                        x_2 = x_2 + self.noise_sampler(sigma_fn(t), sigma_fn(s)) * self.config.s_noise * su
                    else:
                        x_2 = x_2 + noise * self.config.s_noise * su

                    # Step 2
                    # get ancestral step
                    sigma_from = sigma_fn(t)
                    sigma_to = sigma_fn(t_next)
                    su = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
                    sd = (sigma_to**2 - su**2) ** 0.5

                    #using epsilon for new model output:
                    denoised_2 = x_2 - sigma_fn(s) * model_output
                    # 2. Convert to an ODE derivative for 2nd order
                    d = (x_2 - denoised_2) / sigma_fn(s)
                    dt = sd - sigma_next
                    sample = sample + d * dt

                    if self.config.use_noise_sampler:
                        sample = sample + self.noise_sampler(sigma_fn(t), sigma_fn(t_next)) * self.config.s_noise * su
                    else:
                        sample = sample + noise * self.config.s_noise * su
                    del x_2
                    del denoised_2
                    del d
        elif self.config.algorithm_type == "dpmsolver++2Msde":
            if self.config.solver_order == 2:
                if sigma_next == 0:
                    sample = model_output
                else:
                    # DPM-Solver++(2M) SDE
                    t, s = -sigma.log(), -sigma_next.log()
                    h = s - t
                    eta_h = h * 1

                    # 3. Delta timestep
                    dt = sigma_next - sigma
                    sample = sample + model_output * dt

                    sample = sigma_next / sigma * (-eta_h).exp() * sample + (-h - eta_h).expm1().neg() * model_output

                    if self.model_outputs[-2] is not None:
                        r = self.h_last / h
                        if self.solver_type == 'heun':
                            sample = sample + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (model_output - self.model_outputs[-2])
                        elif self.solver_type == 'midpoint':
                            sample = sample + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (model_output - self.model_outputs[-2])

                    if self.config.use_noise_sampler:
                        sample = sample + self.noise_sampler(sigma, sigma_next) * sigma_next * (-2 * eta_h).expm1().neg().sqrt() * self.config.s_noise
                    else:
                        sample = sample + noise * sigma_next * (-2 * eta_h).expm1().neg().sqrt() * self.config.s_noise

                    self.h_last = h
        elif self.config.algorithm_type == "dpmsolver++3Msde":
            if self.config.solver_order == 3:
                if sigma_next == 0:
                    sample = model_output
                else:
                    # DPM-Solver++(3M) SDE
                    t, s = -sigma.log(), -sigma_next.log()
                    h = s - t
                    h_eta = h * 2

                    # 3. Delta timestep
                    dt = sigma_next - sigma
                    sample = sample + model_output * dt

                    sample = torch.exp(-h_eta) * sample + (-h_eta).expm1().neg() * model_output

                    if self.h_2 is not None:
                        r0 = self.h_1 / h
                        r1 = self.h_2 / h
                        d1_0 = (model_output - self.model_outputs[-2]) / r0
                        d1_1 = (self.model_outputs[-2] - self.model_outputs[-3]) / r1
                        d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                        d2 = (d1_0 - d1_1) / (r0 + r1)
                        phi_2 = h_eta.neg().expm1() / h_eta + 1
                        phi_3 = phi_2 / h_eta - 0.5
                        sample = sample + phi_2 * d1 - phi_3 * d2
                        del d1_0
                        del d1_1
                        del d1
                        del d2
                        del phi_2
                        del phi_3
                    elif self.h_1 is not None:
                        r = self.h_1 / h
                        d = (model_output - self.model_outputs[-2]) / r
                        phi_2 = h_eta.neg().expm1() / h_eta + 1
                        sample = sample + phi_2 * d
                        del d
                        del phi_2

                    if self.config.use_noise_sampler:
                        sample = sample + self.noise_sampler(sigma, sigma_next) * sigma_next * (-2 * h).expm1().neg().sqrt() * self.config.s_noise
                    else:
                        sample = sample + noise * sigma_next * (-2 * h).expm1().neg().sqrt() * self.config.s_noise

                    self.h_2 = self.h_1
                    self.h_1 = h
            if not self.config.use_noise_sampler and noise is not None:
                del noise
        prev_sample = sample

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        torch.cuda.empty_cache()

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDPMSolverMultistepSchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def __len__(self):
        return self.config.num_train_timesteps
