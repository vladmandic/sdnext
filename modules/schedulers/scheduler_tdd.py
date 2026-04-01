from typing import Union, List, Optional, Tuple
import numpy as np
import torch
from diffusers.utils import  deprecate, logging
from diffusers.configuration_utils import register_to_config
from diffusers import DPMSolverSinglestepScheduler
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
# from diffusers.schedulers.scheduling_tcd import *
# from diffusers.schedulers.scheduling_dpmsolver_singlestep import *


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TDDScheduler(DPMSolverSinglestepScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        solver_order: int = 1,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = False,
        use_karras_sigmas: Optional[bool] = False,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        tdd_train_step: int = 250,
        special_jump: bool = False,
        t_l: int = -1,
        use_flow_sigmas: bool = False,
    ):
        self.t_l = t_l
        self.special_jump = special_jump
        self.tdd_train_step = tdd_train_step
        if algorithm_type == "dpmsolver":
            deprecation_message = "algorithm_type `dpmsolver` is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead"
            deprecate("algorithm_types=dpmsolver", "1.0.0", deprecation_message)

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Currently we only support VP-type noise schedule
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver", "dpmsolver++"]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(f"{algorithm_type} does is not implemented for {self.__class__}")
        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} does is not implemented for {self.__class__}")

        if algorithm_type != "dpmsolver++" and final_sigmas_type == "zero":
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please chooose `sigma_min` instead."
            )

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.sample = None
        self.order_list = self.get_order_list(num_train_timesteps)
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        #original_steps = self.config.original_inference_steps
        if True:
            original_steps=self.tdd_train_step
            k = 1000 / original_steps
            tcd_origin_timesteps = np.asarray(list(range(1, int(original_steps) + 1))) * k - 1
        else:
            tcd_origin_timesteps = np.asarray(list(range(0, int(self.config.num_train_timesteps))))
        # TCD Inference Steps Schedule
        tcd_origin_timesteps = tcd_origin_timesteps[::-1].copy()
        # Select (approximately) evenly spaced indices from tcd_origin_timesteps.
        inference_indices = np.linspace(0, len(tcd_origin_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        timesteps = tcd_origin_timesteps[inference_indices]
        if self.special_jump:
            if self.tdd_train_step == 50:
                pass
            elif self.tdd_train_step == 250:
                if num_inference_steps == 5:
                    timesteps = np.array([999., 875., 751., 499., 251.])
                elif num_inference_steps == 6:
                    timesteps = np.array([999., 875., 751., 627., 499., 251.])
                elif num_inference_steps == 7:
                    timesteps = np.array([999., 875., 751., 627., 499., 375., 251.])

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        if self.config.use_karras_sigmas:
            log_sigmas = np.log(sigmas)
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round()
        else:
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f" `final_sigmas_type` must be one of `sigma_min` or `zero`, but got {self.config.final_sigmas_type}"
            )
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device=device)

        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self.model_outputs = [None] * self.config.solver_order
        self.sample = None

        if not self.config.lower_order_final and num_inference_steps % self.config.solver_order != 0:
            logger.warning(
                "Changing scheduler {self.config} to have `lower_order_final` set to True to handle uneven amount of inference steps. Please make sure to always use an even number of `num_inference steps when using `lower_order_final=False`."
            )
            self.register_to_config(lower_order_final=True)

        if not self.config.lower_order_final and self.config.final_sigmas_type == "zero":
            logger.warning(
                " `last_sigmas_type='zero'` is not supported for `lower_order_final=False`. Changing scheduler {self.config} to have `lower_order_final` set to True."
            )
            self.register_to_config(lower_order_final=True)

        self.order_list = self.get_order_list(num_inference_steps)

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def set_timesteps_s(self, eta: float = 0.0):
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        num_inference_steps = self.num_inference_steps
        device = self.timesteps.device
        if True:
            original_steps=self.tdd_train_step
            k = 1000 / original_steps
            tcd_origin_timesteps = np.asarray(list(range(1, int(original_steps) + 1))) * k - 1
        else:
            tcd_origin_timesteps = np.asarray(list(range(0, int(self.config.num_train_timesteps))))
        # TCD Inference Steps Schedule
        tcd_origin_timesteps = tcd_origin_timesteps[::-1].copy()
        # Select (approximately) evenly spaced indices from tcd_origin_timesteps.
        inference_indices = np.linspace(0, len(tcd_origin_timesteps), num=num_inference_steps, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        timesteps = tcd_origin_timesteps[inference_indices]
        if self.special_jump:
            if self.tdd_train_step == 50:
                timesteps = np.array([999., 879., 759., 499., 259.])
            elif self.tdd_train_step == 250:
                if num_inference_steps == 5:
                    timesteps = np.array([999., 875., 751., 499., 251.])
                elif num_inference_steps == 6:
                    timesteps = np.array([999., 875., 751., 627., 499., 251.])
                elif num_inference_steps == 7:
                    timesteps = np.array([999., 875., 751., 627., 499., 375., 251.])

        timesteps_s = np.floor((1 - eta) * timesteps).astype(np.int64)

        sigmas_s = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        if self.config.use_karras_sigmas:
            pass
        else:
            sigmas_s = np.interp(timesteps_s, np.arange(0, len(sigmas_s)), sigmas_s)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f" `final_sigmas_type` must be one of `sigma_min` or `zero`, but got {self.config.final_sigmas_type}"
            )

        sigmas_s = np.concatenate([sigmas_s, [sigma_last]]).astype(np.float32)
        self.sigmas_s = torch.from_numpy(sigmas_s).to(device=device)
        self.timesteps_s = torch.from_numpy(timesteps_s).to(device=device, dtype=torch.int64)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        if self.step_index == 0:
            self.set_timesteps_s(eta)

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        order = self.order_list[self.step_index]

        #  For img2img denoising might start with order>1 which is not possible
        #  In this case make sure that the first two steps are both order=1
        while self.model_outputs[-order] is None:
            order -= 1

        # For single-step solvers, we use the initial value at each time with order = 1.
        if order == 1:
            self.sample = sample

        prev_sample = self.singlestep_dpm_solver_update(self.model_outputs, sample=self.sample, order=order)

        if eta > 0:
            if self.step_index != self.num_inference_steps - 1:

                alpha_prod_s = self.alphas_cumprod[self.timesteps_s[self.step_index + 1]]
                alpha_prod_t_prev = self.alphas_cumprod[self.timesteps[self.step_index + 1]]

                noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=prev_sample.dtype
                )
                prev_sample = (alpha_prod_t_prev / alpha_prod_s).sqrt() * prev_sample + (
                    1 - alpha_prod_t_prev / alpha_prod_s
                ).sqrt() * noise

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        sigma_t, sigma_s = self.sigmas_s[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        return x_t

    def singlestep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas_s[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s1, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m1, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s1) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s1) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s1) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s1) * sample
                    - (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        return x_t

    def singlestep_dpm_solver_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        order: int = None,
        **kwargs,
    ) -> torch.FloatTensor:
        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing`sample` as a required keyward argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError(" missing `order` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if order == 1:
            return self.dpm_solver_first_order_update(model_output_list[-1], sample=sample)
        elif order == 2:
            return self.singlestep_dpm_solver_second_order_update(model_output_list, sample=sample)
        else:
            raise ValueError(f"Order must be 1, 2, got {order}")

    def convert_model_output(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.config.algorithm_type == "dpmsolver++":
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned_range"]:
                    model_output = model_output[:, :3]
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverSinglestepScheduler."
                )

            if self.step_index <= self.t_l:
                if self.config.thresholding:
                    x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.config.algorithm_type == "dpmsolver":
            if self.config.prediction_type == "epsilon":
                # DPM-Solver and DPM-Solver++ only need the "mean" output.
                if self.config.variance_type in ["learned_range"]:
                    model_output = model_output[:, :3]
                return model_output
            elif self.config.prediction_type == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the DPMSolverSinglestepScheduler."
                )
