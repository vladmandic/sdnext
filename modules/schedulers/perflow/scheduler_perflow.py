# Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


class Time_Windows():
    def __init__(self, t_initial=1, t_terminal=0, num_windows=4, precision=1./1000) -> None:
        assert t_terminal < t_initial
        time_windows = [ 1.*i/num_windows for i in range(1, num_windows+1)][::-1]

        self.window_starts = time_windows                      # [1.0, 0.75, 0.5, 0.25]
        self.window_ends = time_windows[1:] + [t_terminal]     # [0.75, 0.5, 0.25, 0]
        self.precision = precision

    def get_window(self, tp):
        idx = 0
        # robust to numerical error; e.g, (0.6+1/10000) belongs to [0.6, 0.3)
        while (tp-0.1*self.precision) <= self.window_ends[idx]:
            idx += 1
        return self.window_starts[idx], self.window_ends[idx]

    def lookup_window(self, timepoint):
        if timepoint.dim() == 0:
            t_start, t_end = self.get_window(timepoint)
            t_start = torch.ones_like(timepoint) * t_start
            t_end = torch.ones_like(timepoint) * t_end
        else:
            t_start = torch.zeros_like(timepoint)
            t_end = torch.zeros_like(timepoint)
            bsz = timepoint.shape[0]
            for i in range(bsz):
                tp = timepoint[i]
                ts, te = self.get_window(tp)
                t_start[i] = ts
                t_end[i] = te
        return t_start, t_end


@dataclass
class PeRFlowSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)



class PeRFlowScheduler(SchedulerMixin, ConfigMixin):
    """
    `ReFlowScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        prediction_type (`str`, defaults to `epsilon`, *optional*)
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = False,
        prediction_type: str = "ddim_eps",
        t_noise: float = 1,
        t_clean: float = 0,
        num_time_windows = 4,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        self.time_windows = Time_Windows(t_initial=t_noise, t_terminal=t_clean,
                                         num_windows=num_time_windows,
                                         precision=1./num_train_timesteps)

        assert prediction_type in ["ddim_eps", "diff_eps", "velocity"]


    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor: # pylint: disable=unused-argument
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample


    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        if num_inference_steps < self.config.num_time_windows: # pylint: disable=no-member
            num_inference_steps = self.config.num_time_windows # pylint: disable=no-member
            print(f"### We recommend a num_inference_steps not less than num_time_windows. It's set as {self.config.num_time_windows}.") # pylint: disable=no-member

        timesteps = []
        for i in range(self.config.num_time_windows): # pylint: disable=no-member
            if i < num_inference_steps%self.config.num_time_windows: # pylint: disable=no-member
                num_steps_cur_win = num_inference_steps//self.config.num_time_windows+1 # pylint: disable=no-member
            else:
                num_steps_cur_win = num_inference_steps//self.config.num_time_windows # pylint: disable=no-member

            t_s = self.time_windows.window_starts[i]
            t_e = self.time_windows.window_ends[i]
            timesteps_cur_win = np.linspace(t_s, t_e, num=num_steps_cur_win, endpoint=False)
            timesteps.append(timesteps_cur_win)

        timesteps = np.concatenate(timesteps)

        self.timesteps = torch.from_numpy( # pylint: disable=attribute-defined-outside-init
            (timesteps*self.config.num_train_timesteps).astype(np.int64) # pylint: disable=no-member,
        ).to(device)

    def get_window_alpha(self, timepoints):
        time_windows = self.time_windows
        num_train_timesteps = self.config.num_train_timesteps # pylint: disable=no-member

        t_win_start, t_win_end = time_windows.lookup_window(timepoints)
        t_win_len = t_win_end - t_win_start
        t_interval = timepoints - t_win_start # NOTE: negative value

        idx_start = (t_win_start*num_train_timesteps - 1 ).long()
        alphas_cumprod_start = self.alphas_cumprod[idx_start]

        idx_end = torch.clamp( (t_win_end*num_train_timesteps - 1 ).long(), min=0)
        alphas_cumprod_end = self.alphas_cumprod[idx_end]

        alpha_cumprod_s_e = alphas_cumprod_start / alphas_cumprod_end
        gamma_s_e = alpha_cumprod_s_e ** 0.5

        return t_win_start, t_win_end, t_win_len, t_interval, gamma_s_e, alphas_cumprod_start, alphas_cumprod_end

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[PeRFlowSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.PeRFlowSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.PeRFlowSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.PeRFlowSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """

        if self.config.prediction_type == "ddim_eps": # pylint: disable=no-member
            pred_epsilon = model_output
            t_c = timestep / self.config.num_train_timesteps # pylint: disable=no-member
            t_s, t_e, _, c_to_s, _, alphas_cumprod_start, alphas_cumprod_end = self.get_window_alpha(t_c)

            lambda_s = (alphas_cumprod_end / alphas_cumprod_start)**0.5
            eta_s = (1-alphas_cumprod_end)**0.5 - ( alphas_cumprod_end / alphas_cumprod_start * (1-alphas_cumprod_start) )**0.5

            lambda_t =  ( lambda_s * (t_e - t_s) ) / ( lambda_s *(t_c - t_s) + (t_e - t_c) )
            eta_t = ( eta_s * (t_e - t_c) ) / ( lambda_s *(t_c - t_s) + (t_e - t_c) )

            pred_win_end = lambda_t * sample + eta_t * pred_epsilon
            pred_velocity = (pred_win_end - sample) / (t_e - (t_s + c_to_s))

        elif self.config.prediction_type == "diff_eps": # pylint: disable=no-member
            pred_epsilon = model_output
            t_c = timestep / self.config.num_train_timesteps # pylint: disable=no-member
            t_s, t_e, _, c_to_s, gamma_s_e, _, _ = self.get_window_alpha(t_c)

            lambda_s = 1 / gamma_s_e
            eta_s = -1 * ( 1- gamma_s_e**2)**0.5 / gamma_s_e

            lambda_t =  ( lambda_s * (t_e - t_s) ) / ( lambda_s *(t_c - t_s) + (t_e - t_c) )
            eta_t = ( eta_s * (t_e - t_c) ) / ( lambda_s *(t_c - t_s) + (t_e - t_c) )

            pred_win_end = lambda_t * sample + eta_t * pred_epsilon
            pred_velocity = (pred_win_end - sample) / (t_e - (t_s + c_to_s))

        elif self.config.prediction_type == "velocity": # pylint: disable=no-member
            pred_velocity = model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `velocity`." # pylint: disable=no-member
            )

        # get dt
        idx = torch.argwhere(torch.where(self.timesteps==timestep, 1,0))
        prev_step = self.timesteps[idx+1] if (idx+1)<len(self.timesteps) else 0
        dt = (prev_step - timestep) / self.config.num_train_timesteps # pylint: disable=no-member
        dt = dt.to(sample.device, sample.dtype)

        prev_sample = sample + dt * pred_velocity

        if not return_dict:
            return (prev_sample,)
        return PeRFlowSchedulerOutput(prev_sample=prev_sample, pred_original_sample=None)


    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device) - 1   # indexing from 0

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps # pylint: disable=no-member
