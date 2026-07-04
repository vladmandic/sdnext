# Copyright (C) 2026 Boogu Team.
#
# This file is adapted by Boogu Team from prior open-source scheduler work.
# Boogu-specific modifications include static/dynamic time-shift handling used
# by the released Boogu pipeline.
#
# Original work:
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team.
# All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        do_shift: bool = True,
        dynamic_time_shift: bool = True,
        time_shift_version: str = "v2",
        # seq_len is used to mirror training-side static time shift (when dynamic_time_shift=False)
        # In training, seq_len is the token count used to compute shift.
        seq_len: Optional[int] = None,
        # v1 linear mapping range (matches training defaults)
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        time_shift_v2_half_scaling_factor: float = 60.0,
    ):
        timesteps = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float32)[
            :-1
        ]

        self.timesteps = timesteps

        self._step_index = None
        self._begin_index = None
        self.time_shift_v2_scaling_factor = time_shift_v2_half_scaling_factor * 2

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

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self._timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # --- Helpers to mirror training-side shift logic ---
    @staticmethod
    def _get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def _time_shift_v1(t_np: np.ndarray, mu: float, sigma: float = 1.0) -> np.ndarray:
        # Matches training: t <- 1 - t; logistic transform; then t <- 1 - t
        eps = 1e-8
        t1 = 1.0 - t_np
        t1 = np.clip(t1, eps, 1.0 - eps)
        num = math.exp(mu)
        denom = num + np.power(1.0 / t1 - 1.0, sigma)
        y = num / denom
        out = 1.0 - y
        return out.astype(np.float32)

    @staticmethod
    def _time_shift_v2(t_np: np.ndarray, m: float) -> np.ndarray:
        # Matches training: t' = t / (m - m t + t)
        return (t_np / (m - m * t_np + t_np)).astype(np.float32)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: Union[str, torch.device] | None = None,
        timesteps: Optional[List[float]] = None,
        num_tokens: Optional[int] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        if timesteps is None:
            self.num_inference_steps = num_inference_steps
            t_arr = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[
                :-1
            ]  # Default
            # t_arr = np.linspace(0, 1, num_inference_steps, dtype=np.float32)[:-1]  # my

            # Apply training-consistent time shift only when requested
            if self.config.do_shift:
                # dynamic or static
                if self.config.dynamic_time_shift:
                    # dynamic: depend on per-sample token count
                    if self.config.time_shift_version == "v1":
                        # In training dynamic v1: mu is computed from tokens' linear map where
                        # tokens are approximately (H_lat//2)*(W_lat//2). We approximate this with num_tokens//4.
                        if num_tokens is not None and num_tokens > 0:
                            tokens_reduced = max(1, int(num_tokens) // 4)
                            lin = self._get_lin_function(
                                y1=self.config.base_shift, y2=self.config.max_shift
                            )
                            mu = lin(tokens_reduced)  ## 4096 for 1024x1024 resolution

                            t_arr = self._time_shift_v1(t_arr, mu, sigma=1.0)
                        # else: no-op if we lack num_tokens
                    elif self.config.time_shift_version == "v2":
                        # MUST remain identical to current behavior when v2 + dynamic=True
                        # m = sqrt(num_tokens) / 40; t' = t / (m - m t + t)
                        # When input resolution is 320 * 320, m = 1, when input resolution is 512 * 512, m = 1.6, when input resolution is 1024 * 1024, m = 3.2
                        if num_tokens is not None and num_tokens > 0:
                            m = (
                                float(np.sqrt(num_tokens))
                                / self.time_shift_v2_scaling_factor
                            )
                            t_arr = self._time_shift_v2(t_arr, m)
                        # else: no-op
                else:
                    # static: depend on seq_len configured at scheduler init
                    if self.config.time_shift_version == "v1":
                        if self.config.seq_len is not None and self.config.seq_len > 0:
                            lin = self._get_lin_function(
                                y1=self.config.base_shift, y2=self.config.max_shift
                            )
                            mu = lin(int(self.config.seq_len))
                            t_arr = self._time_shift_v1(t_arr, mu, sigma=1.0)
                            # ###################No dyn#######################
                            # print(f"time_shift_version: v1;  No self.config.dynamic_time_shift: {self.config.dynamic_time_shift}")
                            # print(f"t_arr: {t_arr}")
                            # ################################################

                    elif self.config.time_shift_version == "v2":
                        if self.config.seq_len is not None and self.config.seq_len > 0:
                            # training static v2 uses m = sqrt(seq_len) / 40
                            m = (
                                float(np.sqrt(self.config.seq_len))
                                / self.time_shift_v2_scaling_factor
                            )
                            t_arr = self._time_shift_v2(t_arr, m)

            timesteps = t_arr

        # ######################debug############################
        # print(f">> time_shift_version:  {self.config.time_shift_version}")
        # print(f">> timesteps:  {timesteps}")
        # print(f">> self.time_shift_v2_scaling_factor:  {self.time_shift_v2_scaling_factor}")
        # #######################################################

        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)
        _timesteps = torch.cat([timesteps, torch.ones(1, device=timesteps.device)])

        # ######################debug############################
        # print(f">> len _timesteps:  {len(_timesteps)}")
        # print(f">> _timesteps:  {_timesteps}")
        # #######################################################

        self.timesteps = timesteps
        self._timesteps = _timesteps
        self._step_index = None
        self._begin_index = None

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
        generator: Optional[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
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
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        t = self._timesteps[self.step_index]
        t_next = self._timesteps[self.step_index + 1]

        prev_sample = sample + (t_next - t) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
