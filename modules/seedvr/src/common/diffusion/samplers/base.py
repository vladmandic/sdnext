# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Sampler base class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import torch
from tqdm import tqdm

from ..schedules.base import Schedule
from ..timesteps.base import SamplingTimesteps
from ..types import PredictionType, SamplingDirection
from ..utils import assert_schedule_timesteps_compatible


@dataclass
class SamplerModelArgs:
    x_t: torch.Tensor
    t: torch.Tensor
    i: int


class Sampler(ABC):
    """
    Samplers are ODE/SDE solvers.
    """

    def __init__(
        self,
        schedule: Schedule,
        timesteps: SamplingTimesteps,
        prediction_type: PredictionType,
        return_endpoint: bool = True,
    ):
        assert_schedule_timesteps_compatible(
            schedule=schedule,
            timesteps=timesteps,
        )
        self.schedule = schedule
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.return_endpoint = return_endpoint

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor:
        """
        Generate a new sample given the the intial sample x and score function f.
        """

    def get_next_timestep(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the next sample timestep.
        Support multiple different timesteps t in a batch.
        If no more steps, return out of bound value -1 or T+1.
        """
        T = self.timesteps.T
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1
        bound = -1 if self.timesteps.direction == SamplingDirection.backward else T + 1

        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        s = s.where(next_idx < steps, bound)
        return s

    def get_endpoint(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get to the endpoint of the probability flow.
        """
        x_0, x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        return x_0 if self.timesteps.direction == SamplingDirection.backward else x_T

    def get_progress_bar(self):
        """
        Get progress bar for sampling.
        """
        return tqdm(
            iterable=range(len(self.timesteps) - (0 if self.return_endpoint else 1)),
            dynamic_ncols=True,
            desc=self.__class__.__name__,
        )
