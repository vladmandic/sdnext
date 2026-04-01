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
Utility functions for creating schedules and samplers from config.
"""

import torch

from .samplers.base import Sampler
from .samplers.euler import EulerSampler
from .schedules.base import Schedule
from .schedules.lerp import LinearInterpolationSchedule
from .timesteps.base import SamplingTimesteps
from .timesteps.sampling.trailing import UniformTrailingSamplingTimesteps


def create_schedule_from_config(
    config,
) -> Schedule:
    """
    Create a schedule from configuration.
    """
    if config.type == "lerp":
        return LinearInterpolationSchedule(T=config.get("T", 1.0))

    raise NotImplementedError


def create_sampler_from_config(
    config,
    schedule: Schedule,
    timesteps: SamplingTimesteps,
) -> Sampler:
    """
    Create a sampler from configuration.
    """
    if config.type == "euler":
        return EulerSampler(
            schedule=schedule,
            timesteps=timesteps,
            prediction_type=config.prediction_type,
        )
    raise NotImplementedError


def create_sampling_timesteps_from_config(
    config,
    schedule: Schedule,
    device: torch.device,
) -> SamplingTimesteps:
    if config.type == "uniform_trailing":
        return UniformTrailingSamplingTimesteps(
            T=schedule.T,
            steps=config.steps,
            shift=config.get("shift", 1.0),
            device=device,
        )
    raise NotImplementedError
