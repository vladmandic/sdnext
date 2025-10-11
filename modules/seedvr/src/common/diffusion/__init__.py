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
Diffusion package.
"""

from .config import (
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from .samplers.base import Sampler
from .samplers.euler import EulerSampler
from .schedules.base import Schedule
from .schedules.lerp import LinearInterpolationSchedule
from .timesteps.base import SamplingTimesteps, Timesteps
from .timesteps.sampling.trailing import UniformTrailingSamplingTimesteps
from .types import PredictionType, SamplingDirection
from .utils import classifier_free_guidance, classifier_free_guidance_dispatcher, expand_dims

__all__ = [
    # Configs
    "create_sampler_from_config",
    "create_sampling_timesteps_from_config",
    "create_schedule_from_config",
    # Schedules
    "Schedule",
    "DiscreteVariancePreservingSchedule",
    "LinearInterpolationSchedule",
    # Samplers
    "Sampler",
    "EulerSampler",
    # Timesteps
    "Timesteps",
    "SamplingTimesteps",
    # Types
    "PredictionType",
    "SamplingDirection",
    "UniformTrailingSamplingTimesteps",
    # Utils
    "classifier_free_guidance",
    "classifier_free_guidance_dispatcher",
    "expand_dims",
]
