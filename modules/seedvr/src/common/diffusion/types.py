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
Type definitions.
"""

from enum import Enum


class PredictionType(str, Enum):
    """
    x_0:
        Predict data sample.
    x_T:
        Predict noise sample.
        Proposed by DDPM (https://arxiv.org/abs/2006.11239)
        Proved problematic by zsnr paper (https://arxiv.org/abs/2305.08891)
    v_cos:
        Predict velocity dx/dt based on the cosine schedule (A_t * x_T - B_t * x_0).
        Proposed by progressive distillation (https://arxiv.org/abs/2202.00512)
    v_lerp:
        Predict velocity dx/dt based on the lerp schedule (x_T - x_0).
        Proposed by rectified flow (https://arxiv.org/abs/2209.03003)
    """

    x_0 = "x_0"
    x_T = "x_T"
    v_cos = "v_cos"
    v_lerp = "v_lerp"


class SamplingDirection(str, Enum):
    """
    backward: Sample from x_T to x_0 for data generation.
    forward:  Sample from x_0 to x_T for noise inversion.
    """

    backward = "backward"
    forward = "forward"

    @staticmethod
    def reverse(direction):
        if direction == SamplingDirection.backward:
            return SamplingDirection.forward
        if direction == SamplingDirection.forward:
            return SamplingDirection.backward
        raise NotImplementedError
