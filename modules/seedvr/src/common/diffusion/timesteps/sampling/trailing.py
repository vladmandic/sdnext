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

import torch

from ...types import SamplingDirection
from ..base import SamplingTimesteps


class UniformTrailingSamplingTimesteps(SamplingTimesteps):
    """
    Uniform trailing sampling timesteps.
    Defined in (https://arxiv.org/abs/2305.08891)

    Shift is proposed in SD3 for RF schedule.
    Defined in (https://arxiv.org/pdf/2403.03206) eq.23
    """

    def __init__(
        self,
        T: int,
        steps: int,
        shift: float = 1.0,
        device: torch.device = "cpu",
    ):
        # Create trailing timesteps.
        timesteps = torch.arange(1.0, 0.0, -1.0 / steps, device=device)

        # Shift timesteps.
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)

        # Scale to T range.
        if isinstance(T, float):
            timesteps = timesteps * T
        else:
            timesteps = timesteps.mul(T + 1).sub(1).round().int()

        super().__init__(T=T, timesteps=timesteps, direction=SamplingDirection.backward)
