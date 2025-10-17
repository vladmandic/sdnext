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
Linear interpolation schedule (lerp).
"""

from typing import Union
import torch

from .base import Schedule


class LinearInterpolationSchedule(Schedule):
    """
    Linear interpolation schedule (lerp) is proposed by flow matching and rectified flow.
    It leads to straighter probability flow theoretically. It is also used by Stable Diffusion 3.
    <https://arxiv.org/abs/2209.03003>
    <https://arxiv.org/abs/2210.02747>

        x_t = (1 - t) * x_0 + t * x_T

    Can be either continuous or discrete.
    """

    def __init__(self, T: Union[int, float] = 1.0):
        self._T = T

    @property
    def T(self) -> Union[int, float]:
        return self._T

    def A(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - (t / self.T)

    def B(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.T

    # ----------------------------------------------------

    def isnr(self, snr: torch.Tensor) -> torch.Tensor:
        t = self.T / (1 + snr**0.5)
        t = t if self.is_continuous() else t.round().int()
        return t
