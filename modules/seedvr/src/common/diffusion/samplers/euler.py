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
Euler ODE solver.
"""

from typing import Callable
import torch
from einops import rearrange
from torch.nn import functional as F

#from ....models.dit_v2 import na

from ..types import PredictionType
from ..utils import expand_dims
from .base import Sampler, SamplerModelArgs


class EulerSampler(Sampler):
    """
    The Euler method is the simplest ODE solver.
    <https://en.wikipedia.org/wiki/Euler_method>
    """

    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor:
        timesteps = self.timesteps.timesteps
        #progress = self.get_progress_bar()
        i = 0

        # Optimisations VRAM
        original_dtype = x.dtype
        device = x.device

        for t, s in zip(timesteps[:-1], timesteps[1:]):
            # Appel du modèle avec monitoring
            pred = f(SamplerModelArgs(x, t, i))

            # Étape suivante
            x = self.step_to(pred, x, t, s)

            # Nettoyer les tenseurs temporaires
            del pred

            i += 1
            #progress.update()

        if self.return_endpoint:
            t = timesteps[-1]
            pred = f(SamplerModelArgs(x, t, i))
            x = self.get_endpoint(pred, x, t)
            del pred
            #progress.update()

        # Restaurer le dtype original si nécessaire
        if original_dtype != torch.float16:
            x = x.to(original_dtype)

        return x

    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step to the next timestep.
        """
        return self.step_to(pred, x_t, t, self.get_next_timestep(t))

    def step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        t = expand_dims(t, x_t.ndim)
        s = expand_dims(s, x_t.ndim)
        T = self.schedule.T
        # Step from x_t to x_s.
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        # Clamp x_s to x_0 and x_T if s is out of bound.
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        return pred_x_s
