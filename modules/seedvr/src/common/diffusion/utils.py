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
Utility functions.
"""

from typing import Callable
import torch


def expand_dims(tensor: torch.Tensor, ndim: int):
    """
    Expand tensor to target ndim. New dims are added to the right.
    For example, if the tensor shape was (8,), target ndim is 4, return (8, 1, 1, 1).
    """
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)


def assert_schedule_timesteps_compatible(schedule, timesteps):
    """
    Check if schedule and timesteps are compatible.
    """
    if schedule.T != timesteps.T:
        raise ValueError("Schedule and timesteps must have the same T.")
    if schedule.is_continuous() != timesteps.is_continuous():
        raise ValueError("Schedule and timesteps must have the same continuity.")


def classifier_free_guidance(
    pos: torch.Tensor,
    neg: torch.Tensor,
    scale: float,
    rescale: float = 0.0,
):
    """
    Apply classifier-free guidance.
    """
    # Classifier-free guidance (https://arxiv.org/abs/2207.12598)
    cfg = neg + scale * (pos - neg)

    # Classifier-free guidance rescale (https://arxiv.org/pdf/2305.08891.pdf)
    if rescale != 0.0:
        pos_std = pos.std(dim=list(range(1, pos.ndim)), keepdim=True)
        cfg_std = cfg.std(dim=list(range(1, cfg.ndim)), keepdim=True)
        factor = pos_std / cfg_std
        factor = rescale * factor + (1 - rescale)
        cfg *= factor

    return cfg


def classifier_free_guidance_dispatcher(
    pos: Callable,
    neg: Callable,
    scale: float,
    rescale: float = 0.0,
):
    """
    Optionally execute models depending on classifer-free guidance scale.
    """
    # If scale is 1, no need to execute neg model.
    if scale == 1.0:
        return pos()

    # Otherwise, execute both pos nad neg models and apply cfg.
    return classifier_free_guidance(
        pos=pos(),
        neg=neg(),
        scale=scale,
        rescale=rescale,
    )
