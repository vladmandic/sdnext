"""Logit-normal flow-matching schedule for Ideogram 4.

Sampling integrates the flow-matching ODE with Euler steps over a logit-normal
timestep schedule whose mean shifts with resolution (more steps at high noise for
larger images). Guidance is a per-step weight applied as an asymmetric blend of
the conditional and unconditional velocity. Ported from the reference
(github.com/ideogram-oss/ideogram4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass(frozen=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = torch.special.expit(y)
        t_ = 1 - t_
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return t_.clamp(t_min, t_max).to(torch.float32)


def get_schedule_for_resolution(
    image_resolution: tuple[int, int],
    known_resolution: tuple[int, int] = (512, 512),
    known_mean: float = 1.0,
    std: float = 1.0,
) -> LogitNormalSchedule:
    """Resolution-aware schedule: the mean shifts by half the log pixel-count ratio."""
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> torch.Tensor:
    """Linear step schedule mapped through the logit-normal schedule at sample time."""
    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)


class Ideogram4Scheduler(SchedulerMixin, ConfigMixin):
    """Thin diffusers scheduler holding the logit-normal defaults.

    The denoise loop lives in the pipeline (asymmetric dual-branch CFG over a
    flat per-step guidance), so this only carries the ``mu``/``std`` that
    parameterize the resolution-aware schedule and serves as the registered
    ``scheduler`` component.
    """

    @register_to_config
    def __init__(self, mu: float = 0.0, std: float = 1.75) -> None:
        pass
