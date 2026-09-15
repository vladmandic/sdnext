import os
import sys

import pytest
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from modules.schedulers.scheduler_vdm import VDMScheduler


def _assert_timestep_spacing(spacing: str, num_steps: int) -> None:
    scheduler = VDMScheduler(timestep_spacing=spacing)
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    if spacing == "leading":
        expected = torch.arange(num_steps - 1, -1, -1, dtype=timesteps.dtype) / num_steps
    else:
        expected = torch.arange(num_steps, 0, -1, dtype=timesteps.dtype) / num_steps

    assert len(timesteps) == num_steps
    assert torch.all((0 <= timesteps) & (timesteps <= 1))
    if num_steps > 1:
        assert torch.all(timesteps[1:] < timesteps[:-1])
    torch.testing.assert_close(timesteps, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("spacing", ["leading", "trailing"])
@pytest.mark.parametrize("num_steps", [1, 2, 4, 49, 1000])
def test_timestep_spacing(spacing: str, num_steps: int) -> None:
    _assert_timestep_spacing(spacing, num_steps)


if __name__ == "__main__":
    for test_spacing in ("leading", "trailing"):
        for test_num_steps in (1, 2, 4, 49, 1000):
            _assert_timestep_spacing(test_spacing, test_num_steps)
