from abc import ABC, abstractmethod
from typing import Sequence, Union
import torch

from ..types import SamplingDirection


class Timesteps(ABC):
    """
    Timesteps base class.
    """

    def __init__(self, T: Union[int, float]):
        assert T > 0
        self._T = T

    @property
    def T(self) -> Union[int, float]:
        """
        Maximum timestep inclusive.
        int if discrete, float if continuous.
        """
        return self._T

    def is_continuous(self) -> bool:
        """
        Whether the schedule is continuous.
        """
        return isinstance(self.T, float)


class SamplingTimesteps(Timesteps):
    """
    Sampling timesteps.
    It defines the discretization of sampling steps.
    """

    def __init__(
        self,
        T: Union[int, float],
        timesteps: torch.Tensor,
        direction: SamplingDirection,
    ):
        assert timesteps.ndim == 1
        super().__init__(T)
        self.timesteps = timesteps
        self.direction = direction

    def __len__(self) -> int:
        """
        Number of sampling steps.
        """
        return len(self.timesteps)

    def __getitem__(self, idx: Union[int, torch.IntTensor]) -> torch.Tensor:
        """
        The timestep at the sampling step.
        Returns a scalar tensor if idx is int,
        or tensor of the same size if idx is a tensor.
        """
        return self.timesteps[idx]

    def index(self, t: torch.Tensor) -> torch.Tensor:
        """
        Find index by t.
        Return index of the same shape as t.
        Index is -1 if t not found in timesteps.
        """
        i, j = t.reshape(-1, 1).eq(self.timesteps).nonzero(as_tuple=True)
        idx = torch.full_like(t, fill_value=-1, dtype=torch.int)
        idx.view(-1)[i] = j.int()
        return idx
