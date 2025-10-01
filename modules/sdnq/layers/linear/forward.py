# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch


def check_mats(input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    if weight.is_contiguous():
        if weight.device.type != "xpu":
            weight = weight.t().contiguous().t()
    elif weight.device.type == "xpu":
        weight = weight.contiguous()
    return input, weight


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight), self.bias)
