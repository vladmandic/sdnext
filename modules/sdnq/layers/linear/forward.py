# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch


def check_mats(input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input_stride = input.stride()
    if not (input_stride[0] > input_stride[1] and input_stride[1] == 1):
        input = input.contiguous()
    weight_stride = weight.stride()
    if not (weight_stride[0] == 1 and weight_stride[1] > 1):
        if weight.device.type != "xpu":
            weight = weight.t().contiguous().t()
    elif weight.device.type == "xpu":
        weight = weight.contiguous()
    return input, weight


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight), self.bias)
