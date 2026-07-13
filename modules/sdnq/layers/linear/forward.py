# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...kernel_wrappers import use_contiguous_int8_mm, use_contiguous_fp16_mm


def check_mats(input: torch.Tensor, weight: torch.Tensor, allow_contiguous_mm: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    if allow_contiguous_mm and ((use_contiguous_int8_mm and weight.dtype == torch.int8) or (use_contiguous_fp16_mm and weight.dtype == torch.float16)):
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t().contiguous().t()
    return input, weight


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down), self.bias)
