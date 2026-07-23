# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...kernel_wrappers import use_contiguous_int8_mm, use_contiguous_fp16_mm, use_contiguous_fp8_mm


def check_mats(input: torch.Tensor, weight: torch.Tensor, matmul_dtype: str = "int8") -> tuple[torch.Tensor, torch.Tensor]:
    if input is not None:
        input = input.contiguous()
    if (
        (use_contiguous_int8_mm and matmul_dtype in {"int8", "uint8"})
        or (use_contiguous_fp16_mm and matmul_dtype in {"fp16", "float16"})
        or (use_contiguous_fp8_mm and matmul_dtype in {"fp8", "float8_e4m3fn"})
    ):
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t().contiguous().t()
    return input, weight


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down), self.bias)
