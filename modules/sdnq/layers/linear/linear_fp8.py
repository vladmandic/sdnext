# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import compile_func # noqa: TID252
from ...dequantizer import quantize_fp_mm # noqa: TID252

from .forward import check_mats


def quantize_fp_mm_input(input: torch.FloatTensor, matmul_dtype: str = "float8_e4m3fn") -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2).to(dtype=torch.float32)
    input, input_scale = quantize_fp_mm(input, dim=-1, matmul_dtype=matmul_dtype)
    return input, input_scale


def fp8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    if svd_up is not None:
        input = input.flatten(0,-2)
        svd_bias = torch.mm(torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
    input, input_scale = quantize_fp_mm_input(input)
    input, weight = check_mats(input, weight)
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(dtype=torch.bfloat16)
    result = torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=torch.bfloat16)
    if svd_up is not None:
        result.add_(svd_bias)
    result = result.view(output_shape).to(return_dtype)
    return result


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, self.zero_point, None, None)
    else:
        weight, scale = self.weight, self.scale
    return fp8_matmul(input, weight, scale, bias=self.bias, svd_up=self.svd_up, svd_down=self.svd_down)


fp8_matmul = compile_func(fp8_matmul)
