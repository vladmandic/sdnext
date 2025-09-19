# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import compile_func # noqa: TID252
from ...dequantizer import quantize_fp8 # noqa: TID252


def quantize_fp8_matmul_input(input: torch.FloatTensor) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2).to(dtype=torch.float32)
    input, input_scale = quantize_fp8(input, dim=-1)
    return input, input_scale


def fp8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    input, input_scale = quantize_fp8_matmul_input(input)
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(dtype=torch.bfloat16)
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=torch.bfloat16).view(output_shape).to(return_dtype)


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul(input, self.weight, self.bias, self.sdnq_dequantizer.scale)


fp8_matmul = compile_func(fp8_matmul)
