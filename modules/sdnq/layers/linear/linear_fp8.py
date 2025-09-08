# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import use_torch_compile # noqa: TID252


def quantize_fp8_matmul_input(input: torch.FloatTensor) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous().to(dtype=torch.float32)
    input_scale = torch.amax(input.abs(), dim=-1, keepdims=True).div_(448)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
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
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=return_dtype).view(output_shape)


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul(input, self.weight, self.bias, self.sdnq_dequantizer.scale)


if use_torch_compile:
    fp8_matmul = torch.compile(fp8_matmul, fullgraph=True, dynamic=False)
