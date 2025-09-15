# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import use_torch_compile # noqa: TID252
from ...dequantizer import quantize_fp8, dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252


def quantize_fp8_matmul_input_tensorwise(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2).to(dtype=scale.dtype)
    input, input_scale = quantize_fp8(input, dim=-1)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)
    input, scale = quantize_fp8_matmul_input_tensorwise(input, scale)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, return_dtype, output_shape)


def quantized_linear_forward_fp8_matmul_tensorwise(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul_tensorwise(input, self.weight, self.bias, self.sdnq_dequantizer.scale)


if use_torch_compile:
    fp8_matmul_tensorwise = torch.compile(fp8_matmul_tensorwise, fullgraph=True, dynamic=False)
