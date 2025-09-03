# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple
import torch

from ...common import use_torch_compile # noqa: TID252
from ...packed_int import unpack_int_symetric # noqa: TID252
from ...dequantizer import quantize_int8, dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    input = input.flatten(0,-2).to(dtype=scale.dtype)
    input, input_scale = quantize_int8(input, dim=-1)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def int8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    quantized_weight_shape: torch.Size,
    weights_dtype: str,
) -> torch.FloatTensor:
    if quantized_weight_shape is not None:
        weight = unpack_int_symetric(weight, quantized_weight_shape, weights_dtype, dtype=torch.int8)
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    input, scale = quantize_int8_matmul_input(input, scale)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._int_mm(input, weight), scale, return_dtype, output_shape)


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight)
        quantized_weight_shape = None
    else:
        weight = self.weight
        scale = self.sdnq_dequantizer.scale
        quantized_weight_shape = getattr(self.sdnq_dequantizer, "quantized_weight_shape", None)
    return int8_matmul(input, weight, self.bias, scale, quantized_weight_shape, self.sdnq_dequantizer.weights_dtype)


if use_torch_compile:
    int8_matmul = torch.compile(int8_matmul, fullgraph=True, dynamic=False)
