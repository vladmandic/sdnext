# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import compile_func, int_mm_func # noqa: TID252
from ...packed_int import unpack_int_symetric # noqa: TID252
from ...dequantizer import quantize_int8, dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252
from .forward import check_mats


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
    svd_up: torch.FloatTensor,
    svd_down: torch.FloatTensor,
    quantized_weight_shape: torch.Size,
    weights_dtype: str,
) -> torch.FloatTensor:
    if quantized_weight_shape is not None:
        weight = unpack_int_symetric(weight, quantized_weight_shape, weights_dtype, dtype=torch.int8)
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    if svd_up is not None:
        if bias is not None:
            bias = torch.addmm(bias, torch.mm(input.flatten(0,-2).to(dtype=svd_down.dtype), svd_down), svd_up)
        else:
            bias = torch.mm(torch.mm(input.flatten(0,-2).to(dtype=svd_down.dtype), svd_down), svd_up)
    input, scale = quantize_int8_matmul_input(input, scale)
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(int_mm_func(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(int_mm_func(input, weight), scale, return_dtype, output_shape)


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, self.zero_point, None, None)
        quantized_weight_shape = None
    else:
        weight = self.weight
        scale = self.scale
        quantized_weight_shape = getattr(self.sdnq_dequantizer, "quantized_weight_shape", None)
    return int8_matmul(input, weight, self.bias, scale, self.svd_up, self.svd_down, quantized_weight_shape, self.sdnq_dequantizer.weights_dtype)


int8_matmul = compile_func(int8_matmul)
