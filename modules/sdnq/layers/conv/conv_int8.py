# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import List

import torch

from ...common import use_torch_compile # noqa: TID252
from ...packed_int import unpack_int_symetric # noqa: TID252
from ...dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252
from ..linear.linear_int8 import quantize_int8_matmul_input # noqa: TID252
from .forward import get_conv_args, process_conv_input


def conv_int8_matmul(
    input: torch.FloatTensor,
    weight: torch.CharTensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    quantized_weight_shape: torch.Size,
    result_shape: torch.Size,
    weights_dtype: str,
    reversed_padding_repeated_twice: List[int],
    padding_mode: str, conv_type: int,
    groups: int, stride: List[int],
    padding: List[int], dilation: List[int],
) -> torch.FloatTensor:
    return_dtype = input.dtype
    input, mm_output_shape = process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation)
    input, scale = quantize_int8_matmul_input(input, scale)
    if quantized_weight_shape is not None:
        weight = unpack_int_symetric(weight, quantized_weight_shape, weights_dtype, dtype=torch.int8)

    if groups == 1:
        result = torch._int_mm(input, weight)
    else:
        weight = weight.view(weight.shape[0], groups, weight.shape[1] // groups)
        input = input.view(input.shape[0], groups, input.shape[1] // groups)
        result = []
        for i in range(groups):
            result.append(torch._int_mm(input[:, i], weight[:, i]))
        result = torch.cat(result, dim=-1)
    if bias is not None:
        result = dequantize_symmetric_with_bias(result, scale, bias, return_dtype, mm_output_shape)
    else:
        result = dequantize_symmetric(result, scale, return_dtype, mm_output_shape)

    if conv_type == 1:
        result = result.transpose_(1,2)
    elif conv_type == 2:
        result = result.permute(0,3,1,2)
    elif conv_type == 3:
        result = result.permute(0,4,1,2,3)
    return result


def quantized_conv_forward_int8_matmul(self, input) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[2] < 32:
        return self._conv_forward(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    conv_type, stride, padding, dilation = get_conv_args(input.ndim, self.stride, self.padding, self.dilation)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight)
        quantized_weight_shape = None
    else:
        weight = self.weight
        scale = self.sdnq_dequantizer.scale
        quantized_weight_shape = getattr(self.sdnq_dequantizer, "quantized_weight_shape", None)
    return conv_int8_matmul(
        input, weight, self.bias,
        scale, quantized_weight_shape,
        self.sdnq_dequantizer.result_shape,
        self.sdnq_dequantizer.weights_dtype,
        self._reversed_padding_repeated_twice,
        self.padding_mode, conv_type,
        self.groups, stride, padding, dilation,
    )


if use_torch_compile:
    conv_int8_matmul = torch.compile(conv_int8_matmul, fullgraph=True, dynamic=False)
