# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import List

import torch

from ...common import use_torch_compile # noqa: TID252
from ..linear.linear_fp8 import quantize_fp8_matmul_input # noqa: TID252
from .forward import get_conv_args, process_conv_input


def conv_fp8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    result_shape: torch.Size,
    reversed_padding_repeated_twice: List[int],
    padding_mode: str, conv_type: int,
    groups: int, stride: List[int],
    padding: List[int], dilation: List[int],
) -> torch.FloatTensor:
    return_dtype = input.dtype
    input, mm_output_shape = process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation)
    input, input_scale = quantize_fp8_matmul_input(input)

    if groups == 1:
        if bias is not None and bias.dtype != torch.bfloat16:
            bias = bias.to(dtype=torch.bfloat16)
        result = torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=torch.bfloat16).view(mm_output_shape).to(return_dtype)
    else:
        scale = scale.view(groups, 1, scale.shape[1] // groups)
        input_scale = input_scale.view(groups, input_scale.shape[0] // groups, 1)
        weight = weight.view(weight.shape[0], groups, weight.shape[1] // groups)
        input = input.view(input.shape[0], groups, input.shape[1] // groups)
        result = []
        if bias is not None:
            bias = bias.view(groups, bias.shape[0] // groups)
            if bias.dtype != torch.bfloat16:
                bias = bias.to(dtype=torch.bfloat16)
            for i in range(groups):
                result.append(torch._scaled_mm(input[:, i], weight[:, i], scale_a=input_scale[i], scale_b=scale[i], bias=bias[i], out_dtype=torch.bfloat16))
        else:
            for i in range(groups):
                result.append(torch._scaled_mm(input[:, i], weight[:, i], scale_a=input_scale[i], scale_b=scale[i], bias=None, out_dtype=torch.bfloat16))
        result = torch.cat(result, dim=-1).view(mm_output_shape).to(return_dtype)

    if conv_type == 1:
        result = result.transpose_(1,2)
    elif conv_type == 2:
        result = result.permute(0,3,1,2)
    elif conv_type == 3:
        result = result.permute(0,4,1,2,3)
    return result


def quantized_conv_forward_fp8_matmul(self, input) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[2] < 32:
        return self._conv_forward(input, self.sdnq_dequantizer(self.weight, skip_quantized_matmul=True), self.bias)
    conv_type, stride, padding, dilation = get_conv_args(input.ndim, self.stride, self.padding, self.dilation)
    return conv_fp8_matmul(
        input, self.weight, self.bias,
        self.sdnq_dequantizer.scale,
        self.sdnq_dequantizer.result_shape,
        self._reversed_padding_repeated_twice,
        self.padding_mode, conv_type,
        self.groups, stride, padding, dilation,
    )


if use_torch_compile:
    conv_fp8_matmul = torch.compile(conv_fp8_matmul, fullgraph=True, dynamic=False)
