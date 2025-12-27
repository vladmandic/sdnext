# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import List

import torch

from ...common import compile_func # noqa: TID252
from ...packed_float import unpack_float # noqa: TID252
from ...dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252

from .forward import get_conv_args, process_conv_input
from ..linear.linear_fp8_tensorwise import quantize_fp_mm_input_tensorwise # noqa: TID252
from ..linear.forward import check_mats # noqa: TID252


def conv_fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    result_shape: torch.Size,
    reversed_padding_repeated_twice: List[int],
    padding_mode: str, conv_type: int,
    groups: int, stride: List[int],
    padding: List[int], dilation: List[int],
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    quantized_weight_shape: torch.Size = None,
    weights_dtype: str = None,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    input, mm_output_shape = process_conv_input(conv_type, input, reversed_padding_repeated_twice, padding_mode, result_shape, stride, padding, dilation)
    if svd_up is not None:
        input = input.flatten(0,-2)
        if bias is not None:
            bias = torch.addmm(bias.to(dtype=svd_down.dtype), torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
        else:
            bias = torch.mm(torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)

    input, scale = quantize_fp_mm_input_tensorwise(input, scale)
    if quantized_weight_shape is not None:
        weight = unpack_float(weight, quantized_weight_shape, weights_dtype).to(dtype=torch.float8_e4m3fn)
    input, weight = check_mats(input, weight)
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)

    if groups == 1:
        result = torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype)
    else:
        weight = weight.view(weight.shape[0], groups, weight.shape[1] // groups)
        input = input.view(input.shape[0], groups, input.shape[1] // groups)
        result = []
        for i in range(groups):
            result.append(torch._scaled_mm(input[:, i], weight[:, i], scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype))
        result = torch.cat(result, dim=-1)
    if bias is not None:
        dequantize_symmetric_with_bias(result, scale, bias, dtype=return_dtype, result_shape=mm_output_shape)
    else:
        dequantize_symmetric(result, scale, dtype=return_dtype, result_shape=mm_output_shape)

    if conv_type == 1:
        result = result.transpose_(1,2)
    elif conv_type == 2:
        result = result.permute(0,3,1,2)
    elif conv_type == 3:
        result = result.permute(0,4,1,2,3)
    return result


def quantized_conv_forward_fp8_matmul_tensorwise(self, input) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[2] < 32:
        return self._conv_forward(input, self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, self.zero_point, None, None)
        quantized_weight_shape = None
    else:
        weight, scale = self.weight, self.scale
        quantized_weight_shape = self.sdnq_dequantizer.quantized_weight_shape if self.sdnq_dequantizer.is_packed else None
    conv_type, stride, padding, dilation = get_conv_args(input.ndim, self.stride, self.padding, self.dilation)
    return conv_fp8_matmul_tensorwise(
        input, weight, scale,
        self.sdnq_dequantizer.result_shape,
        self._reversed_padding_repeated_twice,
        self.padding_mode, conv_type,
        self.groups, stride, padding, dilation,
        bias=self.bias,
        svd_up=self.svd_up,
        svd_down=self.svd_down,
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=self.sdnq_dequantizer.weights_dtype,
    )


conv_fp8_matmul_tensorwise = compile_func(conv_fp8_matmul_tensorwise)
