# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import compile_func, int_mm_func
from ...dequantizer import dequantize_asymmetric
from ...quant_utils import quantize_uint_mm, rotate_hadamard, get_hadamard
from ...packed_int import unpack_int

from .forward import check_mats


def quantize_uint_mm_input(input: torch.FloatTensor, dtype: torch.dtype | None = None, matmul_dtype: str = "uint8") -> tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    input = input.flatten(0,-2)
    if dtype is not None:
        input = input.to(dtype=dtype)
    input, input_scale, input_zero_point = quantize_uint_mm(input, dim=-1, matmul_dtype=matmul_dtype)
    if input_scale.dtype == torch.float16: # fp16 will overflow
        input_scale = input_scale.to(dtype=torch.float32)
        input_zero_point = input_zero_point.to(dtype=torch.float32)
    return input, input_scale, input_zero_point


def uint8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    zero_point: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    quantized_weight_shape: torch.Size | None = None,
    weights_dtype: str | None = None,
) -> torch.FloatTensor:
    if quantized_weight_shape is not None:
        weight = unpack_int(weight, weights_dtype, quantized_weight_shape, dtype=torch.int8).t_()
        scale = scale.t()
        if zero_point is not None:
            zero_point = zero_point.t()
        if weight.dtype == torch.uint8:
            weight = weight.view(dtype=torch.int8)
    elif weight.dtype == torch.uint8:
        weight = weight.bitwise_xor(128).view(torch.int8)
        if zero_point is not None:
            zero_point = torch.add(zero_point, scale, alpha=128)
        else:
            zero_point = torch.mul(scale, 128)

    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])

    if hadamard is not None:
        input = rotate_hadamard(input, hadamard=hadamard)
    if svd_up is not None:
        input = input.flatten(0,-2)
        if bias is not None:
            bias = torch.addmm(bias.to(dtype=svd_down.dtype), torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
        else:
            bias = torch.mm(torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)

    input, input_scale, input_zero_point = quantize_uint_mm_input(input, dtype=scale.dtype)
    if zero_point is not None:
        zero_bias = torch.sum(input, dim=-1, keepdim=True, dtype=torch.int32).to(input_scale.dtype).mul_(input_scale).mul(zero_point)
        zero_bias.add_(torch.sum(weight, dim=0, keepdim=True, dtype=torch.int32).to(scale.dtype).mul_(scale).mul(input_zero_point))
        zero_bias.add_(torch.mul(input_zero_point.mul_(input.shape[-1]), zero_point))
    else:
        zero_bias = torch.sum(weight, dim=0, keepdim=True, dtype=torch.int32).to(scale.dtype).mul_(scale).mul(input_zero_point)
    if bias is not None:
        zero_bias.add_(bias)

    input, weight = check_mats(input, weight)
    return dequantize_asymmetric(int_mm_func(input, weight).to(dtype=input_scale.dtype).mul_(input_scale), scale, zero_bias, dtype=return_dtype, result_shape=output_shape)


def quantized_linear_forward_uint8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, zero_point=self.zero_point, svd_up=self.svd_up, svd_down=self.svd_down, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale, zero_point = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, zero_point=self.zero_point)
        quantized_weight_shape = None
    else:
        weight, scale, zero_point = self.weight, self.scale, self.zero_point
        quantized_weight_shape = self.sdnq_dequantizer.quantized_weight_shape if self.sdnq_dequantizer.is_packed else None
    if self.sdnq_dequantizer.use_hadamard:
        hadamard = get_hadamard(self.sdnq_dequantizer.hadamard_group_size, dtype=input.dtype, device=input.device)
    else:
        hadamard = None

    return uint8_matmul(
        input, weight,
        scale, zero_point,
        bias=self.bias,
        svd_up=self.svd_up,
        svd_down=self.svd_down,
        hadamard=hadamard,
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=self.sdnq_dequantizer.weights_dtype,
    )


uint8_matmul = compile_func(uint8_matmul)
