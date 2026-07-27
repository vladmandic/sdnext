# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import compile_func
from ...kernel_wrappers import int_scaled_mm_func, include_mm_kernel_in_compile
from ...quant_utils import quantize_uint_mm, rotate_hadamard, get_hadamard, merge_svd_quantized
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


def get_uint8_matmul_inputs(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    zero_point: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    svd_up_q: torch.CharTensor | None = None,
    svd_up_scale: torch.FloatTensor | None = None,
    svd_down_q: torch.CharTensor | None = None,
    svd_down_scale: torch.FloatTensor | None = None,
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
    if svd_up_q is not None:
        # matmul layout stores factors transposed: concat along rank dim 0 of up, dim 1 of down
        svd_up, svd_down = merge_svd_quantized(svd_up, svd_down, svd_up_q, svd_up_scale, svd_down_q, svd_down_scale, 0, 1, input.dtype)
    if svd_up is not None:
        input = input.flatten(0,-2)
        if bias is not None:
            bias = torch.addmm(bias.to(dtype=svd_down.dtype), torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
        else:
            bias = torch.mm(torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)

    input, input_scale, input_zero_point = quantize_uint_mm_input(input, dtype=scale.dtype)
    if zero_point is not None:
        zero_bias = torch.sum(input, dim=-1, keepdim=True, dtype=torch.int32).to(dtype=input_scale.dtype).mul_(input_scale).mul(zero_point)
        zero_bias.add_(torch.sum(weight, dim=0, keepdim=True, dtype=torch.int32).to(dtype=scale.dtype).mul_(scale).mul(input_zero_point))
        zero_bias.add_(torch.mul(input_zero_point, zero_point), alpha=input.shape[-1])
    else:
        zero_bias = torch.sum(weight, dim=0, keepdim=True, dtype=torch.int32).to(dtype=scale.dtype).mul_(scale).mul(input_zero_point)
    if bias is not None:
        zero_bias.add_(bias)

    input, weight = check_mats(input, weight, matmul_dtype="uint8")
    return input, weight, input_scale, scale, zero_bias, return_dtype, output_shape


def uint8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    zero_point: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    svd_up_q: torch.CharTensor | None = None,
    svd_up_scale: torch.FloatTensor | None = None,
    svd_down_q: torch.CharTensor | None = None,
    svd_down_scale: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    quantized_weight_shape: torch.Size | None = None,
    weights_dtype: str | None = None,
) -> torch.FloatTensor:
    input, weight, input_scale, scale, zero_bias, return_dtype, output_shape = get_uint8_matmul_inputs(
        input, weight,
        scale, zero_point,
        bias=bias,
        svd_up=svd_up,
        svd_down=svd_down,
        svd_up_q=svd_up_q,
        svd_up_scale=svd_up_scale,
        svd_down_q=svd_down_q,
        svd_down_scale=svd_down_scale,
        hadamard=hadamard,
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=weights_dtype,
    )
    return int_scaled_mm_func(input, weight, input_scale, scale, bias=zero_bias, out_dtype=return_dtype).view(output_shape)


def quantized_linear_forward_uint8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, zero_point=self.zero_point, svd_up=self.svd_up, svd_down=self.svd_down, svd_up_q=self.svd_up_q, svd_up_scale=self.svd_up_scale, svd_down_q=self.svd_down_q, svd_down_scale=self.svd_down_scale, skip_quantized_matmul=True), self.bias)
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
        svd_up_q=self.svd_up_q,
        svd_up_scale=self.svd_up_scale,
        svd_down_q=self.svd_down_q,
        svd_down_scale=self.svd_down_scale,
        hadamard=hadamard,
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=self.sdnq_dequantizer.weights_dtype,
    )


if not include_mm_kernel_in_compile:
    get_uint8_matmul_inputs = compile_func(get_uint8_matmul_inputs)
else:
    uint8_matmul = compile_func(uint8_matmul)
