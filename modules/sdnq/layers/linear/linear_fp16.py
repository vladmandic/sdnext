# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import compile_func
from ...kernel_wrappers import fp_scaled_mm_func, include_mm_kernel_in_compile
from ...quant_utils import rotate_hadamard, get_hadamard, merge_svd_quantized
from ...packed_float import unpack_float

from .forward import check_mats
from .linear_fp8 import quantize_fp_mm_input


def get_fp16_matmul_inputs(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
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
        weight = unpack_float(weight, weights_dtype, quantized_weight_shape).to(dtype=torch.float16).t_()
        scale = scale.t()
    elif weight.dtype != torch.float16:
        weight = weight.to(dtype=torch.float16) # fp8 weights
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

    input, input_scale = quantize_fp_mm_input(input, dtype=scale.dtype, matmul_dtype="float16")
    input, weight = check_mats(input, weight, matmul_dtype="float16")
    return input, weight, input_scale, scale, bias, return_dtype, output_shape


def fp16_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
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
    input, weight, input_scale, scale, bias, return_dtype, output_shape = get_fp16_matmul_inputs(
        input, weight, scale,
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
    return fp_scaled_mm_func(input, weight, input_scale, scale, bias=bias, out_dtype=return_dtype).view(output_shape)


def quantized_linear_forward_fp16_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, zero_point=self.zero_point, svd_up=self.svd_up, svd_down=self.svd_down, svd_up_q=self.svd_up_q, svd_up_scale=self.svd_up_scale, svd_down_q=self.svd_down_q, svd_down_scale=self.svd_down_scale, codebook=self.codebook, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, zero_point=self.zero_point, codebook=self.codebook)
        quantized_weight_shape = None
    else:
        weight, scale = self.weight, self.scale
        quantized_weight_shape = self.sdnq_dequantizer.quantized_weight_shape if self.sdnq_dequantizer.is_packed else None
    if self.sdnq_dequantizer.use_hadamard:
        hadamard = get_hadamard(self.sdnq_dequantizer.hadamard_group_size, dtype=input.dtype, device=input.device)
    else:
        hadamard = None

    return fp16_matmul(
        input, weight, scale,
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
    get_fp16_matmul_inputs = compile_func(get_fp16_matmul_inputs)
else:
    fp16_matmul = compile_func(fp16_matmul)
