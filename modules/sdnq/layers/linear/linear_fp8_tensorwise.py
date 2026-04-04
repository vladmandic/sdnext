# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import compile_func # noqa: TID252
from ...packed_float import unpack_float # noqa: TID252
from ...dequantizer import quantize_fp_mm, dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252

from .forward import check_mats


def quantize_fp_mm_input_tensorwise(input: torch.FloatTensor, dtype: torch.dtype | None = None, matmul_dtype: str = "float8_e4m3fn") -> tuple[torch.Tensor, torch.FloatTensor]:
    input = input.flatten(0,-2)
    if dtype is not None:
        input = input.to(dtype=dtype)
    input, input_scale = quantize_fp_mm(input, dim=-1, matmul_dtype=matmul_dtype)
    if input_scale.dtype == torch.float16: # fp16 will overflow
        input_scale = input_scale.to(dtype=torch.float32)
    return input, input_scale


def fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    quantized_weight_shape: torch.Size | None = None,
    weights_dtype: str | None = None,
) -> torch.FloatTensor:
    if quantized_weight_shape is not None:
        weight = unpack_float(weight, weights_dtype, quantized_weight_shape).to(dtype=torch.float8_e4m3fn).t_()
        scale = scale.t()
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    if svd_up is not None:
        input = input.flatten(0,-2)
        if bias is not None:
            bias = torch.addmm(bias.to(dtype=svd_down.dtype), torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
        else:
            bias = torch.mm(torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)
    input, input_scale = quantize_fp_mm_input_tensorwise(input, dtype=scale.dtype)
    input, weight = check_mats(input, weight, allow_contiguous_mm=False)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=input_scale.dtype).to(dtype=input_scale.dtype).mul_(input_scale), scale, bias, dtype=return_dtype, result_shape=output_shape)
    else:
        return dequantize_symmetric(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=input_scale.dtype).to(dtype=input_scale.dtype).mul_(input_scale), scale, dtype=return_dtype, result_shape=output_shape)


def quantized_linear_forward_fp8_matmul_tensorwise(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down, skip_quantized_matmul=True), self.bias)
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, self.zero_point, None, None)
        quantized_weight_shape = None
    else:
        weight, scale = self.weight, self.scale
        quantized_weight_shape = self.sdnq_dequantizer.quantized_weight_shape if self.sdnq_dequantizer.is_packed else None
    return fp8_matmul_tensorwise(
        input, weight, scale,
        bias=self.bias,
        svd_up=self.svd_up,
        svd_down=self.svd_down,
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=self.sdnq_dequantizer.weights_dtype,
    )


fp8_matmul_tensorwise = compile_func(fp8_matmul_tensorwise)
