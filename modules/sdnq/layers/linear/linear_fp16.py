# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import compile_func, fp_mm_func # noqa: TID252
from ...dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias # noqa: TID252

from .forward import check_mats
from .linear_fp8_tensorwise import quantize_fp_mm_input_tensorwise


def fp16_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = (*input.shape[:-1], weight.shape[-1])
    if svd_up is not None:
        input.flatten(0,-2)
        if bias is not None:
            bias = torch.addmm(bias.to(dtype=svd_down.dtype), torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
        else:
            bias = torch.mm(torch.mm(input.to(dtype=svd_down.dtype), svd_down), svd_up)
    input, scale = quantize_fp_mm_input_tensorwise(input, scale, matmul_dtype="float16")
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(fp_mm_func(input, weight), scale, bias, dtype=return_dtype, result_shape=output_shape)
    else:
        return dequantize_symmetric(fp_mm_func(input, weight), scale, dtype=return_dtype, result_shape=output_shape)


def quantized_linear_forward_fp16_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if self.sdnq_dequantizer.re_quantize_for_matmul:
        weight, scale = self.sdnq_dequantizer.re_quantize_matmul(self.weight, self.scale, self.zero_point, None, None)
    else:
        weight, scale = self.weight, self.scale
    return fp16_matmul(input, weight, scale, bias=self.bias, svd_up=self.svd_up, svd_down=self.svd_down)


fp16_matmul = compile_func(fp16_matmul)
