# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import dtype_dict, compile_func # noqa: TID252
from ...packed_int import unpack_int # noqa: TID252
from ...packed_float import unpack_float # noqa: TID252
from ...dequantizer import dequantize_symmetric, dequantize_asymmetric # noqa: TID252


def quantized_embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    zero_point: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    embed_scale: torch.FloatTensor | float | None = None,
    result_dtype: torch.dtype | None = None,
    weight_shape: torch.Size | None = None,
    quantized_weight_shape: torch.Size | None = None,
    weights_dtype: str | None = None,
) -> torch.FloatTensor:
    return_shape = list(input.shape) + [weight_shape[-1] if weight_shape is not None else quantized_weight_shape[-1] if quantized_weight_shape is not None else weight.shape[-1]]
    input = input.flatten()

    if weights_dtype is not None and dtype_dict[weights_dtype]["is_packed"]:
        if dtype_dict[weights_dtype]["is_integer"]:
            weight = unpack_int(weight, weights_dtype, quantized_weight_shape)
        else:
            weight = unpack_float(weight, weights_dtype, quantized_weight_shape)

    if zero_point is not None:
        result = dequantize_asymmetric(weight[input], scale[input], zero_point[input], svd_up=svd_up[input] if svd_up is not None else svd_up, svd_down=svd_down, dtype=result_dtype)
    else:
        result = dequantize_symmetric(weight[input], scale[input], svd_up=svd_up[input] if svd_up is not None else svd_up, svd_down=svd_down, dtype=result_dtype)
    del input

    result = result.view(return_shape).contiguous()
    if embed_scale is not None:
        result = result.mul_(embed_scale)

    return result



def quantized_embedding_forward(self: torch.nn.Module, input: torch.Tensor) -> torch.FloatTensor:
    return quantized_embedding(
        input,
        self.weight,
        self.scale,
        zero_point=self.zero_point,
        svd_up=self.svd_up,
        svd_down=self.svd_down,
        embed_scale=getattr(self, "scalar_embed_scale", None),
        result_dtype=self.sdnq_dequantizer.result_dtype,
        weight_shape=self.sdnq_dequantizer.result_shape,
        quantized_weight_shape=self.sdnq_dequantizer.quantized_weight_shape,
        weights_dtype=self.sdnq_dequantizer.weights_dtype,
    )


quantized_embedding = compile_func(quantized_embedding)
