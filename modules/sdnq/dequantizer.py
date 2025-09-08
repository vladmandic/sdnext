# pylint: disable=redefined-builtin,no-member,protected-access

import torch

from .common import dtype_dict, use_torch_compile
from .packed_int import pack_int_symetric, unpack_int_symetric, pack_int_asymetric, unpack_int_asymetric


def dequantize_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    result = torch.addcmul(zero_point, weight.to(dtype=scale.dtype), scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.view(result_shape)
    return result


def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    result = weight.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype)
    if skip_quantized_matmul:
        result.t_()
    if result_shape is not None:
        result = result.view(result_shape)
    return result


def dequantize_symmetric_with_bias(weight: torch.CharTensor, scale: torch.FloatTensor, bias: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    return torch.addcmul(bias, weight.to(dtype=scale.dtype), scale).to(dtype=dtype).view(result_shape)


def dequantize_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str) -> torch.FloatTensor:
    return dequantize_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, dtype, result_shape)


def dequantize_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, dtype, result_shape, skip_quantized_matmul=skip_quantized_matmul)


class AsymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        scale: torch.FloatTensor,
        zero_point: torch.FloatTensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        weights_dtype: str,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.original_shape = original_shape
        self.use_quantized_matmul = False
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def forward(self, weight, **kwargs): # pylint: disable=unused-argument
        return dequantize_asymmetric_compiled(weight, self.scale, self.zero_point, self.result_dtype, self.result_shape)


class SymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        scale: torch.FloatTensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        weights_dtype: str,
        use_quantized_matmul: bool = False,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.original_shape = original_shape
        self.use_quantized_matmul = use_quantized_matmul
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def forward(self, weight, skip_quantized_matmul=False, **kwargs): # pylint: disable=unused-argument
        return dequantize_symmetric_compiled(weight, self.scale, self.result_dtype, self.result_shape, skip_quantized_matmul=skip_quantized_matmul)


class PackedINTAsymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        scale: torch.FloatTensor,
        zero_point: torch.FloatTensor,
        quantized_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        weights_dtype: str,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = False
        self.original_shape = original_shape
        self.quantized_weight_shape = quantized_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return pack_int_asymetric(weight, self.weights_dtype)

    def forward(self, weight, **kwargs): # pylint: disable=unused-argument
        return dequantize_packed_int_asymmetric_compiled(weight, self.scale, self.zero_point, self.quantized_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype)


class PackedINTSymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        scale: torch.FloatTensor,
        quantized_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        weights_dtype: str,
        use_quantized_matmul: bool = False,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.original_shape = original_shape
        self.use_quantized_matmul = use_quantized_matmul
        self.quantized_weight_shape = quantized_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return pack_int_symetric(weight, self.weights_dtype)

    def forward(self, weight, skip_quantized_matmul=False, **kwargs): # pylint: disable=unused-argument
        return dequantize_packed_int_symmetric_compiled(weight, self.scale, self.quantized_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype, skip_quantized_matmul=skip_quantized_matmul)


dequantizer_dict = {
    "int8": SymmetricWeightsDequantizer,
    "int7": PackedINTSymmetricWeightsDequantizer,
    "int6": PackedINTSymmetricWeightsDequantizer,
    "int5": PackedINTSymmetricWeightsDequantizer,
    "int4": PackedINTSymmetricWeightsDequantizer,
    "int3": PackedINTSymmetricWeightsDequantizer,
    "int2": PackedINTSymmetricWeightsDequantizer,
    "uint8": AsymmetricWeightsDequantizer,
    "uint7": PackedINTAsymmetricWeightsDequantizer,
    "uint6": PackedINTAsymmetricWeightsDequantizer,
    "uint5": PackedINTAsymmetricWeightsDequantizer,
    "uint4": PackedINTAsymmetricWeightsDequantizer,
    "uint3": PackedINTAsymmetricWeightsDequantizer,
    "uint2": PackedINTAsymmetricWeightsDequantizer,
    "uint1": AsymmetricWeightsDequantizer,
    "bool": AsymmetricWeightsDequantizer,
    "float8_e4m3fn": SymmetricWeightsDequantizer,
    "float8_e4m3fnuz": SymmetricWeightsDequantizer,
    "float8_e5m2": SymmetricWeightsDequantizer,
    "float8_e5m2fnuz": SymmetricWeightsDequantizer,
}


if use_torch_compile:
    dequantize_asymmetric_compiled = torch.compile(dequantize_asymmetric, fullgraph=True, dynamic=False)
    dequantize_symmetric_compiled = torch.compile(dequantize_symmetric, fullgraph=True, dynamic=False)
    dequantize_packed_int_asymmetric_compiled = torch.compile(dequantize_packed_int_asymmetric, fullgraph=True, dynamic=False)
    dequantize_packed_int_symmetric_compiled = torch.compile(dequantize_packed_int_symmetric, fullgraph=True, dynamic=False)
else:
    dequantize_asymmetric_compiled = dequantize_asymmetric
    dequantize_symmetric_compiled = dequantize_symmetric
    dequantize_packed_int_asymmetric_compiled = dequantize_packed_int_asymmetric
    dequantize_packed_int_symmetric_compiled = dequantize_packed_int_symmetric
