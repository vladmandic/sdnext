# pylint: disable=redefined-builtin,no-member,protected-access

import torch
from modules import shared

from .common import dtype_dict
from .packed_int import pack_int_symetric, unpack_int_symetric, packed_int_function_dict


def decompress_asymmetric(input: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    result = torch.addcmul(zero_point, input.to(dtype=scale.dtype), scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_symmetric(input: torch.CharTensor, scale: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    if skip_quantized_matmul:
        result = input.transpose(0,1).to(dtype=scale.dtype).mul_(scale.transpose(0,1)).to(dtype=dtype)
    else:
        result = input.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_packed_int_asymmetric(input: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str) -> torch.FloatTensor:
    return decompress_asymmetric(packed_int_function_dict[weights_dtype]["unpack"](input, shape), scale, zero_point, dtype, result_shape)


def decompress_packed_int_symmetric(input: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    if skip_quantized_matmul:
        return decompress_symmetric(unpack_int_symetric(input, shape, weights_dtype, dtype=scale.dtype), scale.transpose(0,1), dtype, result_shape)
    else:
        return decompress_symmetric(unpack_int_symetric(input, shape, weights_dtype, dtype=scale.dtype), scale, dtype, result_shape)


class AsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = False
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def forward(self, weight, **kwargs): # pylint: disable=unused-argument
        return decompress_asymmetric(weight, self.scale, self.zero_point, self.result_dtype, self.result_shape)


class SymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        use_quantized_matmul: bool = False,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = use_quantized_matmul
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def forward(self, weight, skip_quantized_matmul=False, **kwargs): # pylint: disable=unused-argument
        return decompress_symmetric(weight, self.scale, self.result_dtype, self.result_shape, skip_quantized_matmul=skip_quantized_matmul)


class PackedINTAsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        compressed_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = False
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return packed_int_function_dict[self.weights_dtype]["pack"](weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"]))

    def forward(self, weight, **kwargs): # pylint: disable=unused-argument
        return decompress_packed_int_asymmetric(weight, self.scale, self.zero_point, self.compressed_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype)


class PackedINTSymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        compressed_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        use_quantized_matmul: bool = False,
        **kwargs, # pylint: disable=unused-argument
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = use_quantized_matmul
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return pack_int_symetric(weight, self.weights_dtype)

    def forward(self, weight, skip_quantized_matmul=False, **kwargs): # pylint: disable=unused-argument
        return decompress_packed_int_symmetric(weight, self.scale, self.compressed_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype, skip_quantized_matmul=skip_quantized_matmul)


decompressor_dict = {
    "int8": SymmetricWeightsDecompressor,
    "int7": PackedINTSymmetricWeightsDecompressor,
    "int6": PackedINTSymmetricWeightsDecompressor,
    "int5": PackedINTSymmetricWeightsDecompressor,
    "int4": PackedINTSymmetricWeightsDecompressor,
    "int3": PackedINTSymmetricWeightsDecompressor,
    "int2": PackedINTSymmetricWeightsDecompressor,
    "uint8": AsymmetricWeightsDecompressor,
    "uint7": PackedINTAsymmetricWeightsDecompressor,
    "uint6": PackedINTAsymmetricWeightsDecompressor,
    "uint5": PackedINTAsymmetricWeightsDecompressor,
    "uint4": PackedINTAsymmetricWeightsDecompressor,
    "uint3": PackedINTAsymmetricWeightsDecompressor,
    "uint2": PackedINTAsymmetricWeightsDecompressor,
    "uint1": AsymmetricWeightsDecompressor,
    "bool": AsymmetricWeightsDecompressor,
    "float8_e4m3fn": SymmetricWeightsDecompressor,
    "float8_e4m3fnuz": SymmetricWeightsDecompressor,
    "float8_e5m2": SymmetricWeightsDecompressor,
    "float8_e5m2fnuz": SymmetricWeightsDecompressor,
}


if shared.opts.sdnq_decompress_compile:
    try:
        torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
        decompress_asymmetric = torch.compile(decompress_asymmetric, fullgraph=True)
        decompress_symmetric = torch.compile(decompress_symmetric, fullgraph=True)
        decompress_packed_int_asymmetric = torch.compile(decompress_packed_int_asymmetric, fullgraph=True)
        decompress_packed_int_symmetric = torch.compile(decompress_packed_int_symmetric, fullgraph=True)
    except Exception as e:
        shared.log.warning(f"Quantization: type=sdnq Decompress using torch.compile is not available: {e}")
