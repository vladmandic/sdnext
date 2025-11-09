# pylint: disable=redefined-builtin,no-member,protected-access

from typing import Tuple, Optional

import torch

from .common import dtype_dict, compile_func, use_contiguous_mm, use_tensorwise_fp8_matmul
from .packed_int import pack_int_symetric, unpack_int_symetric, pack_int_asymetric, unpack_int_asymetric


def dequantize_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    result = torch.addcmul(zero_point, weight.to(dtype=scale.dtype), scale)
    if result_shape is not None:
        result = result.view(result_shape)
    if svd_up is not None:
        if skip_quantized_matmul:
            svd_up = svd_up.t().contiguous()
            if use_contiguous_mm:
                svd_down = svd_down.t().contiguous()
            else:
                svd_down = svd_down.contiguous().t()
        if result.ndim > 2 and weight.ndim > 2: # convs
            result = result.add_(torch.mm(svd_up, svd_down).unflatten(-1, (*result.shape[1:],)))
        else:
            result = result.addmm_(svd_up, svd_down)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    result = weight.to(dtype=scale.dtype).mul_(scale)
    if skip_quantized_matmul:
        result.t_()
    if result_shape is not None:
        result = result.view(result_shape)
    if svd_up is not None:
        if skip_quantized_matmul:
            svd_up = svd_up.t().contiguous()
            if use_contiguous_mm:
                svd_down = svd_down.t().contiguous()
            else:
                svd_down = svd_down.contiguous().t()
        if result.ndim > 2 and weight.ndim > 2: # convs
            result = result.add_(torch.mm(svd_up, svd_down).unflatten(-1, (*result.shape[1:],)))
        else:
            result = result.addmm_(svd_up, svd_down)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def dequantize_symmetric_with_bias(weight: torch.CharTensor, scale: torch.FloatTensor, bias: torch.FloatTensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.FloatTensor:
    return torch.addcmul(bias, weight.to(dtype=scale.dtype), scale).to(dtype=dtype).view(result_shape)


def dequantize_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, dtype, result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


def dequantize_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, dtype, result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


def quantize_int8(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.div(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


def quantize_fp8(input: torch.FloatTensor, dim: int = -1, is_e5: bool = False) -> Tuple[torch.Tensor, torch.FloatTensor]:
    max_range = 57344 if is_e5 else 448
    fp8_dtype = torch.float8_e5m2 if is_e5 else torch.float8_e4m3fn
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(max_range)
    input = torch.div(input, scale).nan_to_num_().clamp_(-max_range, max_range).to(dtype=fp8_dtype)
    return input, scale


def re_quantize_int8(weight: torch.FloatTensor) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    if weight.ndim > 2: # convs
        weight = weight.flatten(1,-1)
    if use_contiguous_mm:
        weight, scale = quantize_int8(weight.t(), dim=-0)
        weight, scale = weight.contiguous(), scale.contiguous()
    else:
        weight, scale = quantize_int8(weight.contiguous(), dim=-1)
        weight, scale = weight.t_(), scale.t_()
    return weight, scale


def re_quantize_fp8(weight: torch.FloatTensor, is_e5: bool = False) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    if weight.ndim > 2: # convs
        weight = weight.flatten(1,-1)
    weight, scale = quantize_fp8(weight.contiguous(), dim=-1, is_e5=is_e5)
    weight, scale = weight.t_(), scale.t_()
    if not use_tensorwise_fp8_matmul:
        scale = scale.to(dtype=torch.float32)
    return weight, scale


def re_quantize_matmul_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_int8(dequantize_asymmetric(weight, scale, zero_point, scale.dtype, result_shape, svd_up=svd_up, svd_down=svd_down))


def re_quantize_matmul_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_int8(dequantize_symmetric(weight, scale, scale.dtype, result_shape, svd_up=svd_up, svd_down=svd_down))


def re_quantize_matmul_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, result_shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_matmul_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, result_shape, svd_up=svd_up, svd_down=svd_down)


def re_quantize_matmul_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, result_shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_matmul_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, result_shape, svd_up=svd_up, svd_down=svd_down)


def dequantize_sdnq_model(model):
    if hasattr(model, "sdnq_dequantizer"):
        model.weight = torch.nn.Parameter(model.sdnq_dequantizer(model.weight, model.scale, model.zero_point, model.svd_up, model.svd_down))
        del model.sdnq_dequantizer, model.scale, model.zero_point, model.svd_up, model.svd_down
        return model
    has_children = list(model.children())
    if not has_children:
        return model
    for module in model.children():
        if hasattr(module, "sdnq_dequantizer"):
            module.weight = torch.nn.Parameter(module.sdnq_dequantizer(module.weight, module.scale, module.zero_point, module.svd_up, module.svd_down))
            del module.sdnq_dequantizer, module.scale, module.zero_point, module.svd_up, module.svd_down
        else:
            module = dequantize_sdnq_model(module)
    return model


class AsymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        quantized_weight_shape: torch.Size,
        weights_dtype: str,
        group_size: int,
        svd_rank: int,
        use_quantized_matmul: bool,
        re_quantize_for_matmul: bool,
    ):
        super().__init__()
        self.is_packed = False
        self.is_asym = True
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.original_shape = original_shape
        self.quantized_weight_shape = quantized_weight_shape
        self.weights_dtype = weights_dtype
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.use_quantized_matmul = use_quantized_matmul
        self.re_quantize_for_matmul = re_quantize_for_matmul

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def re_quantize_matmul(self, weight, scale, zero_point, svd_up, svd_down, **kwargs): # pylint: disable=unused-argument
        return re_quantize_matmul_asymmetric_compiled(weight, scale, zero_point, self.result_shape, svd_up=svd_up, svd_down=svd_down)

    def forward(self, weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul=False): # pylint: disable=unused-argument
        return dequantize_asymmetric_compiled(weight, scale, zero_point, self.result_dtype, self.result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


class SymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        quantized_weight_shape: torch.Size,
        weights_dtype: str,
        group_size: int,
        svd_rank: int,
        use_quantized_matmul: bool,
        re_quantize_for_matmul: bool,
    ):
        super().__init__()
        self.is_packed = False
        self.is_asym = False
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.original_shape = original_shape
        self.quantized_weight_shape = quantized_weight_shape
        self.weights_dtype = weights_dtype
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.use_quantized_matmul = use_quantized_matmul
        self.re_quantize_for_matmul = re_quantize_for_matmul

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def re_quantize_matmul(self, weight, scale, zero_point, svd_up, svd_down, **kwargs): # pylint: disable=unused-argument
        return re_quantize_matmul_symmetric_compiled(weight, scale, self.result_shape, svd_up=svd_up, svd_down=svd_down)

    def forward(self, weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul=False): # pylint: disable=unused-argument
        skip_quantized_matmul = skip_quantized_matmul and not self.re_quantize_for_matmul
        return dequantize_symmetric_compiled(weight, scale, self.result_dtype, self.result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


class PackedINTAsymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        quantized_weight_shape: torch.Size,
        weights_dtype: str,
        group_size: int,
        svd_rank: int,
        use_quantized_matmul: bool,
        re_quantize_for_matmul: bool,
    ):
        super().__init__()
        self.is_packed = True
        self.is_asym = True
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.original_shape = original_shape
        self.quantized_weight_shape = quantized_weight_shape
        self.weights_dtype = weights_dtype
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.use_quantized_matmul = use_quantized_matmul
        self.re_quantize_for_matmul = re_quantize_for_matmul

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return pack_int_asymetric(weight, self.weights_dtype)

    def re_quantize_matmul(self, weight, scale, zero_point, svd_up, svd_down, **kwargs): # pylint: disable=unused-argument
        return re_quantize_matmul_packed_int_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.result_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down)

    def forward(self, weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul=False): # pylint: disable=unused-argument
        return dequantize_packed_int_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


class PackedINTSymmetricWeightsDequantizer(torch.nn.Module):
    def __init__(
        self,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        quantized_weight_shape: torch.Size,
        weights_dtype: str,
        group_size: int,
        svd_rank: int,
        use_quantized_matmul: bool,
        re_quantize_for_matmul: bool,
    ):
        super().__init__()
        self.is_packed = True
        self.is_asym = False
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.original_shape = original_shape
        self.quantized_weight_shape = quantized_weight_shape
        self.weights_dtype = weights_dtype
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.use_quantized_matmul = use_quantized_matmul
        self.re_quantize_for_matmul = re_quantize_for_matmul

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return pack_int_symetric(weight, self.weights_dtype)

    def re_quantize_matmul(self, weight, scale, zero_point, svd_up, svd_down, **kwargs): # pylint: disable=unused-argument
        return re_quantize_matmul_packed_int_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.result_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down)

    def forward(self, weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul=False): # pylint: disable=unused-argument
        skip_quantized_matmul = skip_quantized_matmul and not self.re_quantize_for_matmul
        return dequantize_packed_int_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


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
    "uint1": PackedINTAsymmetricWeightsDequantizer,
    "bool": PackedINTAsymmetricWeightsDequantizer,
    "float8_e4m3fn": SymmetricWeightsDequantizer,
    "float8_e4m3fnuz": SymmetricWeightsDequantizer,
    "float8_e5m2": SymmetricWeightsDequantizer,
    "float8_e5m2fnuz": SymmetricWeightsDequantizer,
}


dequantize_asymmetric_compiled = compile_func(dequantize_asymmetric)
dequantize_symmetric_compiled = compile_func(dequantize_symmetric)
dequantize_packed_int_asymmetric_compiled = compile_func(dequantize_packed_int_asymmetric)
dequantize_packed_int_symmetric_compiled = compile_func(dequantize_packed_int_symmetric)
re_quantize_matmul_asymmetric_compiled = compile_func(re_quantize_matmul_asymmetric)
re_quantize_matmul_symmetric_compiled = compile_func(re_quantize_matmul_symmetric)
re_quantize_matmul_packed_int_asymmetric_compiled = compile_func(re_quantize_matmul_packed_int_asymmetric)
re_quantize_matmul_packed_int_symmetric_compiled = compile_func(re_quantize_matmul_packed_int_symmetric)
