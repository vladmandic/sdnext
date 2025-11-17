# pylint: disable=redefined-builtin,no-member,protected-access

from typing import List, Tuple, Optional

import torch

from modules import devices
from .common import dtype_dict, compile_func, use_contiguous_mm, use_tensorwise_fp8_matmul
from .packed_int import unpack_int_symetric, unpack_int_asymetric


@devices.inference_context()
def dequantize_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
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


@devices.inference_context()
def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
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


@devices.inference_context()
def dequantize_symmetric_with_bias(weight: torch.CharTensor, scale: torch.FloatTensor, bias: torch.FloatTensor, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None) -> torch.FloatTensor:
    result = torch.addcmul(bias, weight.to(dtype=scale.dtype), scale)
    if result_shape is not None:
        result = result.view(result_shape)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


@devices.inference_context()
def dequantize_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, weights_dtype: str, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, dtype=dtype, result_shape=result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


@devices.inference_context()
def dequantize_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, weights_dtype: str, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, dtype=dtype, result_shape=result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


@devices.inference_context()
def quantize_int8(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.div(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


@devices.inference_context()
def quantize_int8_sr(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.normal(0, 0.1, input.shape, device=input.device, dtype=input.dtype
    ).addcdiv_(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


@devices.inference_context()
def quantize_fp8(input: torch.FloatTensor, dim: int = -1, is_e5: bool = False) -> Tuple[torch.Tensor, torch.FloatTensor]:
    if is_e5:
        max_range = 57344
        fp8_dtype = torch.float8_e5m2
    else:
        max_range = 448
        fp8_dtype = torch.float8_e4m3fn
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(max_range)
    input = torch.div(input, scale).nan_to_num_().clamp_(-max_range, max_range).to(dtype=fp8_dtype)
    return input, scale


@devices.inference_context()
def quantize_fp8_sr(input: torch.FloatTensor, dim: int = -1, is_e5: bool = False) -> Tuple[torch.Tensor, torch.FloatTensor]:
    if is_e5:
        max_range = 57344
        fp8_dtype = torch.float8_e5m2
        mantissa_difference = 2097152
    else:
        max_range = 448
        fp8_dtype = torch.float8_e4m3fn
        mantissa_difference = 1048576
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(max_range)
    input = torch.div(input, scale).to(dtype=torch.float32).view(dtype=torch.int32)
    input = input.add_(torch.randint_like(input, low=0, high=mantissa_difference)).bitwise_and_(-mantissa_difference).view(dtype=torch.float32)
    input = input.nan_to_num_().clamp_(-max_range, max_range).to(dtype=fp8_dtype)
    return input, scale


@devices.inference_context()
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


@devices.inference_context()
def re_quantize_fp8(weight: torch.FloatTensor, is_e5: bool = False) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    if weight.ndim > 2: # convs
        weight = weight.flatten(1,-1)
    weight, scale = quantize_fp8(weight.contiguous(), dim=-1, is_e5=is_e5)
    weight, scale = weight.t_(), scale.t_()
    if not use_tensorwise_fp8_matmul:
        scale = scale.to(dtype=torch.float32)
    return weight, scale


@devices.inference_context()
def re_quantize_matmul_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_int8(dequantize_asymmetric(weight, scale, zero_point, dtype=scale.dtype, result_shape=result_shape, svd_up=svd_up, svd_down=svd_down))


@devices.inference_context()
def re_quantize_matmul_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_int8(dequantize_symmetric(weight, scale, dtype=scale.dtype, result_shape=result_shape, svd_up=svd_up, svd_down=svd_down))


@devices.inference_context()
def re_quantize_matmul_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, weights_dtype: str, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_matmul_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, result_shape=result_shape, svd_up=svd_up, svd_down=svd_down)


@devices.inference_context()
def re_quantize_matmul_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, weights_dtype: str, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    return re_quantize_matmul_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, result_shape=result_shape, svd_down=svd_down)


@devices.inference_context()
def dequantize_layer_weight(self: torch.nn.Module, inplace: bool = False):
    weight = self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down, skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul)
    if inplace:
        self.weight.data = weight
        self.forward = getattr(torch.nn, self.sdnq_dequantizer.layer_class_name).forward
        del self.sdnq_dequantizer, self.scale, self.zero_point, self.svd_up, self.svd_down
    return weight


@devices.inference_context()
def dequantize_sdnq_model(model: torch.nn.Module):
    if hasattr(model, "sdnq_dequantizer"):
        model.weight.data = dequantize_layer_weight(model, inplace=True)
    has_children = list(model.children())
    if not has_children:
        return model
    for module_name, module in model.named_children():
        if hasattr(module, "sdnq_dequantizer"):
            module.weight.data = dequantize_layer_weight(module, inplace=True)
            setattr(model, module_name, module)
        else:
            setattr(model, module_name, dequantize_sdnq_model(module))
    return model


class SDNQDequantizer():
    def __init__(
        self,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        original_stride: List[int],
        quantized_weight_shape: torch.Size,
        weights_dtype: str,
        group_size: int,
        svd_rank: int,
        svd_steps: int,
        use_quantized_matmul: bool,
        re_quantize_for_matmul: bool,
        use_stochastic_rounding: bool,
        layer_class_name: str,
    ):
        self.is_packed = dtype_dict[weights_dtype]["is_packed"]
        self.is_unsigned = dtype_dict[weights_dtype]["is_unsigned"]
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.original_shape = original_shape
        self.original_stride = original_stride
        self.quantized_weight_shape = quantized_weight_shape
        self.weights_dtype = weights_dtype
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.svd_steps = svd_steps
        self.use_quantized_matmul = use_quantized_matmul
        self.re_quantize_for_matmul = re_quantize_for_matmul
        self.use_stochastic_rounding = use_stochastic_rounding
        self.layer_class_name = layer_class_name

    @devices.inference_context()
    def re_quantize_matmul(self, weight, scale, zero_point, svd_up, svd_down): # pylint: disable=unused-argument
        if self.is_packed:
            if self.is_unsigned:
                return re_quantize_matmul_packed_int_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down)
            else:
                return re_quantize_matmul_packed_int_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.weights_dtype, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down)
        else:
            if self.is_unsigned:
                return re_quantize_matmul_asymmetric_compiled(weight, scale, zero_point, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down)
            else:
                return re_quantize_matmul_symmetric_compiled(weight, scale, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down)

    @devices.inference_context()
    def __call__(self, weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul: bool = False, dtype: torch.dtype = None): # pylint: disable=unused-argument
        skip_quantized_matmul = skip_quantized_matmul and not self.re_quantize_for_matmul
        if dtype is None:
            dtype = self.result_dtype
        if self.is_packed:
            if self.is_unsigned:
                return dequantize_packed_int_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, dtype=dtype, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)
            else:
                return dequantize_packed_int_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.weights_dtype, dtype=dtype, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)
        else:
            if self.is_unsigned:
                return dequantize_asymmetric_compiled(weight, scale, zero_point, dtype=dtype, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)
            else:
                return dequantize_symmetric_compiled(weight, scale, dtype=dtype, result_shape=self.result_shape, svd_up=svd_up, svd_down=svd_down, skip_quantized_matmul=skip_quantized_matmul)


dequantize_asymmetric_compiled = compile_func(dequantize_asymmetric)
dequantize_symmetric_compiled = compile_func(dequantize_symmetric)
dequantize_packed_int_asymmetric_compiled = compile_func(dequantize_packed_int_asymmetric)
dequantize_packed_int_symmetric_compiled = compile_func(dequantize_packed_int_symmetric)
re_quantize_matmul_asymmetric_compiled = compile_func(re_quantize_matmul_asymmetric)
re_quantize_matmul_symmetric_compiled = compile_func(re_quantize_matmul_symmetric)
re_quantize_matmul_packed_int_asymmetric_compiled = compile_func(re_quantize_matmul_packed_int_asymmetric)
re_quantize_matmul_packed_int_symmetric_compiled = compile_func(re_quantize_matmul_packed_int_symmetric)
