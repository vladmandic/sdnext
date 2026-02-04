# pylint: disable=redefined-builtin,no-member,protected-access

from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch

from modules import devices
from .common import dtype_dict, compile_func, use_contiguous_mm, use_tensorwise_fp8_matmul
from .packed_int import unpack_int_symetric, unpack_int_asymetric
from .packed_float import unpack_float
from .layers import SDNQLayer


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
            result = result.to(dtype=svd_up.dtype).addmm_(svd_up, svd_down)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


@devices.inference_context()
def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False, re_quantize_for_matmul: bool = False) -> torch.FloatTensor:
    result = weight.to(dtype=scale.dtype).mul_(scale)
    if skip_quantized_matmul and not re_quantize_for_matmul:
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
            result = result.to(dtype=svd_up.dtype).addmm_(svd_up, svd_down)
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
def dequantize_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=result_shape, skip_quantized_matmul=skip_quantized_matmul)


@devices.inference_context()
def dequantize_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False, re_quantize_for_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)


@devices.inference_context()
def dequantize_packed_float_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_asymmetric(unpack_float(weight, shape, weights_dtype), scale, zero_point, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=result_shape, skip_quantized_matmul=skip_quantized_matmul)


@devices.inference_context()
def dequantize_packed_float_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, weights_dtype: str, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None, skip_quantized_matmul: bool = False, re_quantize_for_matmul: bool = False) -> torch.FloatTensor:
    return dequantize_symmetric(unpack_float(weight, shape, weights_dtype), scale, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)


@devices.inference_context()
def quantize_int_mm(input: torch.FloatTensor, dim: int = -1, matmul_dtype: str = "int8") -> Tuple[torch.Tensor, torch.FloatTensor]:
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(dtype_dict[matmul_dtype]["max"])
    input = torch.div(input, scale).round_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return input, scale


@devices.inference_context()
def quantize_int_mm_sr(input: torch.FloatTensor, dim: int = -1, matmul_dtype: str = "int8") -> Tuple[torch.Tensor, torch.FloatTensor]:
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(dtype_dict[matmul_dtype]["max"])
    input = torch.div(input, scale).add_(torch.randn_like(input), alpha=0.1).round_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return input, scale


@devices.inference_context()
def quantize_fp_mm(input: torch.FloatTensor, dim: int = -1, matmul_dtype: str = "float8_e4m3fn") -> Tuple[torch.Tensor, torch.FloatTensor]:
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(dtype_dict[matmul_dtype]["max"])
    input = torch.div(input, scale).nan_to_num_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return input, scale


@devices.inference_context()
def quantize_fp_mm_sr(input: torch.FloatTensor, dim: int = -1, matmul_dtype: str = "float8_e4m3fn") -> Tuple[torch.Tensor, torch.FloatTensor]:
    mantissa_difference = 1 << (23 - dtype_dict[matmul_dtype]["mantissa"])
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(dtype_dict[matmul_dtype]["max"])
    input = torch.div(input, scale).to(dtype=torch.float32).view(dtype=torch.int32)
    input = input.add_(torch.randint_like(input, low=0, high=mantissa_difference, dtype=torch.int32)).bitwise_and_(-mantissa_difference).view(dtype=torch.float32)
    input = input.nan_to_num_().clamp_(dtype_dict[matmul_dtype]["min"], dtype_dict[matmul_dtype]["max"]).to(dtype=dtype_dict[matmul_dtype]["torch_dtype"])
    return input, scale


@devices.inference_context()
def re_quantize_int_mm(weight: torch.FloatTensor) -> Tuple[torch.Tensor, torch.FloatTensor]:
    if weight.ndim > 2: # convs
        weight = weight.flatten(1,-1)
    if use_contiguous_mm:
        weight, scale = quantize_int_mm(weight.t().contiguous(), dim=0)
    else:
        weight, scale = quantize_int_mm(weight.contiguous(), dim=-1)
        weight, scale = weight.t_(), scale.t_()
    return weight, scale


@devices.inference_context()
def re_quantize_fp_mm(weight: torch.FloatTensor, matmul_dtype: str = "float8_e4m3fn") -> Tuple[torch.Tensor, torch.FloatTensor]:
    if weight.ndim > 2: # convs
        weight = weight.flatten(1,-1)
    weight, scale = quantize_fp_mm(weight.contiguous(), dim=-1, matmul_dtype=matmul_dtype)
    weight, scale = weight.t_(), scale.t_()
    if not use_tensorwise_fp8_matmul and dtype_dict[matmul_dtype]["num_bits"] == 8:
        scale = scale.to(dtype=torch.float32)
    return weight, scale


@devices.inference_context()
def re_quantize_matmul_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, matmul_dtype: str, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor, torch.FloatTensor]:
    weight = dequantize_asymmetric(weight, scale, zero_point, svd_up=svd_up, svd_down=svd_down, dtype=scale.dtype, result_shape=result_shape)
    if dtype_dict[matmul_dtype]["is_integer"]:
        return re_quantize_int_mm(weight)
    else:
        return re_quantize_fp_mm(weight, matmul_dtype=matmul_dtype)


@devices.inference_context()
def re_quantize_matmul_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, matmul_dtype: str, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor, torch.FloatTensor]:
    weight = dequantize_symmetric(weight, scale, svd_up=svd_up, svd_down=svd_down, dtype=scale.dtype, result_shape=result_shape)
    if dtype_dict[matmul_dtype]["is_integer"]:
        return re_quantize_int_mm(weight)
    else:
        return re_quantize_fp_mm(weight, matmul_dtype=matmul_dtype)


@devices.inference_context()
def re_quantize_matmul_packed_int_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, weights_dtype: str, matmul_dtype: str, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor, torch.FloatTensor]:
    return re_quantize_matmul_asymmetric(unpack_int_asymetric(weight, shape, weights_dtype), scale, zero_point, matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=result_shape)


@devices.inference_context()
def re_quantize_matmul_packed_int_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, weights_dtype: str, matmul_dtype: str, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor, torch.FloatTensor]:
    return re_quantize_matmul_symmetric(unpack_int_symetric(weight, shape, weights_dtype, dtype=scale.dtype), scale, matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=result_shape)


@devices.inference_context()
def re_quantize_matmul_packed_float_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, shape: torch.Size, weights_dtype: str, matmul_dtype: str, result_shape: torch.Size, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor, torch.FloatTensor]:
    return re_quantize_matmul_asymmetric(unpack_float(weight, shape, weights_dtype), scale, zero_point, matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=result_shape)


@devices.inference_context()
def re_quantize_matmul_packed_float_symmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, shape: torch.Size, weights_dtype: str, matmul_dtype: str, result_shape: Optional[torch.Size] = None, svd_up: Optional[torch.FloatTensor] = None, svd_down: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor, torch.FloatTensor]:
    return re_quantize_matmul_symmetric(unpack_float(weight, shape, weights_dtype), scale, matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=result_shape)


@devices.inference_context()
def dequantize_sdnq_module(model: torch.nn.Module):
    if isinstance(model, SDNQLayer):
        model = model.dequantize()
    has_children = list(model.children())
    if not has_children:
        return model
    for module_name, module in model.named_children():
        if isinstance(module, SDNQLayer):
            setattr(model, module_name, module.dequantize())
        else:
            setattr(model, module_name, dequantize_sdnq_model(module))
    return model


@devices.inference_context()
def dequantize_sdnq_model(model: torch.nn.Module):
    model = dequantize_sdnq_module(model)
    if hasattr(model, "quantization_method"):
        del model.quantization_method
    if hasattr(model, "quantization_config"):
        del model.quantization_config
    if hasattr(model, "config"):
        try:
            if hasattr(model.config, "quantization_config"):
                del model.config.quantization_config
        except Exception:
            pass
        try:
            if hasattr(model.config, "pop"):
                model.config.pop("quantization_config", None)
        except Exception:
            pass
    return model


# SDNQDequantizer has to be a dataclass for torch.compile
@dataclass
class SDNQDequantizer:
    result_dtype: torch.dtype
    result_shape: torch.Size
    original_shape: torch.Size
    original_stride: List[int]
    quantized_weight_shape: torch.Size
    weights_dtype: str
    quantized_matmul_dtype: str
    group_size: int
    svd_rank: int
    svd_steps: int
    use_quantized_matmul: bool
    re_quantize_for_matmul: bool
    use_stochastic_rounding: bool
    layer_class_name: str
    is_packed: bool
    is_unsigned: bool
    is_integer: bool
    is_integer_matmul: bool

    def __init__(
        self,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        original_shape: torch.Size,
        original_stride: List[int],
        quantized_weight_shape: torch.Size,
        weights_dtype: str,
        quantized_matmul_dtype: str,
        group_size: int,
        svd_rank: int,
        svd_steps: int,
        use_quantized_matmul: bool,
        re_quantize_for_matmul: bool,
        use_stochastic_rounding: bool,
        layer_class_name: str,
    ):
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.original_shape = original_shape
        self.original_stride = original_stride
        self.quantized_weight_shape = quantized_weight_shape
        self.weights_dtype = weights_dtype
        self.quantized_matmul_dtype = quantized_matmul_dtype
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.svd_steps = svd_steps
        self.use_quantized_matmul = use_quantized_matmul
        self.re_quantize_for_matmul = re_quantize_for_matmul
        self.use_stochastic_rounding = use_stochastic_rounding
        self.layer_class_name = layer_class_name
        self.is_packed = dtype_dict[weights_dtype]["is_packed"]
        self.is_unsigned = dtype_dict[weights_dtype]["is_unsigned"]
        self.is_integer = dtype_dict[weights_dtype]["is_integer"]
        self.is_integer_matmul = dtype_dict[quantized_matmul_dtype]["is_integer"]

    @devices.inference_context()
    def re_quantize_matmul(self, weight, scale, zero_point, svd_up, svd_down): # pylint: disable=unused-argument
        if self.is_packed:
            if self.is_integer:
                if self.is_unsigned:
                    return re_quantize_matmul_packed_int_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, self.quantized_matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=self.result_shape)
                else:
                    return re_quantize_matmul_packed_int_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.weights_dtype, self.quantized_matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=self.result_shape)
            else:
                if self.is_unsigned:
                    return re_quantize_matmul_packed_float_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, self.quantized_matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=self.result_shape)
                else:
                    return re_quantize_matmul_packed_float_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.weights_dtype, self.quantized_matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=self.result_shape)
        else:
            if self.is_unsigned:
                return re_quantize_matmul_asymmetric_compiled(weight, scale, zero_point, self.quantized_matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=self.result_shape)
            else:
                return re_quantize_matmul_symmetric_compiled(weight, scale, self.quantized_matmul_dtype, svd_up=svd_up, svd_down=svd_down, result_shape=self.result_shape)

    @devices.inference_context()
    def __call__(self, weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul: bool = False, skip_compile: bool = False, dtype: torch.dtype = None): # pylint: disable=unused-argument
        if dtype is None:
            dtype = self.result_dtype
        re_quantize_for_matmul = self.re_quantize_for_matmul or self.is_packed
        if self.is_packed:
            if self.is_integer:
                if self.is_unsigned:
                    if skip_compile: # compiled training needs to be traced with the original function
                        return dequantize_packed_int_asymmetric(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul)
                    else:
                        return dequantize_packed_int_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul)
                else:
                    if skip_compile:
                        return dequantize_packed_int_symmetric(weight, scale, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)
                    else:
                        return dequantize_packed_int_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)
            else:
                if self.is_unsigned:
                    if skip_compile: # compiled training needs to be traced with the original function
                        return dequantize_packed_float_asymmetric(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul)
                    else:
                        return dequantize_packed_float_asymmetric_compiled(weight, scale, zero_point, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul)
                else:
                    if skip_compile:
                        return dequantize_packed_float_symmetric(weight, scale, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)
                    else:
                        return dequantize_packed_float_symmetric_compiled(weight, scale, self.quantized_weight_shape, self.weights_dtype, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)
        else:
            if self.is_unsigned:
                if skip_compile:
                    return dequantize_asymmetric(weight, scale, zero_point, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul)
                else:
                    return dequantize_asymmetric_compiled(weight, scale, zero_point, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul)
            else:
                if skip_compile:
                    return dequantize_symmetric(weight, scale, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)
                else:
                    return dequantize_symmetric_compiled(weight, scale, svd_up=svd_up, svd_down=svd_down, dtype=dtype, result_shape=self.result_shape, skip_quantized_matmul=skip_quantized_matmul, re_quantize_for_matmul=re_quantize_for_matmul)


dequantize_asymmetric_compiled = compile_func(dequantize_asymmetric)
dequantize_symmetric_compiled = compile_func(dequantize_symmetric)
dequantize_packed_int_asymmetric_compiled = compile_func(dequantize_packed_int_asymmetric)
dequantize_packed_int_symmetric_compiled = compile_func(dequantize_packed_int_symmetric)
dequantize_packed_float_asymmetric_compiled = compile_func(dequantize_packed_float_asymmetric)
dequantize_packed_float_symmetric_compiled = compile_func(dequantize_packed_float_symmetric)
re_quantize_matmul_asymmetric_compiled = compile_func(re_quantize_matmul_asymmetric)
re_quantize_matmul_symmetric_compiled = compile_func(re_quantize_matmul_symmetric)
re_quantize_matmul_packed_int_asymmetric_compiled = compile_func(re_quantize_matmul_packed_int_asymmetric)
re_quantize_matmul_packed_int_symmetric_compiled = compile_func(re_quantize_matmul_packed_int_symmetric)
re_quantize_matmul_packed_float_asymmetric_compiled = compile_func(re_quantize_matmul_packed_float_asymmetric)
re_quantize_matmul_packed_float_symmetric_compiled = compile_func(re_quantize_matmul_packed_float_symmetric)

torch.serialization.add_safe_globals([SDNQDequantizer])
