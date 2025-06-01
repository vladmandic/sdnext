# pylint: disable=redefined-builtin,no-member,protected-access

from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import sys
import torch
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from diffusers.utils import get_module_from_name
from accelerate.utils import CustomDtype
from modules import devices, shared


dtype_dict = {
    "int8": {"min": -128, "max": 127, "num_bits": 8, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.int8, "is_unsigned": False, "is_integer": True},
    "uint8": {"min": 0, "max": 255, "num_bits": 8, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "int6": {"min": -32, "max": 31, "num_bits": 6, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "uint6": {"min": 0, "max": 63, "num_bits": 6, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "int4": {"min": -8, "max": 7, "num_bits": 4, "target_dtype": CustomDtype.INT4, "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True},
    "uint4": {"min": 0, "max": 15, "num_bits": 4, "target_dtype": CustomDtype.INT4, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint2": {"min": 0, "max": 3, "num_bits": 2, "target_dtype": CustomDtype.INT2, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "uint1": {"min": 0, "max": 1, "num_bits": 1, "target_dtype": CustomDtype.INT2, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True},
    "float8_e4m3fn": {"min": -448, "max": 448, "num_bits": 8, "target_dtype": torch.float8_e4m3fn, "torch_dtype": torch.float8_e4m3fn, "storage_dtype": torch.float8_e4m3fn, "is_unsigned": False, "is_integer": False},
    "float8_e5m2": {"min": -57344, "max": 57344, "num_bits": 8, "target_dtype": torch.float8_e5m2, "torch_dtype": torch.float8_e5m2, "storage_dtype": torch.float8_e5m2, "is_unsigned": False, "is_integer": False},
    "float8_e4m3fnuz": {"min": -240, "max": 240, "num_bits": 8, "target_dtype": CustomDtype.FP8, "torch_dtype": torch.float8_e4m3fnuz, "storage_dtype": torch.float8_e4m3fnuz, "is_unsigned": False, "is_integer": False},
    "float8_e5m2fnuz": {"min": -57344, "max": 57344, "num_bits": 8, "target_dtype": CustomDtype.FP8, "torch_dtype": torch.float8_e5m2fnuz, "storage_dtype": torch.float8_e5m2fnuz, "is_unsigned": False, "is_integer": False},
}

quantized_matmul_dtypes = ("int8", "int6", "int4", "float8_e4m3fn")

linear_types = ("Linear",)
conv_types = ("Conv1d", "Conv2d", "Conv3d")
conv_transpose_types = ("ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d")
allowed_types = linear_types + conv_types + conv_transpose_types


class QuantizationMethod(str, Enum):
    SDNQ = "sdnq"


def sdnq_quantize_layer(layer, weights_dtype="int8", torch_dtype=None, group_size=0, quant_conv=False, use_quantized_matmul=False, param_name=None, pre_mode=False): # pylint: disable=unused-argument
    layer_class_name = layer.__class__.__name__
    if layer_class_name in allowed_types:
        is_conv_type = False
        is_conv_transpose_type = False
        is_linear_type = False
        result_shape = None
        if torch_dtype is None:
            torch_dtype = devices.dtype

        if layer_class_name in conv_types:
            if not quant_conv:
                return layer
            reduction_axes = [i for i in range(layer.weight.ndim) if i != 0]
            use_quantized_matmul = False
            is_conv_type = True
        elif layer_class_name in conv_transpose_types:
            if not quant_conv:
                return layer
            reduction_axes = [i for i in range(layer.weight.ndim) if i != 1]
            use_quantized_matmul = False
            is_conv_transpose_type = True
        else:
            is_linear_type = True
            reduction_axes = -1
            output_channel_size, channel_size = layer.weight.shape
            if use_quantized_matmul:
                use_quantized_matmul = weights_dtype in quantized_matmul_dtypes and channel_size >= 32 and output_channel_size >= 32
                if use_quantized_matmul and not dtype_dict[weights_dtype]["is_integer"]:
                    use_quantized_matmul = output_channel_size % 16 == 0 and channel_size % 16 == 0

            if not use_quantized_matmul and (group_size > 0 or (dtype_dict[weights_dtype]["num_bits"] < 6 and group_size != -1)):
                if group_size == 0:
                    if dtype_dict[weights_dtype]["num_bits"] < 4:
                        group_size = 32
                    else:
                        group_size = 64
                    num_of_groups = channel_size // group_size

                if group_size >= channel_size:
                    group_size = channel_size
                    num_of_groups = 1
                else:
                    num_of_groups = channel_size // group_size
                    while channel_size % group_size != 0: # find something divisible
                        num_of_groups -= 1
                        if num_of_groups <= 1:
                            group_size = channel_size
                            num_of_groups = 1
                            break
                        group_size = channel_size / num_of_groups

                if num_of_groups > 1:
                    result_shape = layer.weight.shape
                    new_shape = list(result_shape)
                    last_dim_index = layer.weight.ndim
                    new_shape[last_dim_index - 1 : last_dim_index] = (int(num_of_groups), int(group_size))
                    layer.weight.data = layer.weight.reshape(new_shape)

        layer.weight.requires_grad = False
        if shared.opts.diffusers_offload_mode in {"none", "model"}:
            return_device = devices.device
        elif pre_mode:
            if shared.opts.device_map == "gpu":
                return_device = devices.device
            elif shared.opts.sdnq_quantize_with_gpu:
                return_device = devices.cpu
            else:
                return_device = layer.weight.device
        else:
            return_device = layer.weight.device
        if not pre_mode:
            if shared.opts.sdnq_quantize_with_gpu:
                layer.weight.data = layer.weight.to(devices.device).to(dtype=torch.float32)
            else:
                layer.weight.data = layer.weight.to(dtype=torch.float32)

        if dtype_dict[weights_dtype]["is_unsigned"]:
            scale, zero_point = get_scale_asymmetric(layer.weight, reduction_axes, weights_dtype)
        else:
            scale = get_scale_symmetric(layer.weight, reduction_axes, weights_dtype)
            zero_point = None
        layer.weight.data = quantize_weight(layer.weight, scale, zero_point, weights_dtype)

        if not shared.opts.sdnq_decompress_fp32 and not (use_quantized_matmul and not dtype_dict[weights_dtype]["is_integer"]):
            scale = scale.to(torch_dtype)
            if zero_point is not None:
                zero_point = zero_point.to(torch_dtype)

        if use_quantized_matmul:
            scale = scale.transpose(0,1)
            if dtype_dict[weights_dtype]["num_bits"] == 8:
                layer.weight.data = layer.weight.transpose(0,1)
            if not dtype_dict[weights_dtype]["is_integer"]:
                stride = layer.weight.stride()
                if stride[0] > stride[1] and stride[1] == 1:
                    layer.weight.data = layer.weight.t().contiguous().t()
                scale = scale.to(torch.float32)

        layer.sdnq_decompressor = decompressor_dict[weights_dtype](
            scale=scale,
            zero_point=zero_point,
            compressed_weight_shape=layer.weight.shape,
            result_dtype=torch_dtype,
            result_shape=result_shape,
            weights_dtype=weights_dtype,
            use_quantized_matmul=use_quantized_matmul,
        )
        layer.weight.data = layer.sdnq_decompressor.pack_weight(layer.weight).to(return_device)
        layer.sdnq_decompressor = layer.sdnq_decompressor.to(return_device)

        if is_linear_type:
            if use_quantized_matmul:
                if dtype_dict[weights_dtype]["is_integer"]:
                    layer.forward = quantized_linear_forward_int8_matmul
                else:
                    if devices.backend == "cpu" or (devices.backend == "cuda" and sys.platform == "win32" and float(torch.__version__[:3]) <= 2.7 and torch.cuda.get_device_capability(devices.device) == (8,9)):
                        layer.forward = quantized_linear_forward_fp8_matmul_tensorwise
                    else:
                        layer.forward = quantized_linear_forward_fp8_matmul
            else:
                layer.forward = quantized_linear_forward
        elif is_conv_type:
            layer.forward = quantized_conv_forward
        elif is_conv_transpose_type:
            if layer_class_name.endswith("1d"):
                layer.forward = quantized_conv_transpose_1d_forward
            elif layer_class_name.endswith("2d"):
                layer.forward = quantized_conv_transpose_2d_forward
            elif layer_class_name.endswith("3d"):
                layer.forward = quantized_conv_transpose_3d_forward
        layer.forward = layer.forward.__get__(layer, layer.__class__)
        devices.torch_gc(force=False, reason=f"SDNQ param_name: {param_name}")
    return layer


def apply_sdnq_to_module(model, weights_dtype="int8", torch_dtype=None, group_size=0, quant_conv=False, use_quantized_matmul=False, param_name=None):
    has_children = list(model.children())
    if not has_children:
        return model
    for module_param_name, module in model.named_children():
        if hasattr(module, "weight") and module.weight is not None:
            module = sdnq_quantize_layer(
                module,
                weights_dtype=weights_dtype,
                torch_dtype=torch_dtype,
                group_size=group_size,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                param_name=module_param_name,
            )
        module = apply_sdnq_to_module(
                module,
                weights_dtype=weights_dtype,
                torch_dtype=torch_dtype,
                group_size=group_size,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                param_name=module_param_name,
            )
    return model


def get_scale_asymmetric(weight: torch.FloatTensor, reduction_axes: List[int], weights_dtype: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point = torch.amin(weight, dim=reduction_axes, keepdims=True)
    scale = torch.amax(weight, dim=reduction_axes, keepdims=True).sub_(zero_point).div_(dtype_dict[weights_dtype]["max"] - dtype_dict[weights_dtype]["min"])
    eps = torch.finfo(scale.dtype).eps # prevent divison by 0
    scale = torch.where(torch.abs(scale) < eps, eps, scale)
    if dtype_dict[weights_dtype]["min"] != 0:
        zero_point.sub_(torch.mul(scale, dtype_dict[weights_dtype]["min"]))
    return scale, zero_point


def get_scale_symmetric(weight: torch.FloatTensor, reduction_axes: List[int], weights_dtype: str) -> torch.FloatTensor:
    abs_min_values = torch.amin(weight, dim=reduction_axes, keepdims=True).abs_()
    max_values = torch.amax(weight, dim=reduction_axes, keepdims=True)
    scale = torch.where(abs_min_values >= max_values, abs_min_values, -max_values).div_(dtype_dict[weights_dtype]["max"])
    eps = torch.finfo(scale.dtype).eps # prevent divison by 0
    scale = torch.where(torch.abs(scale) < eps, eps, scale)
    return scale


def quantize_weight(weight: torch.FloatTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, weights_dtype: str) -> torch.ByteTensor:
    if zero_point is not None:
        compressed_weight = torch.sub(weight, zero_point).div_(scale)
    else:
        compressed_weight = torch.div(weight, scale)
    if dtype_dict[weights_dtype]["is_integer"]:
        compressed_weight.round_()
    compressed_weight = compressed_weight.clamp_(dtype_dict[weights_dtype]["min"], dtype_dict[weights_dtype]["max"]).to(dtype_dict[weights_dtype]["torch_dtype"])
    return compressed_weight


def decompress_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    result = torch.addcmul(zero_point, input.to(dtype=scale.dtype), scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_symmetric(input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, result_shape: torch.Size, skip_quantized_matmul: bool = False) -> torch.Tensor:
    if skip_quantized_matmul:
        result = input.transpose(0,1).to(dtype=scale.dtype).mul_(scale.transpose(0,1)).to(dtype=dtype)
    else:
        result = input.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_packed_int_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str) -> torch.Tensor:
    return decompress_asymmetric(packed_int_function_dict[weights_dtype]["unpack"](input, shape), scale, zero_point, dtype, result_shape)


def decompress_packed_int_symmetric(input: torch.Tensor, scale: torch.Tensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size, weights_dtype: str, skip_quantized_matmul: bool = False) -> torch.Tensor:
    if skip_quantized_matmul:
        return decompress_symmetric(packed_int_function_dict[weights_dtype]["unpack"](input, shape, dtype=scale.dtype), scale.transpose(0,1), dtype, result_shape)
    else:
        return decompress_symmetric(packed_int_function_dict[weights_dtype]["unpack"](input, shape, dtype=scale.dtype), scale, dtype, result_shape)


def pack_uint6(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.uint8 type is supported.")
    packed_tensor = tensor.contiguous().reshape(-1, 4)
    packed_tensor = torch.stack(
        (
            torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 3], 2), 192)),
            torch.bitwise_or(packed_tensor[:, 1], torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 3], 4), 192)),
            torch.bitwise_or(packed_tensor[:, 2], torch.bitwise_left_shift(packed_tensor[:, 3], 6)),
        ),
        dim=-1
    )
    return packed_tensor


def pack_int6(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.int8 type is supported.")
    return pack_uint6((tensor + 32).to(dtype=torch.uint8))


def pack_uint4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.uint8 type is supported.")
    packed_tensor = tensor.contiguous().reshape(-1, 2)
    packed_tensor = torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 4))
    return packed_tensor


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.int8 type is supported.")
    return pack_uint4((tensor + 8).to(dtype=torch.uint8))


def pack_uint2(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.uint8 type is supported.")
    packed_tensor = tensor.contiguous().reshape(-1, 4)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 2)),
        torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 4), torch.bitwise_left_shift(packed_tensor[:, 3], 6)),
    )
    return packed_tensor


def pack_uint1(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.uint8 type is supported.")
    packed_tensor = tensor.contiguous().reshape(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 1)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 2), torch.bitwise_left_shift(packed_tensor[:, 3], 3))
        ),
        torch.bitwise_or(
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 4], 4), torch.bitwise_left_shift(packed_tensor[:, 5], 5)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 6], 6), torch.bitwise_left_shift(packed_tensor[:, 7], 7))
        ),
    )
    return packed_tensor


def unpack_uint6(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.stack(
        (
            torch.bitwise_and(packed_tensor[:, 0], 63),
            torch.bitwise_and(packed_tensor[:, 1], 63),
            torch.bitwise_and(packed_tensor[:, 2], 63),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 2), 48),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 4), 12),
                ),
                torch.bitwise_right_shift(packed_tensor[:, 2], 6)
            )
        ),
        dim=-1
    ).reshape(shape)
    return result


def unpack_int6(packed_tensor: torch.Tensor, shape: torch.Size, dtype: Optional[torch.dtype] = torch.int8, transpose: Optional[bool] = False) -> torch.Tensor:
    result = unpack_uint6(packed_tensor, shape).to(dtype=dtype).sub_(32)
    if transpose:
        result = result.transpose(0,1)
    return result


def unpack_uint4(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1).reshape(shape)
    return result


def unpack_int4(packed_tensor: torch.Tensor, shape: torch.Size, dtype: Optional[torch.dtype] = torch.int8, transpose: Optional[bool] = False) -> torch.Tensor:
    result = unpack_uint4(packed_tensor, shape).to(dtype=dtype).sub_(8)
    if transpose:
        result = result.transpose(0,1)
    return result


def unpack_uint2(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.stack(
        (
            torch.bitwise_and(packed_tensor, 3),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 2), 3),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 4), 3),
            torch.bitwise_right_shift(packed_tensor, 6),
        ),
        dim=-1
    ).reshape(shape)
    return result


def unpack_uint1(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.stack(
        (
            torch.bitwise_and(packed_tensor, 1),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 1), 1),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 2), 1),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 3), 1),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 4), 1),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 5), 1),
            torch.bitwise_and(torch.bitwise_right_shift(packed_tensor, 6), 1),
            torch.bitwise_right_shift(packed_tensor, 7),
        ),
        dim=-1
    ).reshape(shape)
    return result


def quantize_fp8_matmul_input(input: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous()
    input_scale = torch.div(input.abs().amax(dim=-1, keepdims=True), 448)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(torch.float8_e4m3fn)
    input_scale = input_scale.to(torch.float32)
    return input, input_scale


def quantize_fp8_matmul_input_tensorwise(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.ByteTensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous()
    input_scale = torch.div(input.abs().amax(dim=-1, keepdims=True), 448)
    input = torch.div(input, input_scale).clamp_(-448, 448).to(torch.float8_e4m3fn)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.ByteTensor, torch.FloatTensor]:
    input = input.flatten(0,-2).contiguous()
    input_scale = torch.div(input.abs().amax(dim=-1, keepdims=True), 127)
    input = torch.div(input, input_scale).round_().clamp_(-128, 127).to(torch.int8)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def fp8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    input, input_scale = quantize_fp8_matmul_input(input)
    return torch._scaled_mm(input, weight, input_scale, scale, bias=bias, out_dtype=return_dtype).reshape(output_shape)


# sm89 doesn't support row wise scale in Windows
def fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)
    input, scale = quantize_fp8_matmul_input_tensorwise(input, scale)
    result = decompress_symmetric(torch._scaled_mm(input, weight, dummy_input_scale, dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, return_dtype, output_shape)
    if bias is not None:
        result.add_(bias)
    return result


def int8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    compressed_weight_shape: torch.Size,
    weights_dtype: str,
) -> torch.FloatTensor:
    if compressed_weight_shape is not None:
        weight = packed_int_function_dict[weights_dtype]["unpack"](weight, compressed_weight_shape, transpose=True)
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    input, scale = quantize_int8_matmul_input(input, scale)
    result = decompress_symmetric(torch._int_mm(input, weight), scale, return_dtype, output_shape)
    if bias is not None:
        result.add_(bias)
    return result


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_decompressor(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul(input, self.weight, self.bias, self.sdnq_decompressor.scale)


def quantized_linear_forward_fp8_matmul_tensorwise(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_decompressor(self.weight, skip_quantized_matmul=True), self.bias)
    return fp8_matmul_tensorwise(input, self.weight, self.bias, self.sdnq_decompressor.scale)


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_decompressor(self.weight, skip_quantized_matmul=True), self.bias)
    return int8_matmul(input, self.weight, self.bias, self.sdnq_decompressor.scale, getattr(self.sdnq_decompressor, "compressed_weight_shape", None), self.sdnq_decompressor.weights_dtype)


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_decompressor(self.weight), self.bias)


def quantized_conv_forward(self, input) -> torch.FloatTensor:
    return self._conv_forward(input, self.sdnq_decompressor(self.weight), self.bias)


def quantized_conv_transpose_1d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation)
    return torch.nn.functional.conv_transpose1d(input, self.sdnq_decompressor(self.weight), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


def quantized_conv_transpose_2d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation)
    return torch.nn.functional.conv_transpose2d(input, self.sdnq_decompressor(self.weight), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


def quantized_conv_transpose_3d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation)
    return torch.nn.functional.conv_transpose3d(input, self.sdnq_decompressor(self.weight), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


class AsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        **kwargs,
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

    def forward(self, weight, **kwargs):
        return decompress_asymmetric_compiled(weight, self.scale, self.zero_point, self.result_dtype, self.result_shape)


class SymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        use_quantized_matmul: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = use_quantized_matmul
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"])

    def forward(self, weight, skip_quantized_matmul=False, **kwargs):
        return decompress_symmetric_compiled(weight, self.scale, self.result_dtype, self.result_shape, skip_quantized_matmul=skip_quantized_matmul)


class PackedINTAsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        compressed_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        **kwargs,
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

    def forward(self, weight, **kwargs):
        return decompress_packed_int_asymmetric_compiled(weight, self.scale, self.zero_point, self.compressed_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype)


class PackedINTSymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        compressed_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
        weights_dtype: str,
        use_quantized_matmul: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.weights_dtype = weights_dtype
        self.use_quantized_matmul = use_quantized_matmul
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape
        self.register_buffer("scale", scale)

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return packed_int_function_dict[self.weights_dtype]["pack"](weight.to(dtype=dtype_dict[self.weights_dtype]["torch_dtype"]))

    def forward(self, weight, skip_quantized_matmul=False, **kwargs):
        return decompress_packed_int_symmetric_compiled(weight, self.scale, self.compressed_weight_shape, self.result_dtype, self.result_shape, self.weights_dtype, skip_quantized_matmul=skip_quantized_matmul)


decompressor_dict = {
    "int8": SymmetricWeightsDecompressor,
    "uint8": AsymmetricWeightsDecompressor,
    "int6": PackedINTSymmetricWeightsDecompressor,
    "uint6": PackedINTAsymmetricWeightsDecompressor,
    "int4": PackedINTSymmetricWeightsDecompressor,
    "uint4": PackedINTAsymmetricWeightsDecompressor,
    "uint2": PackedINTAsymmetricWeightsDecompressor,
    "uint1": PackedINTAsymmetricWeightsDecompressor,
    "float8_e4m3fn": SymmetricWeightsDecompressor,
    "float8_e4m3fnuz": SymmetricWeightsDecompressor,
    "float8_e5m2": SymmetricWeightsDecompressor,
    "float8_e5m2fnuz": SymmetricWeightsDecompressor,
}


packed_int_function_dict = {
    "int6": {"pack": pack_int6, "unpack": unpack_int6},
    "uint6": {"pack": pack_uint6, "unpack": unpack_uint6},
    "int4": {"pack": pack_int4, "unpack": unpack_int4},
    "uint4": {"pack": pack_uint4, "unpack": unpack_uint4},
    "uint2": {"pack": pack_uint2, "unpack": unpack_uint2},
    "uint1": {"pack": pack_uint1, "unpack": unpack_uint1},
}


class SDNQQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for SDNQ
    """

    requires_parameters_quantization = True
    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = None
    torch_dtype = None

    def __init__(self, quantization_config, **kwargs): # pylint: disable=useless-parent-delegation
        super().__init__(quantization_config, **kwargs)

    def check_if_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        if param_name.endswith(".weight"):
            split_param_name = param_name.split(".")
            if param_name not in self.modules_to_not_convert and not any(param in split_param_name for param in self.modules_to_not_convert):
                layer_class_name = get_module_from_name(model, param_name)[0].__class__.__name__
                if layer_class_name in allowed_types:
                    if layer_class_name in conv_types or layer_class_name in conv_transpose_types:
                        if self.quantization_config.quant_conv:
                            return True
                    else:
                        return True
        param_value.data = param_value.clone() # safetensors is unable to release the cpu memory without this
        return False

    def check_quantized_param(self, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(*args, **kwargs)

    def create_quantized_param( # pylint: disable=arguments-differ
        self,
        model,
        param_value: torch.FloatTensor,
        param_name: str,
        target_device: torch.device,
        state_dict: Dict[str, Any], # pylint: disable=unused-argument
        unexpected_keys: List[str], # pylint: disable=unused-argument
        **kwargs,
    ):
        # load the model params to target_device first
        layer, _ = get_module_from_name(model, param_name)
        if shared.opts.sdnq_quantize_with_gpu:
            if param_value.dtype == torch.float32 and devices.same_device(param_value.device, devices.device):
                param_value = param_value.clone()
            else:
                param_value = param_value.to(devices.device).to(dtype=torch.float32)
        else:
            if param_value.dtype == torch.float32 and devices.same_device(param_value.device, target_device):
                param_value = param_value.clone()
            else:
                param_value = param_value.to(target_device).to(dtype=torch.float32)
        layer.weight = torch.nn.Parameter(param_value, requires_grad=False)
        layer = sdnq_quantize_layer(
            layer,
            weights_dtype=self.quantization_config.weights_dtype,
            torch_dtype=self.torch_dtype,
            group_size=self.quantization_config.group_size,
            quant_conv=self.quantization_config.quant_conv,
            use_quantized_matmul=self.quantization_config.use_quantized_matmul,
            param_name=param_name,
            pre_mode=True,
        )

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.80 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: torch.dtype) -> torch.dtype: # pylint: disable=unused-argument,arguments-renamed
        return dtype_dict[self.quantization_config.weights_dtype]["target_dtype"]

    def update_torch_dtype(self, torch_dtype: torch.dtype = None) -> torch.dtype:
        if torch_dtype is None:
            torch_dtype = devices.dtype
        self.torch_dtype = torch_dtype
        return torch_dtype

    def _process_model_before_weight_loading( # pylint: disable=arguments-differ
        self,
        model,
        device_map, # pylint: disable=unused-argument
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        model.config.quantization_config = self.quantization_config
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        if keep_in_fp32_modules is not None:
            self.modules_to_not_convert.extend(keep_in_fp32_modules)

    def _process_model_after_weight_loading(self, model, **kwargs):
        if shared.opts.diffusers_offload_mode != "none":
            model = model.to(devices.cpu)
        devices.torch_gc(force=True)
        return model

    def get_cuda_warm_up_factor(self):
        return 32 // dtype_dict[self.quantization_config.weights_dtype]["num_bits"]

    def update_tp_plan(self, config):
        """
        needed for transformers compatibilty, no-op function
        """
        return config

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str) -> List[str]: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return unexpected_keys

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return missing_keys

    def update_expected_keys(self, model, expected_keys: List[str], loaded_keys: List[str]) -> List[str]: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return expected_keys

    @property
    def is_trainable(self):
        return False

    @property
    def is_serializable(self):
        return False


@dataclass
class SDNQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `sdnq`.

    Args:
        weights_dtype (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are:
            ("int8", "uint8", "int6", "uint6", "int4", "uint4", "uint2", "uint1", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
       modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        weights_dtype: str = "int8",
        group_size: int = 0,
        quant_conv: bool = False,
        use_quantized_matmul: bool = False,
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.weights_dtype = weights_dtype
        self.quant_method = QuantizationMethod.SDNQ
        self.group_size = group_size
        self.quant_conv = quant_conv
        self.use_quantized_matmul = use_quantized_matmul
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()
        self.is_integer = dtype_dict[self.weights_dtype]["is_integer"]

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8", "uint8", "int6", "uint6", "int4", "uint4", "uint2", "uint1", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"]
        if self.weights_dtype not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights_dtype}")


class SDNQ_T5DenseGatedActDense(torch.nn.Module): # forward can't find what self is without creating a class
    def __init__(self, T5DenseGatedActDense, dtype):
        super().__init__()
        self.wi_0 = T5DenseGatedActDense.wi_0
        self.wi_1 = T5DenseGatedActDense.wi_1
        self.wo = T5DenseGatedActDense.wo
        self.dropout = T5DenseGatedActDense.dropout
        self.act = T5DenseGatedActDense.act
        self.torch_dtype = dtype

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.to(self.torch_dtype) # this line needs to be forced
        hidden_states = self.wo(hidden_states)
        return hidden_states


if shared.opts.sdnq_decompress_compile:
    try:
        torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
        decompress_asymmetric_compiled = torch.compile(decompress_asymmetric, fullgraph=True)
        decompress_symmetric_compiled = torch.compile(decompress_symmetric, fullgraph=True)
        decompress_packed_int_asymmetric_compiled = torch.compile(decompress_packed_int_asymmetric, fullgraph=True)
        decompress_packed_int_symmetric_compiled = torch.compile(decompress_packed_int_symmetric, fullgraph=True)
        fp8_matmul = torch.compile(fp8_matmul, fullgraph=True)
        fp8_matmul_tensorwise = torch.compile(fp8_matmul_tensorwise, fullgraph=True)
        int8_matmul = torch.compile(int8_matmul, fullgraph=True)
    except Exception as e:
        shared.log.warning(f"Quantization: type=sdnq Decompress using torch.compile is not available: {e}")
        decompress_asymmetric_compiled = decompress_asymmetric
        decompress_symmetric_compiled = decompress_symmetric
        decompress_packed_int_asymmetric_compiled = decompress_packed_int_asymmetric
        decompress_packed_int_symmetric_compiled = decompress_packed_int_symmetric
else:
    decompress_asymmetric_compiled = decompress_asymmetric
    decompress_symmetric_compiled = decompress_symmetric
    decompress_packed_int_asymmetric_compiled = decompress_packed_int_asymmetric
    decompress_packed_int_symmetric_compiled = decompress_packed_int_symmetric
