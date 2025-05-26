# pylint: disable=redefined-builtin,no-member

from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import os
import torch
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from diffusers.utils import get_module_from_name
from accelerate.utils import CustomDtype
from modules import devices, shared


debug = os.environ.get('SD_QUANT_DEBUG', None) is not None
torch_dtype_dict = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int4": CustomDtype.INT4,
    "uint4": CustomDtype.INT4,
}

linear_types = ["Linear"]
conv_types = ["Conv1d", "Conv2d", "Conv3d"]
conv_transpose_types = ["ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]
allowed_types = []
allowed_types.extend(linear_types)
allowed_types.extend(conv_types)
allowed_types.extend(conv_transpose_types)


class QuantizationMethod(str, Enum):
    SDNQ = "sdnq"


def sdnq_quantize_layer(layer, num_bits, is_asym_mode, torch_dtype=None, quant_conv=False, group_size=0, use_int8_matmul=False, param_name=None): # pylint: disable=unused-argument
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
            use_int8_matmul = False
            is_conv_type = True
        elif layer_class_name in conv_transpose_types:
            if not quant_conv:
                return layer
            reduction_axes = [i for i in range(layer.weight.ndim) if i != 1]
            use_int8_matmul = False
            is_conv_transpose_type = True
        else:
            is_linear_type = True
            reduction_axes = -1
            channel_size = layer.weight.shape[-1]
            use_int8_matmul = use_int8_matmul and not is_asym_mode and channel_size >= 32 and layer.weight.shape[0] >= 32

            if not use_int8_matmul and (group_size > 0 or (num_bits == 4 and group_size != -1)):
                if group_size == 0:
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

        if shared.opts.diffusers_offload_mode != "none":
            return_device = layer.weight.data.device
        else:
            return_device = devices.device
        layer.weight.data = layer.weight.data.to(devices.device, dtype=torch.float32)

        if is_asym_mode:
            scale, zero_point = get_int_scale_asymmetric(layer.weight, reduction_axes, num_bits)
        else:
            scale = get_int_scale_symmetric(layer.weight, reduction_axes, num_bits)
            zero_point = None
        compressed_weight = quantize_int(layer.weight, scale, zero_point, is_asym_mode, num_bits)

        if not shared.opts.sdnq_decompress_fp32:
            scale = scale.to(torch_dtype)
            if zero_point is not None:
                zero_point = zero_point.to(torch_dtype)

        if use_int8_matmul:
            scale = scale.squeeze(-1)
            if num_bits == 8:
                compressed_weight = compressed_weight.transpose(0,1)

        if num_bits == 4:
            if is_asym_mode:
                decompressor = INT4AsymmetricWeightsDecompressor(
                    scale=scale.data,
                    zero_point=zero_point.data,
                    compressed_weight_shape=compressed_weight.shape,
                    result_dtype=torch_dtype,
                    result_shape=result_shape,
                )
            else:
                decompressor = INT4SymmetricWeightsDecompressor(
                    scale=scale.data,
                    compressed_weight_shape=compressed_weight.shape,
                    result_dtype=torch_dtype,
                    result_shape=result_shape,
                )
        else:
            if is_asym_mode:
                decompressor = INT8AsymmetricWeightsDecompressor(
                    scale=scale.data,
                    zero_point=zero_point.data,
                    result_dtype=torch_dtype,
                    result_shape=result_shape,
                )
            else:
                decompressor = INT8SymmetricWeightsDecompressor(
                    scale=scale.data,
                    result_dtype=torch_dtype,
                    result_shape=result_shape,
                )

        compressed_weight = decompressor.pack_weight(compressed_weight).to(return_device)
        decompressor = decompressor.to(return_device)

        layer.weight.requires_grad = False
        layer.weight.data = compressed_weight
        layer.sdnq_decompressor = decompressor

        if is_linear_type:
            if use_int8_matmul:
                layer.forward = quantized_linear_forward_int8_matmul
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
    return layer


def apply_sdnq_to_module(model, num_bits, is_asym_mode, quant_conv=False):
    has_children = list(model.children())
    if not has_children:
        return model
    for param_name, module in model.named_children():
        if hasattr(module, "weight") and module.weight is not None:
            module = sdnq_quantize_layer(
                module,
                num_bits,
                is_asym_mode,
                torch_dtype=devices.dtype,
                quant_conv=quant_conv,
                group_size=shared.opts.sdnq_quantize_weights_group_size,
                use_int8_matmul=shared.opts.sdnq_decompress_int8_matmul,
                param_name=param_name,
            )
        module = apply_sdnq_to_module(module, num_bits, is_asym_mode, quant_conv=quant_conv)
    return model


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
        return param_name.endswith(".weight")

    def check_quantized_param(self, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(*args, **kwargs)

    def create_quantized_param( # pylint: disable=arguments-differ
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any], # pylint: disable=unused-argument
        unexpected_keys: List[str], # pylint: disable=unused-argument
        **kwargs,
    ):
        # load the model params to target_device first
        layer, tensor_name = get_module_from_name(model, param_name)
        layer._parameters[tensor_name] = torch.nn.Parameter(param_value).to(device=target_device) # pylint: disable=protected-access

        split_param_name = param_name.split(".")
        if param_name not in self.modules_to_not_convert and not any(param in split_param_name for param in self.modules_to_not_convert):
            layer = sdnq_quantize_layer(
                layer,
                self.quantization_config.num_bits,
                self.quantization_config.is_asym_mode,
                torch_dtype=self.torch_dtype,
                group_size=self.quantization_config.group_size,
                use_int8_matmul=self.quantization_config.use_int8_matmul,
                param_name=param_name,
            )

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.70 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype": # pylint: disable=unused-argument,arguments-renamed
        return torch_dtype_dict[self.quantization_config.weights_dtype]

    def update_torch_dtype(self, torch_dtype: "torch.dtype" = None) -> "torch.dtype":
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
        return model

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
            The target dtype for the weights after quantization. Supported values are ("int8", "uint8", "int4", "uint4")
       modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        weights_dtype: str = "int8",
        group_size: int = 0,
        use_int8_matmul: bool = False,
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.weights_dtype = weights_dtype
        self.quant_method = QuantizationMethod.SDNQ
        self.group_size = group_size
        self.use_int8_matmul = use_int8_matmul
        self.modules_to_not_convert = modules_to_not_convert
        self.num_bits = 8 if self.weights_dtype in {"int8", "uint8"} else 4
        self.is_asym_mode = self.weights_dtype in {"uint8", "uint4"}
        self.is_integer = True
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8", "uint8", "int4", "uint4"]
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


def get_int_scale_asymmetric(weight: torch.FloatTensor, reduction_axes: List[int], num_bits: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point = torch.amin(weight, dim=reduction_axes, keepdims=True)
    max_values = torch.amax(weight, dim=reduction_axes, keepdims=True)
    scale = (max_values - zero_point) / (2**num_bits - 1)
    eps = torch.finfo(scale.dtype).eps # prevent divison by 0
    scale = torch.where(torch.abs(scale) < eps, eps, scale)
    return scale, zero_point


def get_int_scale_symmetric(weight: torch.FloatTensor, reduction_axes: List[int], num_bits: int) -> torch.FloatTensor:
    w_abs_min = torch.abs(torch.amin(weight, dim=reduction_axes, keepdims=True))
    w_max = torch.amax(weight, dim=reduction_axes, keepdims=True)
    scale = torch.where(w_abs_min >= w_max, w_abs_min, -w_max) / (2 ** (num_bits - 1))
    eps = torch.finfo(scale.dtype).eps # prevent divison by 0
    scale = torch.where(torch.abs(scale) < eps, eps, scale)
    return scale


def quantize_int(weight: torch.FloatTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, is_asym_mode: bool, num_bits: int, flatten: Optional[bool] = False) -> torch.ByteTensor:
    dtype = torch.uint8 if is_asym_mode else torch.int8
    level_low = 0 if is_asym_mode else -(2 ** (num_bits - 1))
    level_high = 2**num_bits - 1 if is_asym_mode else 2 ** (num_bits - 1) - 1
    if zero_point is not None:
        compressed_weight = torch.sub(weight, zero_point).div_(scale)
    else:
        compressed_weight = torch.div(weight, scale)
    compressed_weight = compressed_weight.round_().clamp_(level_low, level_high).to(dtype)
    if flatten:
        compressed_weight = compressed_weight.flatten(0,-2)
    return compressed_weight


def decompress_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    result = torch.addcmul(zero_point, input.to(dtype=scale.dtype), scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_symmetric(input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    result = input.to(dtype=scale.dtype).mul_(scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_int4_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    return decompress_asymmetric(unpack_uint4(input, shape), scale, zero_point, dtype, result_shape)


def decompress_int4_symmetric(input: torch.Tensor, scale: torch.Tensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    return decompress_symmetric(unpack_int4(input, shape, dtype=scale.dtype), scale, dtype, result_shape)


def pack_uint4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.uint8 type is supported.")
    packed_tensor = tensor.contiguous().reshape(-1, 2)
    packed_tensor = torch.bitwise_and(packed_tensor[..., ::2], 15) | packed_tensor[..., 1::2] << 4
    return packed_tensor


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int8:
        raise RuntimeError(f"Invalid tensor dtype {tensor.type}. torch.int8 type is supported.")
    tensor = tensor + 8
    return pack_uint4(tensor.to(dtype=torch.uint8))


def unpack_uint4(packed_tensor: torch.Tensor, shape: torch.Size, transpose: Optional[bool] = False) -> torch.Tensor:
    result = torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1).reshape(shape)
    if transpose:
        result = result.transpose(0,1)
    return result


def unpack_int4(packed_tensor: torch.Tensor, shape: torch.Size, dtype: Optional[torch.dtype] = torch.int8, transpose: Optional[bool] = False) -> torch.Tensor:
    result = unpack_uint4(packed_tensor, shape).to(dtype=dtype).sub_(8)
    if transpose:
        result = result.transpose(0,1)
    return result


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: torch.FloatTensor) -> Tuple[torch.ByteTensor, torch.FloatTensor]:
    input_scale = torch.div(input.abs().max(dim=-1).values, 127).unsqueeze(-1)
    input = torch.div(input, input_scale).round_().clamp_(-128, 127).to(torch.int8).flatten(0,-2).contiguous()
    scale = torch.mul(input_scale, scale).flatten(0,-2).contiguous()
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def int8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor,
    scale: torch.FloatTensor,
    compressed_weight_shape: torch.Size,
) -> torch.FloatTensor:
    if compressed_weight_shape is not None:
        weight = unpack_int4_compiled(weight, compressed_weight_shape, transpose=True)
    return_dtype = input.dtype
    output_shape = list(input.shape)
    output_shape[-1] = weight.shape[-1]
    input, scale = quantize_int8_matmul_input_compiled(input, scale)
    result = decompress_symmetric_compiled(torch._int_mm(input, weight), scale, return_dtype, output_shape) # pylint: disable=protected-access
    if bias is not None:
        result.add_(bias)
    return result


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return torch.nn.functional.linear(input, self.sdnq_decompressor(self, return_decompressed_only=True, skip_int8_matmul=True), self.bias)
    return int8_matmul(input, self.weight, self.bias, self.sdnq_decompressor.scale, getattr(self.sdnq_decompressor, "compressed_weight_shape", None))


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.linear(input, self.sdnq_decompressor(self, return_decompressed_only=True), self.bias)


def quantized_conv_forward(self, input) -> torch.FloatTensor:
    return self._conv_forward(input, self.sdnq_decompressor(self, return_decompressed_only=True), self.bias)


def quantized_conv_transpose_1d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation)
    return torch.nn.functional.conv_transpose1d(input, self.sdnq_decompressor(self, return_decompressed_only=True), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


def quantized_conv_transpose_2d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
    output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation)
    return torch.nn.functional.conv_transpose2d(input, self.sdnq_decompressor(self, return_decompressed_only=True), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


def quantized_conv_transpose_3d_forward(self, input: torch.FloatTensor, output_size: Optional[list[int]] = None) -> torch.FloatTensor:
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation)
        return torch.nn.functional.conv_transpose3d(input, self.sdnq_decompressor(self, return_decompressed_only=True), self.bias, self.stride, self.padding, output_padding, self.groups, self.dilation)


class INT8AsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
    ):
        super().__init__()
        self.num_bits = 8
        self.quantization_mode = "asymmetric"
        self.scale = scale
        self.zero_point = zero_point
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < 0) | (weight > 255)):
                raise ValueError("Weight values are not in [0, 255].")
        return weight.to(dtype=torch.uint8)

    def forward(self, x, input=None, *args, return_decompressed_only=False): # pylint: disable=keyword-arg-before-vararg,unused-argument
        result = decompress_asymmetric_compiled(x.weight, self.scale, self.zero_point, self.result_dtype, self.result_shape)
        if return_decompressed_only:
            return result
        else:
            x.weight = result


class INT8SymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
    ):
        super().__init__()
        self.num_bits = 8
        self.quantization_mode = "symmetric"
        self.scale = scale
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < -128) | (weight > 127)):
                raise ValueError("Weight values are not in [-128, 127].")
        return weight.to(dtype=torch.int8)

    def forward(self, x, input=None, *args, return_decompressed_only=False, skip_int8_matmul=False): # pylint: disable=unused-argument,keyword-arg-before-vararg
        if skip_int8_matmul:
            return decompress_symmetric_compiled(x.weight.transpose(0,1), self.scale.unsqueeze(-1), self.result_dtype, self.result_shape)
        result = decompress_symmetric_compiled(x.weight, self.scale, self.result_dtype, self.result_shape)
        if return_decompressed_only:
            return result
        else:
            x.weight = result


class INT4AsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        compressed_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
    ):
        super().__init__()
        self.num_bits = 4
        self.quantization_mode = "asymmetric"
        self.scale = scale
        self.zero_point = zero_point
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < 0) | (weight > 15)):
                raise ValueError("Weight values are not in [0, 15].")
        return pack_uint4(weight.to(dtype=torch.uint8))

    def forward(self, x, input=None, *args, return_decompressed_only=False): # pylint: disable=unused-argument,keyword-arg-before-vararg
        result = decompress_int4_asymmetric_compiled(x.weight, self.scale, self.zero_point, self.compressed_weight_shape, self.result_dtype, self.result_shape)
        if return_decompressed_only:
            return result
        else:
            x.weight = result


class INT4SymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        compressed_weight_shape: torch.Size,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
    ):
        super().__init__()
        self.num_bits = 4
        self.quantization_mode = "symmetric"
        self.scale = scale
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < -8) | (weight > 7)):
                raise ValueError("Tensor values are not in [-8, 7].")
        return pack_int4(weight.to(dtype=torch.int8))

    def forward(self, x, input=None, *arg, return_decompressed_only=False, skip_int8_matmul=False): # pylint: disable=keyword-arg-before-vararg,unused-argument
        if skip_int8_matmul:
            return decompress_int4_symmetric_compiled(x.weight, self.scale.unsqueeze(-1), self.compressed_weight_shape, self.result_dtype, self.result_shape)
        result = decompress_int4_symmetric_compiled(x.weight, self.scale, self.compressed_weight_shape, self.result_dtype, self.result_shape)
        if return_decompressed_only:
            return result
        else:
            x.weight = result


if shared.opts.sdnq_decompress_compile:
    try:
        torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit) # pylint: disable=protected-access
        decompress_asymmetric_compiled = torch.compile(decompress_asymmetric, fullgraph=True)
        decompress_symmetric_compiled = torch.compile(decompress_symmetric, fullgraph=True)
        decompress_int4_asymmetric_compiled = torch.compile(decompress_int4_asymmetric, fullgraph=True)
        decompress_int4_symmetric_compiled = torch.compile(decompress_int4_symmetric, fullgraph=True)
        if devices.backend != "ipex": # pytorch uses the cpu device in torch._int_mm op with ipex + torch.compile
            quantize_int8_matmul_input_compiled = quantize_int8_matmul_input
            unpack_int4_compiled = unpack_int4
            int8_matmul = torch.compile(int8_matmul, fullgraph=True)
        else:
            quantize_int8_matmul_input_compiled = torch.compile(quantize_int8_matmul_input, fullgraph=True)
            unpack_int4_compiled = torch.compile(unpack_int4, fullgraph=True)
    except Exception as e:
        shared.log.warning(f"Quantization: type=sdnq Decompress using torch.compile is not available: {e}")
        decompress_asymmetric_compiled = decompress_asymmetric
        decompress_symmetric_compiled = decompress_symmetric
        decompress_int4_asymmetric_compiled = decompress_int4_asymmetric
        decompress_int4_symmetric_compiled = decompress_int4_symmetric
        quantize_int8_matmul_input_compiled = quantize_int8_matmul_input
        unpack_int4_compiled = unpack_int4
else:
    decompress_asymmetric_compiled = decompress_asymmetric
    decompress_symmetric_compiled = decompress_symmetric
    decompress_int4_asymmetric_compiled = decompress_int4_asymmetric
    decompress_int4_symmetric_compiled = decompress_int4_symmetric
    quantize_int8_matmul_input_compiled = quantize_int8_matmul_input
    unpack_int4_compiled = unpack_int4
