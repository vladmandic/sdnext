from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import os
import torch
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from diffusers.utils import get_module_from_name

from accelerate import init_empty_weights
from accelerate.utils import CustomDtype

from modules import devices, shared


debug = os.environ.get('SD_QUANT_DEBUG', None) is not None

torch_dtype_dict = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int4": CustomDtype.INT4,
    "uint4": CustomDtype.INT4,
}

weights_dtype_dict = {
    "int8_asym": "uint8",
    "int8_sym": "int8",
    "int4_asym": "uint4",
    "int4_sym": "int4",
    "int8": "uint8",
    "int4": "uint4",
}

linear_types = ["NNCFLinear", "Linear"]
conv_types = ["NNCFConv1d", "NNCFConv2d", "NNCFConv3d", "Conv1d", "Conv2d", "Conv3d"]
conv_transpose_types = ["NNCFConvTranspose1d", "NNCFConvTranspose2d", "NNCFConvTranspose3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"]

allowed_types = []
allowed_types.extend(linear_types)
allowed_types.extend(conv_types)
allowed_types.extend(conv_transpose_types)

class QuantizationMethod(str, Enum):
    NNCF = "nncf"


# de-abstracted and modified from the actual quant functions of nncf 2.16.0:
def nncf_compress_layer(layer, num_bits, is_asym_mode, torch_dtype=None, quant_conv=False, param_name=None):
    if layer.__class__.__name__ in allowed_types:
        if torch_dtype is None:
            torch_dtype = devices.dtype
        result_shape = None

        if layer.__class__.__name__ in conv_types:
            if is_asym_mode or not quant_conv: # don't quant convs with asym mode
                return layer
            reduction_axes = [i for i in range(layer.weight.ndim) if i != 0]
        if layer.__class__.__name__ in conv_transpose_types:
            if is_asym_mode or not quant_conv: # don't quant convs with asym mode
                return layer
            reduction_axes = [i for i in range(layer.weight.ndim) if i != 1]
        else:
            reduction_axes = -1
            if shared.opts.nncf_compress_weights_num_of_groups > 1:
                num_of_groups = shared.opts.nncf_compress_weights_num_of_groups
                channel_size = layer.weight.shape[-1]

                group_size = channel_size / num_of_groups
                while channel_size % group_size != 0: # find something divisible
                    num_of_groups -= 1
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
            level_low = 0
            level_high = 2**num_bits - 1
            min_values = torch.amin(layer.weight, dim=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
            max_values = torch.amax(layer.weight, dim=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]

            levels = level_high - level_low + 1
            scale = ((max_values - min_values) / (levels - 1)).to(dtype=torch.float32)
            eps = torch.finfo(scale.dtype).eps

            scale = torch.where(torch.abs(scale) < eps, eps, scale)
            zero_point = level_low - torch.round(min_values / scale)
            #this is for packing zero_point to int:
            #zero_point = torch.clip(zero_point.to(dtype=torch.int32), level_low, level_high)
            zero_point = zero_point.to(dtype=torch.float32)
        else:
            factor = 2 ** (num_bits - 1)

            w_abs_min = torch.abs(torch.amin(layer.weight, dim=reduction_axes, keepdims=True))
            w_max = torch.amax(layer.weight, dim=reduction_axes, keepdims=True)

            scale = torch.where(w_abs_min >= w_max, w_abs_min, -w_max)
            scale /= factor

            eps = torch.finfo(scale.dtype).eps
            scale = torch.where(torch.abs(scale) < eps, eps, scale)
            zero_point = None

        dtype = torch.uint8 if is_asym_mode else torch.int8
        level_low = 0 if is_asym_mode else -(2 ** (num_bits - 1))
        level_high = 2**num_bits - 1 if is_asym_mode else 2 ** (num_bits - 1) - 1

        compressed_weight = layer.weight.data / scale
        if not shared.opts.nncf_decompress_fp32:
           scale = scale.to(torch_dtype)

        if zero_point is not None:
            compressed_weight += zero_point
            zero_point = zero_point.to(scale.dtype)

        compressed_weight = torch.round(compressed_weight)
        compressed_weight = torch.clip(compressed_weight, level_low, level_high).to(dtype)

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

        compressed_weight = decompressor.pack_weight(compressed_weight)
        compressed_weight = compressed_weight.to(return_device)

        decompressor = decompressor.to(return_device)
        layer.register_pre_forward_operation(decompressor)

        layer.weight.requires_grad = False
        layer.weight.data = compressed_weight
    return layer


def apply_nncf_to_module(model, num_bits, is_asym_mode, quant_conv=False):
    has_children = list(model.children())
    if not has_children:
        return model
    for param_name, module in model.named_children():
        if module.__class__.__name__.startswith("NNCF") and hasattr(module, "weight") and module.weight is not None:
            module = nncf_compress_layer(module, num_bits, is_asym_mode, torch_dtype=devices.dtype, quant_conv=quant_conv, param_name=param_name)
        module = apply_nncf_to_module(module, num_bits, is_asym_mode, quant_conv=quant_conv)
    return model


def nncf_send_to_device(model, device):
    for child in model.children():
        if "WeightsDecompressor" in child.__class__.__name__:
            child.scale = child.scale.to(device)
            if hasattr(child, "zero_point"):
                child.zero_point = child.zero_point.to(device)
        nncf_send_to_device(child, device)


class NNCFQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for NNCF
    """

    requires_parameters_quantization = True
    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["nncf"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def check_if_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        return module.__class__.__name__.startswith("NNCF") and param_name.endswith(".weight")

    def check_quantized_param(self, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(*args, **kwargs)

    def create_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
        **kwargs,
    ):
        # load the model params to target_device first
        layer, tensor_name = get_module_from_name(model, param_name)
        layer._parameters[tensor_name] = torch.nn.Parameter(param_value).to(device=target_device)

        split_param_name = param_name.split(".")
        if param_name not in self.modules_to_not_convert and not any(param in split_param_name for param in self.modules_to_not_convert):
            layer = nncf_compress_layer(
                layer,
                self.quantization_config.num_bits,
                self.quantization_config.is_asym_mode,
                torch_dtype=self.torch_dtype,
                param_name=param_name
            )

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.70 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        return torch_dtype_dict[self.quantization_config.weights_dtype]

    def update_torch_dtype(self, torch_dtype: "torch.dtype" = None) -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = devices.dtype
        self.torch_dtype = torch_dtype
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model,
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules

        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        if keep_in_fp32_modules is not None:
            self.modules_to_not_convert.extend(keep_in_fp32_modules)

        model.config.quantization_config = self.quantization_config
        with init_empty_weights():
            model, _ = replace_modules_by_nncf_modules(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        nncf_send_to_device(model, devices.device)
        return model

    def update_tp_plan(self, config):
        """
        needed for transformers compatibilty, no-op function
        """
        return config

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str) -> List[str]:
        """
        needed for transformers compatibilty, no-op function
        """
        return unexpected_keys

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        needed for transformers compatibilty, no-op function
        """
        return missing_keys

    def update_expected_keys(self, model, expected_keys: List[str], loaded_keys: List[str]) -> List[str]:
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
class NNCFConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `nncf`.

    Args:
        weights_dtype (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("int8", "int8_sym", "int4", "int4_sym")
       modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__(
        self,
        weights_dtype: str = "int8_sym",
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.NNCF
        self.weights_dtype = weights_dtype_dict[weights_dtype.lower()]
        self.modules_to_not_convert = modules_to_not_convert

        self.post_init()

        self.num_bits = 8 if self.weights_dtype in {"int8", "uint8"} else 4
        self.is_asym_mode = self.weights_dtype in {"uint8", "uint4"}
        self.is_integer = True
        self.group_size = -1

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8", "uint8", "int4", "uint4"]
        if self.weights_dtype not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights_dtype}")


class NNCF_T5DenseGatedActDense(torch.nn.Module): # forward can't find what self is without creating a class
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


# WeightsDecompressor classes and functions are modified from NNCF 2.16.0

def unpack_uint4(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1).reshape(shape)


def unpack_int4(packed_tensor: torch.Tensor, shape: torch.Size, dtype: Optional[torch.dtype] = torch.int8) -> torch.Tensor:
    return unpack_uint4(packed_tensor, shape).to(dtype=dtype) - 8


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


def decompress_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    result = torch.mul(torch.sub(input.to(dtype=scale.dtype), zero_point), scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_symmetric(input: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    result = torch.mul(input.to(dtype=scale.dtype), scale).to(dtype=dtype)
    if result_shape is not None:
        result = result.reshape(result_shape)
    return result


def decompress_int4_asymmetric(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    return decompress_asymmetric(unpack_uint4(input, shape), scale, zero_point, dtype, result_shape)


def decompress_int4_symmetric(input: torch.Tensor, scale: torch.Tensor, shape: torch.Size, dtype: torch.dtype, result_shape: torch.Size) -> torch.Tensor:
    return decompress_symmetric(unpack_int4(input, shape, dtype=scale.dtype), scale, dtype, result_shape)


if shared.opts.nncf_decompress_compile:
    try:
        torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit) # pylint: disable=protected-access
        decompress_asymmetric = torch.compile(decompress_asymmetric, fullgraph=True)
        decompress_symmetric = torch.compile(decompress_symmetric, fullgraph=True)
        decompress_int4_asymmetric = torch.compile(decompress_int4_asymmetric, fullgraph=True)
        decompress_int4_symmetric = torch.compile(decompress_int4_symmetric, fullgraph=True)
    except Exception as e:
        shared.log.warning(f"Quantization: type=nncf Decompress using torch.compile is not available: {e}")


class INT8AsymmetricWeightsDecompressor(torch.nn.Module):
    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        result_dtype: torch.dtype,
        result_shape: torch.Size,
    ):
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    @property
    def num_bits(self):
        return 8

    @property
    def quantization_mode(self):
        return "asymmetric"

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < 0) | (weight > 255)):
                raise ValueError("Weight values are not in [0, 255].")
        return weight.to(dtype=torch.uint8)

    def forward(self, x, *args, return_decompressed_only=False):
        result = decompress_asymmetric(x.weight, self.scale, self.zero_point, self.result_dtype, self.result_shape)
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
        self.scale = scale
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    @property
    def num_bits(self):
        return 8

    @property
    def quantization_mode(self):
        return "symmetric"

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < -128) | (weight > 127)):
                raise ValueError("Weight values are not in [-128, 127].")
        return weight.to(dtype=torch.int8)

    def forward(self, x, *args, return_decompressed_only=False):
        result = decompress_symmetric(x.weight, self.scale, self.result_dtype, self.result_shape)
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
        self.scale = scale
        self.zero_point = zero_point
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    @property
    def num_bits(self):
        return 4

    @property
    def quantization_mode(self):
        return "asymmetric"

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < 0) | (weight > 15)):
                raise ValueError("Weight values are not in [0, 15].")
        return pack_uint4(weight.to(dtype=torch.uint8))

    def forward(self, x, *args, return_decompressed_only=False):
        result = decompress_int4_asymmetric(x.weight, self.scale, self.zero_point, self.compressed_weight_shape, self.result_dtype, self.result_shape)
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
        self.scale = scale
        self.compressed_weight_shape = compressed_weight_shape
        self.result_dtype = result_dtype
        self.result_shape = result_shape

    @property
    def num_bits(self):
        return 4

    @property
    def quantization_mode(self):
        return "symmetric"

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if debug:
            if torch.any((weight < -8) | (weight > 7)):
                raise ValueError("Tensor values are not in [-8, 7].")
        return pack_int4(weight.to(dtype=torch.int8))

    def forward(self, x, *arg, return_decompressed_only=False):
        result = decompress_int4_symmetric(x.weight, self.scale, self.compressed_weight_shape, self.result_dtype, self.result_shape)
        if return_decompressed_only:
            return result
        else:
            x.weight = result
