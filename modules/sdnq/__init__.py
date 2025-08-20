# pylint: disable=redefined-builtin,no-member,protected-access

from typing import Any, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import re
import torch
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from diffusers.utils import get_module_from_name
from modules import devices, shared

from .common import dtype_dict, use_tensorwise_fp8_matmul, quantized_matmul_dtypes, allowed_types, conv_types, conv_transpose_types
from .dequantizer import dequantizer_dict
from .forward import get_forward_func


class QuantizationMethod(str, Enum):
    SDNQ = "sdnq"


def get_scale_asymmetric(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point = torch.amin(weight, dim=reduction_axes, keepdims=True)
    scale = torch.amax(weight, dim=reduction_axes, keepdims=True).sub_(zero_point).div_(dtype_dict[weights_dtype]["max"] - dtype_dict[weights_dtype]["min"])
    if dtype_dict[weights_dtype]["min"] != 0:
        zero_point.sub_(torch.mul(scale, dtype_dict[weights_dtype]["min"]))
    return scale, zero_point


def get_scale_symmetric(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> torch.FloatTensor:
    return torch.amax(weight.abs(), dim=reduction_axes, keepdims=True).div_(dtype_dict[weights_dtype]["max"])


def quantize_weight(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> Tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if dtype_dict[weights_dtype]["is_unsigned"]:
        scale, zero_point = get_scale_asymmetric(weight, reduction_axes, weights_dtype)
        quantized_weight = torch.sub(weight, zero_point).div_(scale)
    else:
        scale = get_scale_symmetric(weight, reduction_axes, weights_dtype)
        quantized_weight = torch.div(weight, scale)
        zero_point = None
    if dtype_dict[weights_dtype]["is_integer"]:
        quantized_weight.round_()
    quantized_weight = quantized_weight.clamp_(dtype_dict[weights_dtype]["min"], dtype_dict[weights_dtype]["max"]).to(dtype_dict[weights_dtype]["torch_dtype"])
    return quantized_weight, scale, zero_point


@devices.inference_context()
def sdnq_quantize_layer(layer, weights_dtype="int8", torch_dtype=None, group_size=0, quant_conv=False, use_quantized_matmul=False, use_quantized_matmul_conv=False, dequantize_fp32=False, non_blocking=False, quantization_device=None, return_device=None, param_name=None): # pylint: disable=unused-argument
    layer_class_name = layer.__class__.__name__
    if layer_class_name in allowed_types:
        is_conv_type = False
        is_conv_transpose_type = False
        is_linear_type = False
        result_shape = None
        original_shape = layer.weight.shape
        if torch_dtype is None:
            torch_dtype = devices.dtype

        if layer_class_name in conv_types:
            if not quant_conv:
                return layer
            if dtype_dict[weights_dtype]["num_bits"] < 4:
                weights_dtype = "uint4"
            is_conv_type = True
            reduction_axes = 1
            output_channel_size, channel_size = layer.weight.shape[:2]
            group_channel_size = channel_size // layer.groups
            use_quantized_matmul = False
            if use_quantized_matmul_conv:
                use_quantized_matmul = weights_dtype in quantized_matmul_dtypes and group_channel_size >= 32 and output_channel_size >= 32
                if use_quantized_matmul and not dtype_dict[weights_dtype]["is_integer"]:
                    use_quantized_matmul = output_channel_size % 16 == 0 and group_channel_size % 16 == 0
                if use_quantized_matmul:
                    result_shape = layer.weight.shape
                    layer.weight.data = layer.weight.reshape(output_channel_size, -1)
        elif layer_class_name in conv_transpose_types:
            if not quant_conv:
                return layer
            if dtype_dict[weights_dtype]["num_bits"] < 4:
                weights_dtype = "uint4"
            is_conv_transpose_type = True
            reduction_axes = 0
            channel_size, output_channel_size = layer.weight.shape[:2]
            use_quantized_matmul = False
        else:
            is_linear_type = True
            reduction_axes = -1
            try:
                output_channel_size, channel_size = layer.weight.shape
            except Exception as e:
                raise ValueError(f"SDNQ: layer_class_name={layer_class_name} layer_weight_shape={layer.weight.shape} weights_dtype={weights_dtype} unsupported") from e
            if use_quantized_matmul:
                use_quantized_matmul = weights_dtype in quantized_matmul_dtypes and channel_size >= 32 and output_channel_size >= 32
                if use_quantized_matmul:
                    if dtype_dict[weights_dtype]["is_integer"]:
                        use_quantized_matmul = output_channel_size % 8 == 0 and channel_size % 8 == 0
                    else:
                        use_quantized_matmul = output_channel_size % 16 == 0 and channel_size % 16 == 0

        if group_size == 0:
            if is_linear_type:
                group_size = 2 ** (2 + dtype_dict[weights_dtype]["num_bits"])
            else:
                group_size = 2 ** (1 + dtype_dict[weights_dtype]["num_bits"])
        elif group_size != -1 and not is_linear_type:
            group_size = max(group_size // 2, 1)

        if not use_quantized_matmul and group_size > 0:
            if group_size >= channel_size:
                group_size = channel_size
                num_of_groups = 1
            else:
                num_of_groups = channel_size // group_size
                while num_of_groups * group_size != channel_size: # find something divisible
                    num_of_groups -= 1
                    if num_of_groups <= 1:
                        group_size = channel_size
                        num_of_groups = 1
                        break
                    group_size = channel_size // num_of_groups
            group_size = int(group_size)
            num_of_groups = int(num_of_groups)

            if num_of_groups > 1:
                result_shape = layer.weight.shape
                new_shape = list(result_shape)
                if is_conv_type:
                    # output_channel_size, channel_size, X, X
                    # output_channel_size, num_of_groups, group_size, X, X
                    new_shape[1] = group_size
                    new_shape.insert(1, num_of_groups)
                    reduction_axes = 2
                elif is_conv_transpose_type:
                    #channel_size, output_channel_size, X, X
                    #num_of_groups, group_size, output_channel_size, X, X
                    new_shape[0] = group_size
                    new_shape.insert(0, num_of_groups)
                    reduction_axes = 1
                elif is_linear_type:
                    # output_channel_size, channel_size
                    # output_channel_size, num_of_groups, group_size
                    last_dim_index = layer.weight.ndim
                    new_shape[last_dim_index - 1 : last_dim_index] = (num_of_groups, group_size)
                layer.weight.data = layer.weight.reshape(new_shape)

        layer.weight.requires_grad = False
        if return_device is None:
            return_device = layer.weight.device
        if quantization_device is not None:
            layer.weight.data = layer.weight.to(quantization_device, non_blocking=non_blocking)
        if layer.weight.dtype != torch.float32:
            layer.weight.data = layer.weight.to(dtype=torch.float32)

        layer.weight.data, scale, zero_point = quantize_weight(layer.weight, reduction_axes, weights_dtype)
        if not dequantize_fp32 and not (use_quantized_matmul and not dtype_dict[weights_dtype]["is_integer"] and not use_tensorwise_fp8_matmul):
            scale = scale.to(torch_dtype)
            if zero_point is not None:
                zero_point = zero_point.to(torch_dtype)

        if use_quantized_matmul:
            scale.transpose_(0,1)
            layer.weight.transpose_(0,1)
            if not dtype_dict[weights_dtype]["is_integer"]:
                stride = layer.weight.stride()
                if stride[0] > stride[1] and stride[1] == 1:
                    layer.weight.data = layer.weight.t().contiguous().t()
                if not use_tensorwise_fp8_matmul:
                    scale = scale.to(torch.float32)

        layer.sdnq_dequantizer = dequantizer_dict[weights_dtype](
            scale=scale,
            zero_point=zero_point,
            quantized_weight_shape=layer.weight.shape,
            result_dtype=torch_dtype,
            result_shape=result_shape,
            original_shape=original_shape,
            weights_dtype=weights_dtype,
            use_quantized_matmul=use_quantized_matmul,
        )
        layer.weight.data = layer.sdnq_dequantizer.pack_weight(layer.weight).to(return_device, non_blocking=non_blocking)
        layer.sdnq_dequantizer = layer.sdnq_dequantizer.to(return_device, non_blocking=non_blocking)

        layer.forward = get_forward_func(layer_class_name, use_quantized_matmul, dtype_dict[weights_dtype]["is_integer"], use_tensorwise_fp8_matmul)
        layer.forward = layer.forward.__get__(layer, layer.__class__)
    return layer


def apply_sdnq_to_module(model, weights_dtype="int8", torch_dtype=None, group_size=0, quant_conv=False, use_quantized_matmul=False, use_quantized_matmul_conv=False, dequantize_fp32=False, non_blocking=False, quantization_device=None, return_device=None, modules_to_not_convert: List[str] = [], modules_dtype_dict: Dict[str, List[str]] = {}, op=None): # pylint: disable=unused-argument
    has_children = list(model.children())
    if not has_children:
        return model
    for param_name, module in model.named_children():
        if param_name in modules_to_not_convert:
            continue
        if hasattr(module, "weight") and module.weight is not None:
            if len(modules_dtype_dict.keys()) > 0:
                for key, value in modules_dtype_dict.items():
                    if param_name in value:
                        key = key.lower()
                        if key in {"8bit", "8bits"}:
                            if dtype_dict[weights_dtype]["num_bits"] != 8:
                                weights_dtype = "int8"
                        elif key.startswith("minimum_"):
                            minimum_bits_str = key.removeprefix("minimum_").removesuffix("bits").removesuffix("bit")
                            minimum_bits = int(minimum_bits_str)
                            if dtype_dict[weights_dtype]["num_bits"] < minimum_bits:
                                weights_dtype = "int" + minimum_bits_str
                                if minimum_bits <= 4:
                                    weights_dtype = "u" + weights_dtype
                        else:
                            weights_dtype = key

            module = sdnq_quantize_layer(
                module,
                weights_dtype=weights_dtype,
                torch_dtype=torch_dtype,
                group_size=group_size,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                use_quantized_matmul_conv=use_quantized_matmul_conv,
                dequantize_fp32=dequantize_fp32,
                non_blocking=non_blocking,
                quantization_device=quantization_device,
                return_device=return_device,
                param_name=param_name,
            )
        module = apply_sdnq_to_module(
                module,
                weights_dtype=weights_dtype,
                torch_dtype=torch_dtype,
                group_size=group_size,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                use_quantized_matmul_conv=use_quantized_matmul_conv,
                dequantize_fp32=dequantize_fp32,
                non_blocking=non_blocking,
                quantization_device=quantization_device,
                return_device=return_device,
                modules_to_not_convert=modules_to_not_convert,
                modules_dtype_dict=modules_dtype_dict,
                op=op,
            )
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

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.modules_to_not_convert = []

    def check_if_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any], # pylint: disable=unused-argument
        **kwargs, # pylint: disable=unused-argument
    ):
        if param_name.endswith(".weight"):
            split_param_name = param_name.split(".")
            if (
                param_name not in self.modules_to_not_convert
                and not any(param in split_param_name for param in self.modules_to_not_convert)
                and not any("*" in param and re.match(param.replace(".*", "\\.*").replace("*", ".*"), param_name) for param in self.modules_to_not_convert)
            ):
                layer_class_name = get_module_from_name(model, param_name)[0].__class__.__name__
                if layer_class_name in allowed_types:
                    if layer_class_name in conv_types or layer_class_name in conv_transpose_types:
                        if self.quantization_config.quant_conv:
                            return True
                    else:
                        return True
        with devices.inference_context():
            param_value.data = param_value.clone() # safetensors is unable to release the cpu memory without this
        return False

    def check_quantized_param(self, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(*args, **kwargs)

    @devices.inference_context()
    def create_quantized_param( # pylint: disable=arguments-differ
        self,
        model,
        param_value: torch.FloatTensor,
        param_name: str,
        target_device: torch.device,
        state_dict: Dict[str, Any], # pylint: disable=unused-argument
        unexpected_keys: List[str], # pylint: disable=unused-argument
        **kwargs, # pylint: disable=unused-argument
    ):
        weights_dtype = self.quantization_config.weights_dtype
        if len(self.quantization_config.modules_dtype_dict.keys()) > 0:
            split_param_name = param_name.split(".")
            for key, value in self.quantization_config.modules_dtype_dict.items():
                if (
                    param_name in value
                    or any(param in split_param_name for param in value)
                    or any("*" in param and re.match(param.replace(".*", "\\.*").replace("*", ".*"), param_name) for param in value)
                ):
                    key = key.lower()
                    if key in {"8bit", "8bits"}:
                        if dtype_dict[weights_dtype]["num_bits"] != 8:
                            weights_dtype = "int8"
                    elif key.startswith("minimum_"):
                        minimum_bits_str = key.removeprefix("minimum_").removesuffix("bits").removesuffix("bit")
                        minimum_bits = int(minimum_bits_str)
                        if dtype_dict[weights_dtype]["num_bits"] < minimum_bits:
                            weights_dtype = "int" + minimum_bits_str
                            if minimum_bits <= 4:
                                weights_dtype = "u" + weights_dtype
                    else:
                        weights_dtype = key

        if self.quantization_config.return_device is not None:
            return_device = self.quantization_config.return_device
        else:
            return_device = target_device

        if self.quantization_config.quantization_device is not None:
            target_device = self.quantization_config.quantization_device

        if param_value.dtype == torch.float32 and devices.same_device(param_value.device, target_device):
            param_value = param_value.clone()
        else:
            param_value = param_value.to(target_device, non_blocking=self.quantization_config.non_blocking).to(dtype=torch.float32)

        layer, _ = get_module_from_name(model, param_name)
        layer.weight = torch.nn.Parameter(param_value, requires_grad=False)
        layer = sdnq_quantize_layer(
            layer,
            weights_dtype=weights_dtype,
            torch_dtype=self.torch_dtype,
            group_size=self.quantization_config.group_size,
            quant_conv=self.quantization_config.quant_conv,
            use_quantized_matmul=self.quantization_config.use_quantized_matmul,
            use_quantized_matmul_conv=self.quantization_config.use_quantized_matmul_conv,
            dequantize_fp32=self.quantization_config.dequantize_fp32,
            non_blocking=self.quantization_config.non_blocking,
            quantization_device=None,
            return_device=return_device,
            param_name=param_name,
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
        **kwargs, # pylint: disable=unused-argument
    ):
        if keep_in_fp32_modules is not None:
            self.modules_to_not_convert.extend(keep_in_fp32_modules)
        elif getattr(model, "_keep_in_fp32_modules", None) is not None:
            self.modules_to_not_convert.extend(model._keep_in_fp32_modules) # pylint: disable=protected-access
        if getattr(model, "_skip_layerwise_casting_patterns", None) is not None:
            self.modules_to_not_convert.extend(model._skip_layerwise_casting_patterns) # pylint: disable=protected-access
        self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)
        self.quantization_config.modules_to_not_convert = self.modules_to_not_convert
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs): # pylint: disable=unused-argument
        if shared.opts.diffusers_offload_mode != "none":
            model = model.to(devices.cpu)
        devices.torch_gc(force=True, reason='sdnq')
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

    def update_param_name(self, param_name: str) -> str:
        """
        needed for transformers compatibilty, no-op function
        """
        return param_name

    @property
    def is_trainable(self):
        return False

    @property
    def is_serializable(self):
        return True

    @property
    def is_compileable(self):
        return True


@dataclass
class SDNQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `sdnq`.

    Args:
        weights_dtype (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are:
            ("int8", "int7", "int6", "int5", "int4", "int3", "int2", "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1", "bool", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
        weights_dtype (`int`, *optional*, defaults to `0`):
            Used to decide how many elements of a tensor will share the same quantization group.
        quant_conv (`bool`, *optional*, defaults to `False`):
            Enabling this option will quantize the convolutional layers in UNet models too.
        use_quantized_matmul (`bool`, *optional*, defaults to `False`):
            Enabling this option will use quantized INT8 or FP8 MatMul instead of BF16 / FP16.
        use_quantized_matmul_conv (`bool`, *optional*, defaults to `False`):
            Same as use_quantized_matmul_conv but for the convolutional layers with UNets like SDXL.
        dequantize_fp32 (`bool`, *optional*, defaults to `False`):
            Enabling this option will use FP32 on the dequantization step.
        non_blocking (`bool`, *optional*, defaults to `False`):
            Enabling this option will use non blocking ops when moving layers between the quantization device and the return device.
        quantization_device (`torch.device`, *optional*, defaults to `None`):
            Used to set which device will be used for the quantization calculation on model load.
        return_device (`torch.device`, *optional*, defaults to `None`):
            Used to set which device will the quantized weights be sent back to.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
        modules_dtype_dict (`dict`, *optional*, default to `None`):
            The dict of dtypes and list of modules, useful for quantizing some modules with a different dtype.
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        weights_dtype: str = "int8",
        group_size: int = 0,
        quant_conv: bool = False,
        use_quantized_matmul: bool = False,
        use_quantized_matmul_conv: bool = False,
        dequantize_fp32: bool = False,
        non_blocking: bool = False,
        quantization_device: Optional[torch.device] = None,
        return_device: Optional[torch.device] = None,
        modules_to_not_convert: Optional[List[str]] = None,
        modules_dtype_dict: Optional[Dict[str, List[str]]] = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.weights_dtype = weights_dtype
        self.quant_method = QuantizationMethod.SDNQ
        self.group_size = group_size
        self.quant_conv = quant_conv
        self.use_quantized_matmul = use_quantized_matmul
        self.use_quantized_matmul_conv = use_quantized_matmul_conv
        self.dequantize_fp32 = dequantize_fp32
        self.non_blocking = non_blocking
        self.quantization_device = quantization_device
        self.return_device = return_device
        self.modules_to_not_convert = modules_to_not_convert
        self.modules_dtype_dict = modules_dtype_dict
        self.post_init()
        self.is_integer = dtype_dict[self.weights_dtype]["is_integer"]

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8", "int7", "int6", "int5", "int4", "int3", "int2", "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1", "bool", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"]
        if self.weights_dtype not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights_dtype}")

        if self.modules_to_not_convert is None:
            self.modules_to_not_convert = []
        elif not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        if self.modules_dtype_dict is None:
            self.modules_dtype_dict = {}
