# pylint: disable=redefined-builtin,no-member,protected-access

from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import re
import torch
from transformers.quantizers import HfQuantizer
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from diffusers.utils import get_module_from_name
from modules import devices, shared

from .common import dtype_dict, use_tensorwise_fp8_matmul, allowed_types, conv_types, conv_transpose_types, use_contiguous_mm
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
    else:
        quantized_weight.nan_to_num_()
    quantized_weight = quantized_weight.clamp_(dtype_dict[weights_dtype]["min"], dtype_dict[weights_dtype]["max"]).to(dtype_dict[weights_dtype]["torch_dtype"])
    return quantized_weight, scale, zero_point


def apply_svdquant(weight: torch.FloatTensor, rank: int = 32) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    reshape_weight = False
    if weight.ndim > 2: # convs
        reshape_weight = True
        weight_shape = weight.shape
        weight = weight.flatten(1,-1)
    U, S, svd_down = torch.svd_lowrank(weight, q=rank)
    svd_up = torch.mul(U, S.unsqueeze(0))
    svd_down = svd_down.t_()
    weight = weight.sub_(torch.mm(svd_up, svd_down))
    if reshape_weight:
        weight = weight.unflatten(-1, (*weight_shape[1:],)) # pylint: disable=possibly-used-before-assignment
    return weight, svd_up, svd_down


@devices.inference_context()
def sdnq_quantize_layer(layer, weights_dtype="int8", torch_dtype=None, group_size=0, svd_rank=32, use_svd=False, quant_conv=False, use_quantized_matmul=False, use_quantized_matmul_conv=False, dequantize_fp32=False, non_blocking=False, quantization_device=None, return_device=None, param_name=None): # pylint: disable=unused-argument
    layer_class_name = layer.__class__.__name__
    if layer_class_name in allowed_types:
        num_of_groups = 1
        is_conv_type = False
        is_conv_transpose_type = False
        is_linear_type = False
        result_shape = None
        original_shape = layer.weight.shape
        if torch_dtype is None:
            torch_dtype = layer.weight.dtype

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
                use_quantized_matmul = group_channel_size >= 32 and output_channel_size >= 32
                if use_quantized_matmul and not dtype_dict[weights_dtype]["is_integer"]:
                    use_quantized_matmul = output_channel_size % 16 == 0 and group_channel_size % 16 == 0
            if use_quantized_matmul and dtype_dict[weights_dtype]["num_bits"] == 8:
                result_shape = layer.weight.shape
                layer.weight.data = layer.weight.flatten(1,-1)
                reduction_axes = -1
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
                raise ValueError(f"SDNQ: param_name={param_name} layer_class_name={layer_class_name} layer_weight_shape={layer.weight.shape} weights_dtype={weights_dtype} unsupported") from e
            if use_quantized_matmul:
                use_quantized_matmul = channel_size >= 32 and output_channel_size >= 32
                if use_quantized_matmul:
                    if dtype_dict[weights_dtype]["is_integer"]:
                        use_quantized_matmul = output_channel_size % 8 == 0 and channel_size % 8 == 0
                    else:
                        use_quantized_matmul = output_channel_size % 16 == 0 and channel_size % 16 == 0

        layer.weight.requires_grad = False
        if return_device is None:
            return_device = layer.weight.device
        if quantization_device is not None:
            layer.weight.data = layer.weight.to(quantization_device, non_blocking=non_blocking)
        if layer.weight.dtype != torch.float32:
            layer.weight.data = layer.weight.to(dtype=torch.float32)

        if use_svd:
            layer.weight.data, svd_up, svd_down = apply_svdquant(layer.weight, rank=svd_rank)
            if use_quantized_matmul:
                svd_up = svd_up.t_()
                svd_down = svd_down.t_()
        else:
            svd_up, svd_down = None, None

        if group_size == 0:
            if use_quantized_matmul and dtype_dict[weights_dtype]["num_bits"] >= 6:
                group_size = -1
            elif is_linear_type:
                group_size = 2 ** ((2 if not use_svd else 3) + dtype_dict[weights_dtype]["num_bits"])
            else:
                group_size = 2 ** ((1 if not use_svd else 2) + dtype_dict[weights_dtype]["num_bits"])
        elif use_quantized_matmul and dtype_dict[weights_dtype]["num_bits"] == 8:
            group_size = -1 # override user value, re-quantizing 8bit into 8bit is pointless
        elif group_size != -1 and not is_linear_type:
            group_size = max(group_size // 2, 1)

        if group_size > 0:
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
                if result_shape is None:
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

        layer.weight.data, scale, zero_point = quantize_weight(layer.weight, reduction_axes, weights_dtype)
        if not dequantize_fp32 and not (use_quantized_matmul and not dtype_dict[weights_dtype]["is_integer"] and not use_tensorwise_fp8_matmul):
            scale = scale.to(dtype=torch_dtype)
            if zero_point is not None:
                zero_point = zero_point.to(dtype=torch_dtype)
            if svd_up is not None:
                svd_up = svd_up.to(dtype=torch_dtype)
                svd_down = svd_down.to(dtype=torch_dtype)

        re_quantize_for_matmul = (num_of_groups > 1 or zero_point is not None)
        if use_quantized_matmul and not re_quantize_for_matmul:
            scale.t_()
            layer.weight.t_()
            if use_contiguous_mm:
                layer.weight.data = layer.weight.contiguous()
            elif layer.weight.is_contiguous():
                layer.weight.data = layer.weight.t_().contiguous().t_()
            if not use_tensorwise_fp8_matmul and not dtype_dict[weights_dtype]["is_integer"]:
                scale = scale.to(dtype=torch.float32)

        scale = scale.to(return_device, non_blocking=non_blocking)
        layer.scale = torch.nn.Parameter(scale, requires_grad=False)
        if zero_point is not None:
            zero_point = zero_point.to(return_device, non_blocking=non_blocking)
            layer.zero_point = torch.nn.Parameter(zero_point, requires_grad=False)
        else:
            layer.zero_point = None
        if svd_up is not None:
            svd_up = svd_up.to(return_device, non_blocking=non_blocking)
            svd_down = svd_down.to(return_device, non_blocking=non_blocking)
            layer.svd_up = torch.nn.Parameter(svd_up, requires_grad=False)
            layer.svd_down = torch.nn.Parameter(svd_down, requires_grad=False)
        else:
            layer.svd_up, layer.svd_down = None, None

        layer.sdnq_dequantizer = dequantizer_dict[weights_dtype](
            quantized_weight_shape=layer.weight.shape,
            result_dtype=torch_dtype,
            result_shape=result_shape,
            original_shape=original_shape,
            weights_dtype=weights_dtype,
            use_quantized_matmul=use_quantized_matmul,
            re_quantize_for_matmul=re_quantize_for_matmul,
        )
        layer.weight.data = layer.sdnq_dequantizer.pack_weight(layer.weight).to(return_device, non_blocking=non_blocking)
        layer.sdnq_dequantizer = layer.sdnq_dequantizer.to(return_device, non_blocking=non_blocking)

        layer.forward = get_forward_func(layer_class_name, use_quantized_matmul, dtype_dict[weights_dtype]["is_integer"], use_tensorwise_fp8_matmul)
        layer.forward = layer.forward.__get__(layer, layer.__class__)
    return layer


def apply_sdnq_to_module(model, weights_dtype="int8", torch_dtype=None, group_size=0, svd_rank=32, use_svd=False, quant_conv=False, use_quantized_matmul=False, use_quantized_matmul_conv=False, dequantize_fp32=False, non_blocking=False, quantization_device=None, return_device=None, modules_to_not_convert: List[str] = None, modules_dtype_dict: Dict[str, List[str]] = None, full_param_name="", op=None): # pylint: disable=unused-argument
    has_children = list(model.children())
    if not has_children:
        return model
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    for param_name, module in model.named_children():
        if param_name == "sdnq_dequantizer":
            continue
        if full_param_name:
            param_name = full_param_name + "." + param_name
        if hasattr(module, "weight") and module.weight is not None:
            param_name = param_name + ".weight"
            split_param_name = param_name.split(".")
            if (
                param_name in modules_to_not_convert
                or any(param in split_param_name for param in modules_to_not_convert)
                or any("*" in param and re.match(param.replace(".*", "\\.*").replace("*", ".*"), param_name) for param in modules_to_not_convert)
            ):
                continue
            layer_class_name = module.__class__.__name__
            if layer_class_name in allowed_types:
                if (layer_class_name in conv_types or layer_class_name in conv_transpose_types) and not quant_conv:
                    continue
            else:
                continue
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
                svd_rank=svd_rank,
                use_svd=use_svd,
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
                svd_rank=svd_rank,
                use_svd=use_svd,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                use_quantized_matmul_conv=use_quantized_matmul_conv,
                dequantize_fp32=dequantize_fp32,
                non_blocking=non_blocking,
                quantization_device=quantization_device,
                return_device=return_device,
                modules_to_not_convert=modules_to_not_convert,
                modules_dtype_dict=modules_dtype_dict,
                full_param_name=param_name,
                op=op,
            )
    return model


def add_module_skip_keys(model, modules_to_not_convert: List[str] = None, modules_dtype_dict: Dict[str, List[str]]):
    if getattr(model, "_keep_in_fp32_modules", None) is not None:
        modules_to_not_convert.extend(model._keep_in_fp32_modules) # pylint: disable=protected-access
    if getattr(model, "_skip_layerwise_casting_patterns", None) is not None:
        modules_to_not_convert.extend(model._skip_layerwise_casting_patterns) # pylint: disable=protected-access
    if model.__class__.__name__ == "ChromaTransformer2DModel":
        modules_to_not_convert.append("distilled_guidance_layer")
    elif model.__class__.__name__ == "QwenImageTransformer2DModel":
        modules_to_not_convert.extend(["transformer_blocks.0.img_mod.1.weight", "time_text_embed", "img_in", "txt_in", "proj_out", "norm_out", "pos_embed"])
        if "minimum_6bit" not in modules_dtype_dict.keys():
            modules_dtype_dict["minimum_6bit"] = ["img_mod"]
        else:
            modules_dtype_dict["minimum_6bit"].append("img_mod")
    return model, modules_to_not_convert, modules_dtype_dict


def sdnq_post_load_quant(
    model,
    weights_dtype="int8",
    torch_dtype: torch.dtype = None,
    group_size: int = 0,
    svd_rank: int = 32,
    use_svd: bool = False,
    quant_conv: bool = False,
    use_quantized_matmul: bool = False,
    use_quantized_matmul_conv: bool = False,
    dequantize_fp32: bool = False,
    non_blocking: bool = False,
    add_skip_keys:bool = True,
    quantization_device: torch.device = None,
    return_device: torch.device = None,
    modules_to_not_convert: List[str] = None,
    modules_dtype_dict: Dict[str, List[str]] = None,
    op=None,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}

    if add_skip_keys:
        model, modules_to_not_convert, modules_dtype_dict = add_module_skip_keys(model, modules_to_not_convert, modules_dtype_dict)

    model.eval()
    model = apply_sdnq_to_module(
        model,
        weights_dtype=weights_dtype,
        torch_dtype=torch_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        use_svd=use_svd,
        quant_conv=quant_conv,
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=use_quantized_matmul_conv,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict.copy(),
        op=op,
    )

    model.quantization_config = SDNQConfig(
        weights_dtype=weights_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        use_svd=use_svd,
        quant_conv=quant_conv,
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=use_quantized_matmul_conv,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict.copy(),
    )

    if hasattr(model, "config"):
        model.config.quantization_config = model.quantization_config
    model.quantization_method = QuantizationMethod.SDNQ

    return model


class SDNQQuantizer(DiffusersQuantizer, HfQuantizer):
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
        *args, **kwargs, # pylint: disable=unused-argument
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
        if param_value is not None:
            with devices.inference_context():
                param_value.data = param_value.clone() # safetensors is unable to release the cpu memory without this
        return False

    def check_quantized_param(self, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(*args, **kwargs)

    def param_needs_quantization(self, model, param_name: str, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(model, None, param_name, *args, **kwargs)

    @devices.inference_context()
    def create_quantized_param( # pylint: disable=arguments-differ
        self,
        model,
        param_value: torch.FloatTensor,
        param_name: str,
        target_device: torch.device,
        *args, **kwargs, # pylint: disable=unused-argument
    ):
        weights_dtype = self.quantization_config.weights_dtype
        torch_dtype = param_value.dtype if self.torch_dtype is None else self.torch_dtype

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
            torch_dtype=torch_dtype,
            group_size=self.quantization_config.group_size,
            svd_rank=self.quantization_config.svd_rank,
            use_svd=self.quantization_config.use_svd,
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
        self.torch_dtype = torch_dtype
        return torch_dtype

    def _process_model_before_weight_loading( # pylint: disable=arguments-differ
        self,
        model,
        device_map, # pylint: disable=unused-argument
        keep_in_fp32_modules: List[str] = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.quantization_config.add_skip_keys:
            if keep_in_fp32_modules is not None:
                self.modules_to_not_convert.extend(keep_in_fp32_modules)
            model, self.modules_to_not_convert, self.quantization_config.modules_dtype_dict = add_module_skip_keys(
                model, self.modules_to_not_convert, self.quantization_config.modules_dtype_dict
            )
        self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)
        self.quantization_config.modules_to_not_convert = self.modules_to_not_convert
        if hasattr(model, "config"):
            model.config.quantization_config = self.quantization_config
        model.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model, **kwargs): # pylint: disable=unused-argument
        if shared.opts.diffusers_offload_mode != "none":
            model = model.to(devices.cpu)
        devices.torch_gc(force=True, reason="sdnq")
        return model

    def get_accelerator_warm_up_factor(self):
        return 32 // dtype_dict[self.quantization_config.weights_dtype]["num_bits"]

    def get_cuda_warm_up_factor(self):
        """
        needed for transformers compatibilty, returns self.get_accelerator_warm_up_factor
        """
        return self.get_accelerator_warm_up_factor()

    def update_tp_plan(self, config, *args, **kwargs): # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return config

    def update_ep_plan(self, config, *args, **kwargs): # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return config

    def update_unexpected_keys(self, model, unexpected_keys: List[str], *args, **kwargs) -> List[str]: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return unexpected_keys

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], *args, **kwargs) -> List[str]: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return missing_keys

    def update_state_dict_with_metadata(self, state_dict: dict, *args, **kwargs) -> dict: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return state_dict

    def update_expected_keys(self, model, expected_keys: List[str], *args, **kwargs) -> List[str]: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return expected_keys

    def update_param_name(self, param_name: str, *args, **kwargs) -> str: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return param_name

    def update_dtype(self, dtype: torch.dtype, *args, **kwargs) -> torch.dtype: # pylint: disable=unused-argument
        """
        needed for transformers compatibilty, no-op function
        """
        return dtype

    def is_serializable(self, *args, **kwargs) -> bool:  # pylint: disable=unused-argument, invalid-overridden-method
        """
        needed for transformers compatibilty, returns True
        """
        return True

    @property
    def is_trainable(self):
        return False

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
        group_size (`int`, *optional*, defaults to `0`):
            Used to decide how many elements of a tensor will share the same quantization group.
            group_size = 0 will automatically select a group size based on weights_dtype.
        svd_rank (`int`, *optional*, defaults to `32`):
            The rank size used for the SVDQuant algorithm.
        use_svd (`bool`, *optional*, defaults to `False`):
            Enabling this option will use SVDQuant algorithm.
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
        add_skip_keys (`bool`, *optional*, defaults to `True`):
            Disabling this option won't add model specific modules_to_not_convert and modules_dtype_dict keys.
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
        svd_rank: int = 32,
        use_svd: bool = False,
        quant_conv: bool = False,
        use_quantized_matmul: bool = False,
        use_quantized_matmul_conv: bool = False,
        dequantize_fp32: bool = False,
        non_blocking: bool = False,
        add_skip_keys: bool = True,
        quantization_device: Optional[torch.device] = None,
        return_device: Optional[torch.device] = None,
        modules_to_not_convert: Optional[List[str]] = None,
        modules_dtype_dict: Optional[Dict[str, List[str]]] = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.weights_dtype = weights_dtype
        self.quant_method = QuantizationMethod.SDNQ
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.use_svd = use_svd
        self.quant_conv = quant_conv
        self.use_quantized_matmul = use_quantized_matmul
        self.use_quantized_matmul_conv = use_quantized_matmul_conv
        self.dequantize_fp32 = dequantize_fp32
        self.non_blocking = non_blocking
        self.add_skip_keys = add_skip_keys
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

    def to_dict(self):
        dct = self.__dict__.copy() # make serializable
        dct["quantization_device"] = str(dct["quantization_device"]) if dct["quantization_device"] is not None else None
        dct["return_device"] = str(dct["return_device"]) if dct["return_device"] is not None else None
        return dct


import diffusers.quantizers.auto # noqa: E402,RUF100 # pylint: disable=wrong-import-order
diffusers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq"] = SDNQQuantizer
diffusers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq"] = SDNQConfig

import transformers.quantizers.auto # noqa: E402,RUF100 # pylint: disable=wrong-import-order
transformers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq"] = SDNQQuantizer
transformers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq"] = SDNQConfig
