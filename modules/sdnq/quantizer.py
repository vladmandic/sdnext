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
from accelerate import init_empty_weights

from modules import devices, shared
from .common import sdnq_version, dtype_dict, common_skip_keys, module_skip_keys_dict, accepted_weight_dtypes, accepted_matmul_dtypes, allowed_types, linear_types, conv_types, conv_transpose_types, compile_func, use_tensorwise_fp8_matmul, use_contiguous_mm, check_torch_compile
from .dequantizer import SDNQDequantizer, dequantize_sdnq_model
from .packed_int import pack_int_symetric, pack_int_asymetric
from .forward import get_forward_func


class QuantizationMethod(str, Enum):
    SDNQ = "sdnq"
    SDNQ_TRAINING = "sdnq_training"


@devices.inference_context()
def get_scale_asymmetric(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point = torch.amin(weight, dim=reduction_axes, keepdims=True)
    scale = torch.amax(weight, dim=reduction_axes, keepdims=True).sub_(zero_point).div_(dtype_dict[weights_dtype]["max"] - dtype_dict[weights_dtype]["min"])
    if dtype_dict[weights_dtype]["min"] != 0:
        zero_point.sub_(torch.mul(scale, dtype_dict[weights_dtype]["min"]))
    return scale, zero_point


@devices.inference_context()
def get_scale_symmetric(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> torch.FloatTensor:
    return torch.amax(weight.abs(), dim=reduction_axes, keepdims=True).div_(dtype_dict[weights_dtype]["max"])


@devices.inference_context()
def quantize_weight(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str, use_stochastic_rounding: bool = False) -> Tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    weight = weight.to(dtype=torch.float32)

    if dtype_dict[weights_dtype]["is_unsigned"]:
        scale, zero_point = get_scale_asymmetric(weight, reduction_axes, weights_dtype)
        quantized_weight = torch.sub(weight, zero_point)
        scale_inplace = True
    else:
        scale = get_scale_symmetric(weight, reduction_axes, weights_dtype)
        quantized_weight = weight
        scale_inplace = False
        zero_point = None

    is_integer = dtype_dict[weights_dtype]["is_integer"]
    if use_stochastic_rounding and is_integer: # this case can be fused with addcdiv_
        quantized_weight = torch.normal(0, 0.1, weight.shape, device=weight.device, dtype=weight.dtype).addcdiv_(quantized_weight, scale)
    elif scale_inplace:
        quantized_weight.div_(scale)
    else:
        quantized_weight = torch.div(quantized_weight, scale)

    if is_integer:
        quantized_weight.round_()
    else:
        if use_stochastic_rounding:
            mantissa_difference = 1 << (23 - dtype_dict[weights_dtype]["mantissa"])
            quantized_weight = quantized_weight.view(dtype=torch.int32)
            quantized_weight = torch.randint_like(quantized_weight, low=0, high=mantissa_difference).add_(quantized_weight).bitwise_and_(-mantissa_difference).view(dtype=torch.float32)
        quantized_weight.nan_to_num_()
    quantized_weight = quantized_weight.clamp_(dtype_dict[weights_dtype]["min"], dtype_dict[weights_dtype]["max"]).to(dtype_dict[weights_dtype]["torch_dtype"])
    return quantized_weight, scale, zero_point


@devices.inference_context()
def apply_svdquant(weight: torch.FloatTensor, rank: int = 32, niter: int = 8) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    reshape_weight = False
    if weight.ndim > 2: # convs
        reshape_weight = True
        weight_shape = weight.shape
        weight = weight.flatten(1,-1)
    weight = weight.to(dtype=torch.float32)
    U, S, svd_down = torch.svd_lowrank(weight, q=rank, niter=niter)
    svd_up = torch.mul(U, S.unsqueeze(0))
    svd_down = svd_down.t_()
    weight = weight.sub(torch.mm(svd_up, svd_down))
    if reshape_weight:
        weight = weight.unflatten(-1, (*weight_shape[1:],)) # pylint: disable=possibly-used-before-assignment
    return weight, svd_up, svd_down


@devices.inference_context()
def prepare_weight_for_matmul(weight: torch.Tensor) -> torch.Tensor:
    if use_contiguous_mm:
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t_().contiguous().t_()
    return weight


@devices.inference_context()
def prepare_svd_for_matmul(svd_up: torch.FloatTensor, svd_down: torch.FloatTensor, use_quantized_matmul: bool) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    if svd_up is not None:
        if use_quantized_matmul:
            svd_up = prepare_weight_for_matmul(svd_up)
        else:
            svd_up = svd_up.contiguous()
    if svd_down is not None:
        svd_down = prepare_weight_for_matmul(svd_down)
    return svd_up, svd_down


def check_param_name_in(param_name: str, param_list: List[str]) -> bool:
    split_param_name = param_name.split(".")
    for param in param_list:
        if param.startswith("."):
            if param_name.startswith(param[1:]):
                return True
            else:
                continue
        if (
            param_name == param
            or param in split_param_name
            or ("*" in param and re.match(param.replace(".*", "\\.*").replace("*", ".*"), param_name))
        ):
            return True
    return False


def get_quant_args_from_config(quantization_config: Union["SDNQConfig", dict]) -> dict:
    if isinstance(quantization_config, SDNQConfig):
        quantization_config_dict = quantization_config.to_dict()
    else:
        quantization_config_dict = quantization_config.copy()
    quantization_config_dict.pop("is_integer", None)
    quantization_config_dict.pop("quant_method", None)
    quantization_config_dict.pop("quantization_device", None)
    quantization_config_dict.pop("return_device", None)
    quantization_config_dict.pop("non_blocking", None)
    quantization_config_dict.pop("add_skip_keys", None)
    quantization_config_dict.pop("use_static_quantization", None)
    quantization_config_dict.pop("use_stochastic_rounding", None)
    quantization_config_dict.pop("use_grad_ckpt", None)
    quantization_config_dict.pop("is_training", None)
    quantization_config_dict.pop("sdnq_version", None)
    return quantization_config_dict


def get_minimum_dtype(weights_dtype: str, param_name: str, modules_dtype_dict: Dict[str, List[str]]):
    if len(modules_dtype_dict.keys()) > 0:
        for key, value in modules_dtype_dict.items():
            if check_param_name_in(param_name, value):
                key = key.lower()
                if key in {"8bit", "8bits"}:
                    if dtype_dict[weights_dtype]["num_bits"] != 8:
                        return "int8"
                elif key.startswith("minimum_"):
                    minimum_bits_str = key.removeprefix("minimum_").removesuffix("bits").removesuffix("bit")
                    if minimum_bits_str.startswith("uint"):
                        is_unsigned = True
                        minimum_bits_str = minimum_bits_str.removeprefix("uint")
                    else:
                        is_unsigned = False
                        minimum_bits_str = minimum_bits_str.removeprefix("int")
                    minimum_bits = int(minimum_bits_str)
                    if dtype_dict[weights_dtype]["num_bits"] < minimum_bits:
                        if is_unsigned or minimum_bits <= 4:
                            return "uint" + minimum_bits_str
                        else:
                            return "int" + minimum_bits_str
                else:
                    return key
    return weights_dtype


def add_module_skip_keys(model, modules_to_not_convert: List[str] = None, modules_dtype_dict: Dict[str, List[str]] = None):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    if getattr(model, "_keep_in_fp32_modules", None) is not None:
        modules_to_not_convert.extend(model._keep_in_fp32_modules) # pylint: disable=protected-access
    if getattr(model, "_tied_weights_keys", None) is not None:
        if isinstance(model._tied_weights_keys, dict): # pylint: disable=protected-access
            modules_to_not_convert.extend(model._tied_weights_keys.keys()) # pylint: disable=protected-access
            modules_to_not_convert.extend(model._tied_weights_keys.values()) # pylint: disable=protected-access
        else:
            modules_to_not_convert.extend(model._tied_weights_keys) # pylint: disable=protected-access

    skip_key_list = module_skip_keys_dict.get(model.__class__.__name__, None)
    if skip_key_list is not None:
        modules_to_not_convert.extend(skip_key_list[0])
        for key, value in skip_key_list[1].items():
            if key in modules_dtype_dict.keys():
                modules_dtype_dict[key].extend(value)
            else:
                modules_dtype_dict[key] = value
    else:
        modules_to_not_convert.extend(common_skip_keys)
        if getattr(model, "_skip_layerwise_casting_patterns", None) is not None:
            modules_to_not_convert.extend(model._skip_layerwise_casting_patterns) # pylint: disable=protected-access

    # dedupe
    modules_to_not_convert = list(set(modules_to_not_convert))
    for key, value in modules_dtype_dict.items():
        modules_dtype_dict[key] = list(set(value))

    return model, modules_to_not_convert, modules_dtype_dict


@devices.inference_context()
def sdnq_quantize_layer_weight(weight, layer_class_name=None, weights_dtype="int8", quantized_matmul_dtype=None, torch_dtype=None, group_size=0, svd_rank=32, svd_steps=8, use_svd=False, use_quantized_matmul=False, use_stochastic_rounding=False, dequantize_fp32=False, param_name=None): # pylint: disable=unused-argument
    num_of_groups = 1
    is_conv_type = False
    is_conv_transpose_type = False
    is_linear_type = False
    result_shape = None
    original_shape = weight.shape
    original_stride = weight.stride()

    if torch_dtype is None:
        torch_dtype = weight.dtype
    if quantized_matmul_dtype is None:
        if dtype_dict[weights_dtype]["is_integer"]:
            quantized_matmul_dtype = "int8"
        elif dtype_dict[weights_dtype]["num_bits"] == 8:
            quantized_matmul_dtype = "float8_e4m3fn"
        else:
            quantized_matmul_dtype = "float16"

    re_quantize_for_matmul = bool(
        dtype_dict[weights_dtype]["is_unsigned"]
        or dtype_dict[weights_dtype]["is_integer"] != dtype_dict[quantized_matmul_dtype]["is_integer"]
        or dtype_dict[weights_dtype]["num_bits"] > dtype_dict[quantized_matmul_dtype]["num_bits"]
    )

    if layer_class_name in conv_types:
        if dtype_dict[weights_dtype]["num_bits"] < 4:
            weights_dtype = "uint4"
        is_conv_type = True
        reduction_axes = 1
        output_channel_size, channel_size = weight.shape[:2]
        if use_quantized_matmul:
            use_quantized_matmul = channel_size >= 32 and output_channel_size >= 32
            use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
        if use_quantized_matmul and not re_quantize_for_matmul and not dtype_dict[weights_dtype]["is_packed"]:
            result_shape = weight.shape
            weight = weight.flatten(1,-1)
            reduction_axes = -1
    elif layer_class_name in conv_transpose_types:
        if dtype_dict[weights_dtype]["num_bits"] < 4:
            weights_dtype = "uint4"
        is_conv_transpose_type = True
        reduction_axes = 0
        channel_size, output_channel_size = weight.shape[:2]
        use_quantized_matmul = False
    elif layer_class_name in linear_types:
        is_linear_type = True
        reduction_axes = -1
        try:
            output_channel_size, channel_size = weight.shape
        except Exception as e:
            raise ValueError(f"SDNQ: param_name={param_name} layer_class_name={layer_class_name} weight_shape={weight.shape} weights_dtype={weights_dtype} quantized_matmul_dtype={quantized_matmul_dtype} unsupported") from e
        if use_quantized_matmul:
            use_quantized_matmul = channel_size >= 32 and output_channel_size >= 32
            use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
    else:
        if weight.ndim > 1:
            output_channel_size, channel_size = weight.shape[-2:]
        else:
            output_channel_size, channel_size = 1, weight.shape[-1]
        reduction_axes = -1
        use_quantized_matmul = False

    if use_svd:
        try:
            weight, svd_up, svd_down = apply_svdquant(weight, rank=svd_rank, niter=svd_steps)
            if use_quantized_matmul:
                svd_up = svd_up.t_()
                svd_down = svd_down.t_()
            svd_up, svd_down = prepare_svd_for_matmul(svd_up, svd_down, use_quantized_matmul)
        except Exception:
            svd_up, svd_down = None, None
    else:
        svd_up, svd_down = None, None

    if group_size == 0:
        if use_quantized_matmul and not re_quantize_for_matmul and dtype_dict[weights_dtype]["num_bits"] >= 6:
            group_size = -1
        elif is_linear_type:
            group_size = 2 ** ((2 if svd_up is None else 3) + dtype_dict[weights_dtype]["num_bits"])
        else:
            group_size = 2 ** ((1 if svd_up is None else 2) + dtype_dict[weights_dtype]["num_bits"])

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
                result_shape = weight.shape
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
            else:
                # output_channel_size, channel_size
                # output_channel_size, num_of_groups, group_size
                last_dim_index = weight.ndim
                new_shape[last_dim_index - 1 : last_dim_index] = (num_of_groups, group_size)
            weight = weight.reshape(new_shape)
        else:
            group_size = -1

    weight, scale, zero_point = quantize_weight(weight, reduction_axes, weights_dtype)
    if (
        not dequantize_fp32
        and dtype_dict[weights_dtype]["num_bits"] <= 8
        and not (
            use_quantized_matmul
            and not dtype_dict[quantized_matmul_dtype]["is_integer"]
            and (not use_tensorwise_fp8_matmul or dtype_dict[quantized_matmul_dtype]["num_bits"] == 16)
        )
    ):
        scale = scale.to(dtype=torch_dtype)
        if zero_point is not None:
            zero_point = zero_point.to(dtype=torch_dtype)
        if svd_up is not None:
            svd_up = svd_up.to(dtype=torch_dtype)
            svd_down = svd_down.to(dtype=torch_dtype)

    re_quantize_for_matmul = re_quantize_for_matmul or num_of_groups > 1
    if use_quantized_matmul and not re_quantize_for_matmul:
        scale.t_()
        weight.t_()
        weight = prepare_weight_for_matmul(weight)
        if not use_tensorwise_fp8_matmul and not dtype_dict[weights_dtype]["is_integer"]:
            scale = scale.to(dtype=torch.float32)

    sdnq_dequantizer = SDNQDequantizer(
        result_dtype=torch_dtype,
        result_shape=result_shape,
        original_shape=original_shape,
        original_stride=original_stride,
        quantized_weight_shape=weight.shape,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_quantized_matmul=use_quantized_matmul,
        re_quantize_for_matmul=re_quantize_for_matmul,
        use_stochastic_rounding=use_stochastic_rounding,
        layer_class_name=layer_class_name,
    )

    if dtype_dict[weights_dtype]["is_packed"]:
        if dtype_dict[weights_dtype]["is_unsigned"]:
            weight = pack_int_asymetric(weight, weights_dtype)
        else:
            weight = pack_int_symetric(weight, weights_dtype)
    else:
        weight = weight.to(dtype=dtype_dict[weights_dtype]["torch_dtype"])

    return weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer


@devices.inference_context()
def sdnq_quantize_layer(layer, weights_dtype="int8", quantized_matmul_dtype=None, torch_dtype=None, group_size=0, svd_rank=32, svd_steps=8, use_svd=False, quant_conv=False, use_quantized_matmul=False, use_quantized_matmul_conv=False, use_stochastic_rounding=False, dequantize_fp32=False, non_blocking=False, quantization_device=None, return_device=None, param_name=None): # pylint: disable=unused-argument
    layer_class_name = layer.__class__.__name__
    if layer_class_name in conv_transpose_types or layer_class_name in conv_types:
        if not quant_conv:
            return layer
        use_quantized_matmul = use_quantized_matmul_conv

    layer.weight.requires_grad_(False)
    if return_device is None:
        return_device = layer.weight.device
    if quantization_device is not None:
        layer.weight.data = layer.weight.to(quantization_device, non_blocking=non_blocking)

    (
        layer.weight.data,
        layer.scale, layer.zero_point,
        layer.svd_up, layer.svd_down,
        layer.sdnq_dequantizer,
    ) = sdnq_quantize_layer_weight(
        layer.weight,
        layer_class_name=layer_class_name,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        torch_dtype=torch_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        use_quantized_matmul=use_quantized_matmul,
        use_stochastic_rounding=use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        param_name=param_name,
    )

    layer.weight = torch.nn.Parameter(layer.weight.to(return_device, non_blocking=non_blocking), requires_grad=False)
    layer.scale = torch.nn.Parameter(layer.scale.to(return_device, non_blocking=non_blocking), requires_grad=False)
    if layer.zero_point is not None:
        layer.zero_point = torch.nn.Parameter(layer.zero_point.to(return_device, non_blocking=non_blocking), requires_grad=False)
    if layer.svd_up is not None:
        layer.svd_up = torch.nn.Parameter(layer.svd_up.to(return_device, non_blocking=non_blocking), requires_grad=False)
        layer.svd_down = torch.nn.Parameter(layer.svd_down.to(return_device, non_blocking=non_blocking), requires_grad=False)

    layer = layer.to(return_device, non_blocking=non_blocking)
    layer.forward = get_forward_func(layer_class_name, layer.sdnq_dequantizer.quantized_matmul_dtype, layer.sdnq_dequantizer.use_quantized_matmul)
    layer.forward = layer.forward.__get__(layer, layer.__class__)
    return layer


@devices.inference_context()
def apply_sdnq_to_module(model, weights_dtype="int8", quantized_matmul_dtype=None, torch_dtype=None, group_size=0, svd_rank=32, svd_steps=8, use_svd=False, quant_conv=False, use_quantized_matmul=False, use_quantized_matmul_conv=False, use_stochastic_rounding=False, dequantize_fp32=False, non_blocking=False, quantization_device=None, return_device=None, modules_to_not_convert: List[str] = None, modules_dtype_dict: Dict[str, List[str]] = None, full_param_name=""): # pylint: disable=unused-argument
    has_children = list(model.children())
    if not has_children:
        return model
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    for module_name, module in model.named_children():
        if full_param_name:
            param_name = full_param_name + "." + module_name
        else:
            param_name = module_name
        if hasattr(module, "weight") and module.weight is not None:
            param_name = param_name + ".weight"
            if check_param_name_in(param_name, modules_to_not_convert):
                continue
            layer_class_name = module.__class__.__name__
            if layer_class_name in allowed_types and module.weight.dtype in {torch.float32, torch.float16, torch.bfloat16}:
                if (layer_class_name in conv_types or layer_class_name in conv_transpose_types) and not quant_conv:
                    continue
                setattr(model, module_name, sdnq_quantize_layer(
                    module,
                    weights_dtype=get_minimum_dtype(weights_dtype, param_name, modules_dtype_dict),
                    quantized_matmul_dtype=quantized_matmul_dtype,
                    torch_dtype=torch_dtype,
                    group_size=group_size,
                    svd_rank=svd_rank,
                    svd_steps=svd_steps,
                    use_svd=use_svd,
                    quant_conv=quant_conv,
                    use_quantized_matmul=use_quantized_matmul,
                    use_quantized_matmul_conv=use_quantized_matmul_conv,
                    use_stochastic_rounding=use_stochastic_rounding,
                    dequantize_fp32=dequantize_fp32,
                    non_blocking=non_blocking,
                    quantization_device=quantization_device,
                    return_device=return_device,
                    param_name=param_name,
                ))
        setattr(model, module_name, apply_sdnq_to_module(
                module,
                weights_dtype=weights_dtype,
                quantized_matmul_dtype=quantized_matmul_dtype,
                torch_dtype=torch_dtype,
                group_size=group_size,
                svd_rank=svd_rank,
                svd_steps=svd_steps,
                use_svd=use_svd,
                quant_conv=quant_conv,
                use_quantized_matmul=use_quantized_matmul,
                use_quantized_matmul_conv=use_quantized_matmul_conv,
                use_stochastic_rounding=use_stochastic_rounding,
                dequantize_fp32=dequantize_fp32,
                non_blocking=non_blocking,
                quantization_device=quantization_device,
                return_device=return_device,
                modules_to_not_convert=modules_to_not_convert,
                modules_dtype_dict=modules_dtype_dict,
                full_param_name=param_name,
            ))
    return model


@devices.inference_context()
def sdnq_post_load_quant(
    model: torch.nn.Module,
    weights_dtype: str = "int8",
    quantized_matmul_dtype: str = None,
    torch_dtype: torch.dtype = None,
    group_size: int = 0,
    svd_rank: int = 32,
    svd_steps: int = 8,
    use_svd: bool = False,
    quant_conv: bool = False,
    use_quantized_matmul: bool = False,
    use_quantized_matmul_conv: bool = False,
    use_stochastic_rounding: bool = False,
    dequantize_fp32: bool = False,
    non_blocking: bool = False,
    add_skip_keys:bool = True,
    quantization_device: Optional[torch.device] = None,
    return_device: Optional[torch.device] = None,
    modules_to_not_convert: List[str] = None,
    modules_dtype_dict: Dict[str, List[str]] = None,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}

    modules_to_not_convert = modules_to_not_convert.copy()
    modules_dtype_dict = modules_dtype_dict.copy()
    if add_skip_keys:
        model, modules_to_not_convert, modules_dtype_dict = add_module_skip_keys(model, modules_to_not_convert, modules_dtype_dict)

    quantization_config = SDNQConfig(
        weights_dtype=weights_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        quant_conv=quant_conv,
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=use_quantized_matmul_conv,
        use_stochastic_rounding=use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        add_skip_keys=add_skip_keys,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
    )

    model.eval()
    model = apply_sdnq_to_module(
        model,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        torch_dtype=torch_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        quant_conv=quant_conv,
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=use_quantized_matmul_conv,
        use_stochastic_rounding=use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
    )

    model.quantization_config = quantization_config
    if hasattr(model, "config"):
        try:
            model.config.quantization_config = model.quantization_config
        except Exception:
            pass
        try:
            model.config["quantization_config"] = model.quantization_config.to_dict()
        except Exception:
            pass
    model.quantization_method = QuantizationMethod.SDNQ

    return model


class SDNQQuantize():
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        model: torch.nn.Module = None,
        full_layer_name: str = None,
        missing_keys: list[str] = None, # pylint: disable=unused-argument
        **kwargs, # pylint: disable=unused-argument
    ) -> dict[str, torch.FloatTensor]:
        _module_name, value = tuple(input_dict.items())[0]
        value = value[0]
        self.hf_quantizer.create_quantized_param(model, value, full_layer_name, value.device)
        param, name = get_module_from_name(model, full_layer_name)
        param = getattr(param, name)
        return {full_layer_name: param}

    @property
    def reverse_op(self):
        raise NotImplementedError


class SDNQQuantizer(DiffusersQuantizer, HfQuantizer):
    r"""
    Diffusers and Transformers Quantizer for SDNQ
    """

    requires_parameters_quantization = True
    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = None
    torch_dtype = None

    def check_if_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        *args, **kwargs, # pylint: disable=unused-argument,keyword-arg-before-vararg
    ):
        if self.pre_quantized:
            layer, _tensor_name = get_module_from_name(model, param_name)
            if hasattr(layer, "sdnq_dequantizer"):
                return True
        elif param_name.endswith(".weight"):
            if not check_param_name_in(param_name, self.quantization_config.modules_to_not_convert):
                layer_class_name = get_module_from_name(model, param_name)[0].__class__.__name__
                if layer_class_name in allowed_types:
                    if layer_class_name in conv_types or layer_class_name in conv_transpose_types:
                        if self.quantization_config.quant_conv:
                            return True
                    else:
                        return True
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
        if self.pre_quantized:
            layer, tensor_name = get_module_from_name(model, param_name)
            if param_value is not None:
                return_dtype = param_value.dtype if tensor_name == "weight" else torch.float32 if self.quantization_config.dequantize_fp32 else kwargs.get("dtype", param_value.dtype if self.torch_dtype is None else self.torch_dtype)
                if param_value.dtype == return_dtype and devices.same_device(param_value.device, target_device):
                    param_value = param_value.clone()
                else:
                    param_value = param_value.to(target_device, dtype=return_dtype)

                if tensor_name == "weight" and layer.sdnq_dequantizer.use_quantized_matmul and not layer.sdnq_dequantizer.re_quantize_for_matmul:
                    param_value = prepare_weight_for_matmul(param_value)
                elif tensor_name == "svd_up":
                    param_value, _ = prepare_svd_for_matmul(param_value, None, layer.sdnq_dequantizer.use_quantized_matmul)
                elif tensor_name == "svd_down":
                    _, param_value = prepare_svd_for_matmul(None, param_value, layer.sdnq_dequantizer.use_quantized_matmul)

                param_value = torch.nn.Parameter(param_value, requires_grad=False)
                param_value._is_hf_initialized = True # pylint: disable=protected-access
            setattr(layer, tensor_name, param_value)
            return

        torch_dtype = kwargs.get("dtype", param_value.dtype if self.torch_dtype is None else self.torch_dtype)
        weights_dtype = get_minimum_dtype(self.quantization_config.weights_dtype, param_name, self.quantization_config.modules_dtype_dict)

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
            quantized_matmul_dtype=self.quantization_config.quantized_matmul_dtype,
            torch_dtype=torch_dtype,
            group_size=self.quantization_config.group_size,
            svd_rank=self.quantization_config.svd_rank,
            svd_steps=self.quantization_config.svd_steps,
            use_svd=self.quantization_config.use_svd,
            quant_conv=self.quantization_config.quant_conv,
            use_quantized_matmul=self.quantization_config.use_quantized_matmul,
            use_quantized_matmul_conv=self.quantization_config.use_quantized_matmul_conv,
            use_stochastic_rounding=self.quantization_config.use_stochastic_rounding,
            dequantize_fp32=self.quantization_config.dequantize_fp32,
            non_blocking=self.quantization_config.non_blocking,
            quantization_device=None,
            return_device=return_device,
            param_name=param_name,
        )

        layer.weight._is_hf_initialized = True # pylint: disable=protected-access
        layer.scale._is_hf_initialized = True # pylint: disable=protected-access
        if layer.zero_point is not None:
            layer.zero_point._is_hf_initialized = True # pylint: disable=protected-access
        if layer.svd_up is not None:
            layer.svd_up._is_hf_initialized = True # pylint: disable=protected-access
            layer.svd_down._is_hf_initialized = True # pylint: disable=protected-access

    def get_quantize_ops(self):
        return SDNQQuantize(self)

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.80 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: torch.dtype) -> torch.dtype: # pylint: disable=unused-argument,arguments-renamed
        return dtype_dict[self.quantization_config.weights_dtype]["target_dtype"]

    def update_torch_dtype(self, torch_dtype: torch.dtype = None) -> torch.dtype:
        self.torch_dtype = torch_dtype
        return torch_dtype

    def update_dtype(self, dtype: torch.dtype = None) -> torch.dtype:
        """
        needed for transformers compatibilty, returns self.update_torch_dtype
        """
        return self.update_torch_dtype(dtype)

    def _process_model_before_weight_loading( # pylint: disable=arguments-differ
        self,
        model,
        device_map, # pylint: disable=unused-argument
        keep_in_fp32_modules: List[str] = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        if self.pre_quantized:
            self.quantization_config.quantization_device = None
            self.quantization_config.return_device = None
            self.quantization_config.non_blocking = False
            self.quantization_config.add_skip_keys = False

            with init_empty_weights():
                model = sdnq_post_load_quant(model, torch_dtype=self.torch_dtype, add_skip_keys=False, **get_quant_args_from_config(self.quantization_config))

        if self.quantization_config.add_skip_keys:
            if keep_in_fp32_modules is not None:
                self.quantization_config.modules_to_not_convert.extend(keep_in_fp32_modules)
            if hasattr(self, "get_modules_to_not_convert") and hasattr(model, "tie_weights"):
                self.quantization_config.modules_to_not_convert.extend(self.get_modules_to_not_convert(model, add_default_skips=True))
            model, self.quantization_config.modules_to_not_convert, self.quantization_config.modules_dtype_dict = add_module_skip_keys(
                model, self.quantization_config.modules_to_not_convert, self.quantization_config.modules_dtype_dict
            )
        if hasattr(model, "config"):
            try:
                model.config.quantization_config = self.quantization_config
            except Exception:
                pass
            try:
                model.config["quantization_config"] = self.quantization_config.to_dict()
            except Exception:
                pass
        model.quantization_config = self.quantization_config
        model.quantization_method = QuantizationMethod.SDNQ

    def _process_model_after_weight_loading(self, model, **kwargs): # pylint: disable=unused-argument
        if self.pre_quantized:
            from .loader import post_process_model
            model = post_process_model(model)
        if self.quantization_config.is_training:
            from .training import convert_sdnq_model_to_training
            model = convert_sdnq_model_to_training(
                model,
                dtype=self.torch_dtype,
                quantized_matmul_dtype=self.quantization_config.quantized_matmul_dtype,
                use_grad_ckpt=self.quantization_config.use_grad_ckpt,
                use_quantized_matmul=self.quantization_config.use_quantized_matmul,
                use_stochastic_rounding=self.quantization_config.use_stochastic_rounding,
                dequantize_fp32=self.quantization_config.dequantize_fp32,
            )
        if shared.opts.diffusers_offload_mode != "none":
            try:
                model = model.to(device=devices.cpu)
            except Exception:
                model = model.to_empty(device=devices.cpu)
        devices.torch_gc(force=True, reason="sdnq")
        return model

    def get_accelerator_warm_up_factor(self):
        return 32 // dtype_dict[self.quantization_config.weights_dtype]["num_bits"]

    def get_cuda_warm_up_factor(self):
        """
        needed for transformers compatibilty, returns self.get_accelerator_warm_up_factor
        """
        return self.get_accelerator_warm_up_factor()

    def _dequantize(self, model):
        return dequantize_sdnq_model(model)

    def is_serializable(self, *args, **kwargs) -> bool:  # pylint: disable=unused-argument, invalid-overridden-method
        return not self.quantization_config.is_training

    @property
    def is_trainable(self):
        return self.quantization_config.is_training

    @property
    def is_qat_trainable(self) -> bool:
        return self.is_trainable()

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
            ("int16", "int8", "int7", "int6", "int5", "int4", "int3", "int2", "uint16", "uint8", "uint7", "uint6", "uint5", "uint4", "uint3", "uint2", "uint1", "bool", "float16", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
        quantized_matmul_dtype (`str`, *optional*, defaults to `None`):
            The target dtype for quantized matmul.
            `None` will use "int8" with integer weight dtypes and "float8_e4m3fn" or "float16" with float weight dtypes.
            Supported values are: ("int8", "float8_e4m3fn", "float16")
        group_size (`int`, *optional*, defaults to `0`):
            Used to decide how many elements of a tensor will share the same quantization group.
            group_size = 0 will automatically select a group size based on weights_dtype.
        svd_rank (`int`, *optional*, defaults to `32`):
            The rank size used for the SVDQuant algorithm.
        svd_steps (`int`, *optional*, defaults to `8`):
            The number of iterations to use in svd lowrank estimation.
        use_svd (`bool`, *optional*, defaults to `False`):
            Enabling this option will use SVDQuant algorithm on top of SDNQ quantization.
        quant_conv (`bool`, *optional*, defaults to `False`):
            Enabling this option will quantize the convolutional layers in UNet models too.
        use_quantized_matmul (`bool`, *optional*, defaults to `False`):
            Enabling this option will use quantized INT8 or FP8 MatMul instead of BF16 / FP16.
        use_quantized_matmul_conv (`bool`, *optional*, defaults to `False`):
            Same as use_quantized_matmul_conv but for the convolutional layers with UNets like SDXL.
        use_stochastic_rounding (`bool`, *optional*, defaults to `False`):
            Enabling this option will use stochastic rounding on the quantization step.
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
        quantized_matmul_dtype: str = None,
        group_size: int = 0,
        svd_rank: int = 32,
        svd_steps: int = 8,
        use_svd: bool = False,
        use_grad_ckpt: bool = True,
        quant_conv: bool = False,
        use_quantized_matmul: bool = False,
        use_quantized_matmul_conv: bool = False,
        use_static_quantization: bool = True,
        use_stochastic_rounding: bool = False,
        dequantize_fp32: bool = False,
        non_blocking: bool = False,
        add_skip_keys: bool = True,
        quantization_device: Optional[torch.device] = None,
        return_device: Optional[torch.device] = None,
        modules_to_not_convert: Optional[List[str]] = None,
        modules_dtype_dict: Optional[Dict[str, List[str]]] = None,
        is_training: bool = False,
        **kwargs, # pylint: disable=unused-argument
    ):
        self.weights_dtype = weights_dtype
        self.quantized_matmul_dtype = quantized_matmul_dtype
        self.is_training = is_training
        if self.is_training:
            self.quant_method = QuantizationMethod.SDNQ_TRAINING
        else:
            self.quant_method = QuantizationMethod.SDNQ
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.svd_steps = svd_steps
        self.use_svd = use_svd
        self.use_grad_ckpt = use_grad_ckpt
        self.quant_conv = quant_conv
        self.use_quantized_matmul = use_quantized_matmul
        self.use_quantized_matmul_conv = use_quantized_matmul_conv
        self.use_static_quantization = use_static_quantization
        self.use_stochastic_rounding = use_stochastic_rounding
        self.dequantize_fp32 = dequantize_fp32
        self.non_blocking = non_blocking
        self.add_skip_keys = add_skip_keys
        self.quantization_device = quantization_device
        self.return_device = return_device
        self.modules_to_not_convert = modules_to_not_convert
        self.modules_dtype_dict = modules_dtype_dict
        self.is_integer = dtype_dict[self.weights_dtype]["is_integer"]
        self.sdnq_version = sdnq_version
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.use_quantized_matmul and not check_torch_compile():
            raise RuntimeError("SDNQ Quantized MatMul requires a working Triton install.")
        if self.weights_dtype not in accepted_weight_dtypes:
            raise ValueError(f"SDNQ only support weight dtypes in {accepted_weight_dtypes} but found {self.weights_dtype}")
        if self.quantized_matmul_dtype is not None and self.quantized_matmul_dtype not in accepted_matmul_dtypes:
            raise ValueError(f"SDNQ only support quantized matmul dtypes in {accepted_matmul_dtypes} but found {self.quantized_matmul_dtype}")

        if self.modules_to_not_convert is None:
            self.modules_to_not_convert = []
        elif isinstance(self.modules_to_not_convert, str):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        elif isinstance(self.modules_to_not_convert, tuple):
            self.modules_to_not_convert = list(self.modules_to_not_convert)
        elif not isinstance(self.modules_to_not_convert, list):
            raise ValueError(f"modules_to_not_convert must be a list but got {type(self.modules_to_not_convert)}")

        if self.modules_dtype_dict is None:
            self.modules_dtype_dict = {}
        elif not isinstance(self.modules_dtype_dict, dict):
            raise ValueError(f"modules_dtype_dict must be a dict but got {type(self.modules_dtype_dict)}")
        elif len(self.modules_dtype_dict.keys()) > 0:
            self.modules_dtype_dict = self.modules_dtype_dict.copy()
            for key, value in self.modules_dtype_dict.items():
                if isinstance(value, str):
                    value = [value]
                    self.modules_dtype_dict[key] = value
                elif isinstance(value, tuple):
                    value = list(value)
                    self.modules_dtype_dict[key] = value
                if not isinstance(key, str) or not isinstance(value, list):
                    raise ValueError(f"modules_dtype_dict must be a dictionary of strings and lists but got {type(key)} and {type(value)}")

        self.modules_to_not_convert = self.modules_to_not_convert.copy()
        self.modules_dtype_dict = self.modules_dtype_dict.copy()

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

sdnq_quantize_layer_weight_compiled = compile_func(sdnq_quantize_layer_weight)
