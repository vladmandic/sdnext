# pylint: disable=redefined-builtin,no-member,protected-access

from dataclasses import dataclass
from enum import Enum

import math
import torch

from transformers.quantizers import HfQuantizer
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin

from diffusers.utils import get_module_from_name
from accelerate import init_empty_weights

from modules import devices, shared
from .common import sdnq_version, dtype_dict, accepted_weight_dtypes, accepted_matmul_dtypes, weights_dtype_order, allowed_types, linear_types, embedding_types, conv_types, conv_transpose_types, compile_func, is_fp8_mm_supported, use_tensorwise_fp8_matmul, check_torch_compile
from .dequantizer import SDNQDequantizer, dequantize_sdnq_model
from .packed_int import pack_int
from .packed_float import pack_float
from .forward import get_forward_func
from .layers import get_sdnq_wrapper_class

from .quant_utils import quantize_weight, apply_svdquant, rotate_hadamard, prepare_weight_for_matmul, prepare_svd_for_matmul
from .utils import check_param_name_in, get_quant_args_from_config, get_quant_kwargs, add_module_skip_keys


class QuantizationMethod(str, Enum):
    SDNQ = "sdnq"
    SDNQ_TRAINING = "sdnq_training"


@devices.inference_context()
def sdnq_quantize_layer_weight(weight, layer_class_name=None, weights_dtype="int8", quantized_matmul_dtype=None, torch_dtype=None, group_size=0, hadamard_group_size=128, svd_rank=32, svd_steps=8, use_svd=False, use_hadamard=False, use_quantized_matmul=False, use_stochastic_rounding=False, dequantize_fp32=True, using_pre_calculated_svd=False, skip_sr=False, param_name=None): # pylint: disable=unused-argument
    num_of_groups = 1
    is_conv_type = False
    is_conv_transpose_type = False
    is_linear_type = False
    result_shape = None
    scale_dtype = None

    original_shape = weight.shape
    original_stride = weight.stride()
    weight = weight.detach()

    if torch_dtype is None:
        torch_dtype = weight.dtype
    if quantized_matmul_dtype is None:
        if dtype_dict[weights_dtype]["is_integer"]:
            quantized_matmul_dtype = "int8"
        elif dtype_dict[weights_dtype]["num_bits"] < 16:
            quantized_matmul_dtype = "float8_e4m3fn"
        else:
            quantized_matmul_dtype = "float16"

    re_quantize_for_matmul = bool(
        dtype_dict[weights_dtype]["is_unsigned"]
        or dtype_dict[weights_dtype]["is_integer"] != dtype_dict[quantized_matmul_dtype]["is_integer"]
        or dtype_dict[weights_dtype]["num_bits"] > dtype_dict[quantized_matmul_dtype]["num_bits"]
        or (
            dtype_dict[weights_dtype]["is_packed"]
            and not dtype_dict[weights_dtype]["is_integer"]
            and not dtype_dict[quantized_matmul_dtype]["is_integer"]
            and (
                    dtype_dict[weights_dtype]["num_bits"] >= dtype_dict[quantized_matmul_dtype]["num_bits"]
                    or dtype_dict[weights_dtype]["max"] > dtype_dict[quantized_matmul_dtype]["max"]
                )
        )
    )

    if layer_class_name in conv_types:
        is_conv_type = True
        reduction_axes = 1
        output_channel_size, channel_size = weight.shape[:2]
        use_quantized_matmul = use_quantized_matmul and channel_size >= 32 and output_channel_size >= 32 and output_channel_size % 16 == 0 and channel_size % 16 == 0
        if use_quantized_matmul and not re_quantize_for_matmul and not dtype_dict[weights_dtype]["is_packed"]:
            result_shape = weight.shape
            weight = weight.flatten(1,-1)
            reduction_axes = -1
    elif layer_class_name in conv_transpose_types:
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
        use_quantized_matmul = use_quantized_matmul and channel_size >= 32 and output_channel_size >= 32 and output_channel_size % 16 == 0 and channel_size % 16 == 0
    else:
        if weight.ndim > 1:
            output_channel_size, channel_size = weight.shape[-2:]
        else:
            output_channel_size, channel_size = 1, weight.shape[-1]
        reduction_axes = -1
        use_quantized_matmul = False

    if (
        not dequantize_fp32
        and dtype_dict[weights_dtype]["max"] <= 16384 # 1/fp16_min_normal
        and not (
            use_quantized_matmul
            and not dtype_dict[quantized_matmul_dtype]["is_integer"]
            and (not use_tensorwise_fp8_matmul or dtype_dict[quantized_matmul_dtype]["num_bits"] == 16)
        )
    ):
        scale_dtype = torch_dtype

    if use_hadamard:
        if channel_size % hadamard_group_size != 0:
            hadamard_pow2 = int(math.log2(hadamard_group_size))
            while channel_size % hadamard_group_size != 0:
                hadamard_pow2 -= 1
                hadamard_group_size = 2 ** hadamard_pow2
        if hadamard_group_size < 4:
            use_hadamard = False
    if use_hadamard:
        weight = rotate_hadamard(weight, group_size=hadamard_group_size, is_conv=bool(is_conv_type or is_conv_transpose_type))

    if use_svd:
        try:
            weight, svd_up, svd_down = apply_svdquant(weight, rank=svd_rank, niter=svd_steps, dtype=torch_dtype)
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
            group_size = 2 ** ((3 if (svd_up is not None or using_pre_calculated_svd) else 2) + dtype_dict[weights_dtype]["num_bits"])
        else:
            group_size = 2 ** ((2 if (svd_up is not None or using_pre_calculated_svd) else 1) + dtype_dict[weights_dtype]["num_bits"])

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
            if is_conv_type:
                # output_channel_size, channel_size, X, X
                # output_channel_size, num_of_groups, group_size, X, X
                reduction_axes = 2
                weight = weight.unflatten(1, (num_of_groups, group_size))
            elif is_conv_transpose_type:
                #channel_size, output_channel_size, X, X
                #num_of_groups, group_size, output_channel_size, X, X
                reduction_axes = 1
                weight = weight.unflatten(1, (group_size, num_of_groups))
            else:
                # output_channel_size, channel_size
                # output_channel_size, num_of_groups, group_size
                reduction_axes = -1
                weight = weight.unflatten(-1, (num_of_groups, group_size))
        else:
            group_size = -1


    cast_scale = True
    transpose_weights = False
    re_quantize_for_matmul = re_quantize_for_matmul or num_of_groups > 1
    if use_quantized_matmul and not re_quantize_for_matmul and not dtype_dict[weights_dtype]["is_packed"]:
        transpose_weights = True
        if not use_tensorwise_fp8_matmul and not dtype_dict[quantized_matmul_dtype]["is_integer"]:
            cast_scale = False

    weight, scale, zero_point = quantize_weight(weight, reduction_axes, weights_dtype, dtype=(scale_dtype if cast_scale else None), use_stochastic_rounding=(use_stochastic_rounding and not skip_sr))

    if transpose_weights:
        scale.t_()
        weight.t_()
        weight = prepare_weight_for_matmul(weight)

    quantized_weight_shape = weight.shape
    if dtype_dict[weights_dtype]["is_packed"]:
        if dtype_dict[weights_dtype]["is_integer"]:
            weight = pack_int(weight, weights_dtype)
        else:
            weight = pack_float(weight, weights_dtype)
    else:
        weight = weight.to(dtype=dtype_dict[weights_dtype]["torch_dtype"])

    sdnq_dequantizer = SDNQDequantizer(
        result_dtype=torch_dtype,
        result_shape=result_shape,
        original_shape=original_shape,
        original_stride=original_stride,
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        hadamard_group_size=hadamard_group_size,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_quantized_matmul=use_quantized_matmul,
        re_quantize_for_matmul=re_quantize_for_matmul,
        use_stochastic_rounding=use_stochastic_rounding,
        use_hadamard=use_hadamard,
        layer_class_name=layer_class_name,
    )

    return sdnq_dequantizer, {"weight": weight, "scale": scale, "zero_point": zero_point, "svd_up": svd_up, "svd_down": svd_down}


@devices.inference_context()
def sdnq_quantize_layer_weight_dynamic(weight, layer_class_name=None, weights_dtype="uint4", quantized_matmul_dtype=None, group_size=0, hadamard_group_size=128, svd_rank=32, svd_steps=8, dynamic_loss_threshold=None, use_svd=False, use_hadamard=False, use_quantized_matmul=False, use_stochastic_rounding=False, dequantize_fp32=True, quantization_config=None, torch_dtype=None, param_name=None): # pylint: disable=unused-argument
    if torch_dtype is None:
        torch_dtype = weight.dtype
    if dynamic_loss_threshold is None or dynamic_loss_threshold < 0:
        dynamic_loss_threshold = 10 ** -(dtype_dict[weights_dtype]["num_bits"] / 2)

    if weight.dtype != torch.float64:
        weight = weight.to(dtype=torch.float32)
    weight_std = weight.std().square_().clamp_(min=1e-8)

    if use_svd:
        try:
            svd_weight, svd_up, svd_down = apply_svdquant(weight, rank=svd_rank, niter=svd_steps, dtype=torch_dtype)
            svd_up, svd_down = prepare_svd_for_matmul(svd_up, svd_down, False)
            if use_quantized_matmul:
                svd_up_t, svd_down_t = svd_up.clone().t_(), svd_down.clone().t_()
                svd_up_t, svd_down_t = prepare_svd_for_matmul(svd_up, svd_down, True)
            else:
                svd_up_t, svd_down_t = None, None
        except Exception:
            svd_up, svd_down = None, None
            svd_up_t, svd_down_t = None, None
            svd_weight = weight
    else:
        svd_up, svd_down = None, None
        svd_up_t, svd_down_t = None, None
        svd_weight = weight

    quantization_loss = None
    for i in range(weights_dtype_order.index(weights_dtype), len(weights_dtype_order)):
        current_weights_dtype = weights_dtype_order[i]
        if quantized_matmul_dtype is None and not is_fp8_mm_supported and not dtype_dict[current_weights_dtype]["is_integer"] and dtype_dict[current_weights_dtype]["num_bits"] < 16:
            current_use_quantized_matmul = False
        else:
            current_use_quantized_matmul = use_quantized_matmul

        sdnq_dequantizer, weight_data = sdnq_quantize_layer_weight(
            svd_weight,
            layer_class_name=layer_class_name,
            weights_dtype=current_weights_dtype,
            quantized_matmul_dtype=quantized_matmul_dtype,
            torch_dtype=torch_dtype,
            hadamard_group_size=hadamard_group_size,
            group_size=group_size,
            svd_rank=svd_rank,
            svd_steps=svd_steps,
            use_svd=False,
            use_hadamard=use_hadamard,
            using_pre_calculated_svd=use_svd,
            use_quantized_matmul=current_use_quantized_matmul,
            use_stochastic_rounding=use_stochastic_rounding,
            dequantize_fp32=dequantize_fp32,
            param_name=param_name,
        )

        if sdnq_dequantizer.use_quantized_matmul:
            weight_data["svd_up"] = svd_up_t
            weight_data["svd_down"] = svd_down_t
        else:
            weight_data["svd_up"] = svd_up
            weight_data["svd_down"] = svd_down

        quantization_loss = torch.nn.functional.mse_loss(weight, sdnq_dequantizer(**weight_data, skip_quantized_matmul=sdnq_dequantizer.use_quantized_matmul, dtype=weight.dtype, skip_compile=True)).div_(weight_std)
        if quantization_loss <= dynamic_loss_threshold:
            return sdnq_dequantizer, weight_data
    return None


@devices.inference_context()
def sdnq_quantize_layer(layer, quantization_config: "SDNQConfig", torch_dtype: torch.dtype | None = None, param_name: str = "", quant_kwargs: dict | None = None): # pylint: disable=unused-argument
    if quant_kwargs is None:
        quant_kwargs = get_quant_kwargs(layer, quantization_config, torch_dtype=torch_dtype, param_name=param_name)

    layer_class_name = layer.__class__.__name__
    if (
        (layer_class_name in embedding_types and not quantization_config.quant_embedding)
        or ((layer_class_name in conv_transpose_types or layer_class_name in conv_types) and not quantization_config.quant_conv)
    ):
        quantization_config.modules_to_not_convert.append(param_name)
        return layer, quantization_config

    return_device = quant_kwargs.pop("return_device")
    quantization_device = quant_kwargs.pop("quantization_device")
    non_blocking = quant_kwargs.pop("non_blocking")
    use_dynamic_quantization = quant_kwargs.pop("use_dynamic_quantization")

    layer.weight.requires_grad_(False)
    if return_device is None:
        return_device = layer.weight.device
    if quantization_device is not None:
        layer.weight.data = layer.weight.to(quantization_device, non_blocking=non_blocking)

    if use_dynamic_quantization:
        weight_data = sdnq_quantize_layer_weight_dynamic(layer.weight, **quant_kwargs)
    else:
        weight_data = sdnq_quantize_layer_weight(layer.weight, **quant_kwargs)

    if weight_data is not None:
        layer.sdnq_dequantizer, weight_data = weight_data
        layer = get_sdnq_wrapper_class(layer, get_forward_func(layer_class_name, layer.sdnq_dequantizer.quantized_matmul_dtype, layer.sdnq_dequantizer.use_quantized_matmul))

        for key, value in weight_data.items():
            if isinstance(value, (torch.Tensor, torch.nn.Parameter)):
                setattr(layer, key, torch.nn.Parameter(value.to(return_device, non_blocking=non_blocking), requires_grad=False))
                setattr(getattr(layer, key), "_is_hf_initialized", True) # noqa: B010
            else:
                setattr(layer, key, value)
        del weight_data

        if use_dynamic_quantization:
            if layer.sdnq_dequantizer.weights_dtype not in quantization_config.modules_dtype_dict.keys():
                quantization_config.modules_dtype_dict[layer.sdnq_dequantizer.weights_dtype] = [param_name]
            else:
                quantization_config.modules_dtype_dict[layer.sdnq_dequantizer.weights_dtype].append(param_name)

        if quant_kwargs["use_quantized_matmul"] and not layer.sdnq_dequantizer.use_quantized_matmul:
            quantization_config.modules_to_not_use_matmul.append(param_name)
    else:
        layer.weight = layer.weight.to(return_device, dtype=torch_dtype, non_blocking=non_blocking)
        if use_dynamic_quantization:
            quantization_config.modules_to_not_convert.append(param_name)

    return layer, quantization_config


@devices.inference_context()
def apply_sdnq_to_module(model, quantization_config: "SDNQConfig", torch_dtype: torch.dtype | None = None, full_param_name: str = ""): # pylint: disable=unused-argument
    if not list(model.children()):
        return model, quantization_config
    for module_name, module in model.named_children():
        if full_param_name:
            param_name = full_param_name + "." + module_name
        else:
            param_name = module_name
        if hasattr(module, "weight") and module.weight is not None:
            param_name = param_name + ".weight"
            layer_class_name = module.__class__.__name__
            if (
                layer_class_name in allowed_types
                and module.weight.dtype in {torch.float64, torch.float32, torch.float16, torch.bfloat16}
                and check_param_name_in(param_name, quantization_config.modules_to_not_convert) is None
                and not (layer_class_name in embedding_types and not quantization_config.quant_embedding)
                and not ((layer_class_name in conv_types or layer_class_name in conv_transpose_types) and not quantization_config.quant_conv)
            ):
                module, quantization_config = sdnq_quantize_layer(module, quantization_config, torch_dtype=torch_dtype, param_name=param_name)
                setattr(model, module_name, module)
            else:
                quantization_config.modules_to_not_convert.append(param_name)

        module, quantization_config = apply_sdnq_to_module(module, quantization_config, torch_dtype=torch_dtype, full_param_name=param_name)
        setattr(model, module_name, module)
    return model, quantization_config


@devices.inference_context()
def sdnq_post_load_quant(
    model: torch.nn.Module,
    weights_dtype: str = "int8",
    quantized_matmul_dtype: str | None = None,
    hadamard_group_size: int = 128,
    group_size: int = 0,
    svd_rank: int = 32,
    svd_steps: int = 8,
    dynamic_loss_threshold: float | None = None,
    use_svd: bool = False,
    use_hadamard: bool = False,
    quant_conv: bool = False,
    quant_embedding: bool = False,
    use_quantized_matmul: bool = False,
    use_quantized_matmul_conv: bool = False,
    use_dynamic_quantization: bool = False,
    use_stochastic_rounding: bool = False,
    dequantize_fp32: bool = True,
    non_blocking: bool = False,
    add_skip_keys:bool = True,
    modules_to_not_convert: list[str] | None = None,
    modules_to_not_use_matmul: list[str] | None = None,
    modules_dtype_dict: dict[str, list[str]] | None = None,
    modules_quant_config: dict[str, dict] | None = None,
    quantization_device: torch.device | None = None,
    return_device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
):
    quantization_config = SDNQConfig(
        weights_dtype=weights_dtype,
        hadamard_group_size=hadamard_group_size,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        dynamic_loss_threshold=dynamic_loss_threshold,
        use_svd=use_svd,
        use_hadamard=use_hadamard,
        quant_conv=quant_conv,
        quant_embedding=quant_embedding,
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=use_quantized_matmul_conv,
        use_dynamic_quantization=use_dynamic_quantization,
        use_stochastic_rounding=use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        add_skip_keys=add_skip_keys,
        modules_to_not_convert=modules_to_not_convert,
        modules_to_not_use_matmul=modules_to_not_use_matmul,
        modules_dtype_dict=modules_dtype_dict,
        modules_quant_config=modules_quant_config,
        quantization_device=quantization_device,
        return_device=return_device,
    )
    if add_skip_keys:
        model, quantization_config = add_module_skip_keys(model, quantization_config)

    model.eval()
    model, quantization_config = apply_sdnq_to_module(model, quantization_config, torch_dtype=torch_dtype)

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


class SDNQQuantize:
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None, # pylint: disable=unused-argument
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
            if not check_param_name_in(param_name, self.quantization_config.modules_to_not_convert) is not None:
                layer_class_name = get_module_from_name(model, param_name)[0].__class__.__name__
                if layer_class_name in allowed_types:
                    if layer_class_name in embedding_types:
                        if self.quantization_config.quant_embedding:
                            return True
                    elif layer_class_name in conv_types or layer_class_name in conv_transpose_types:
                        if self.quantization_config.quant_conv:
                            return True
                    else:
                        return True
            self.quantization_config.modules_to_not_convert.append(param_name)
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
        layer, tensor_name = get_module_from_name(model, param_name)

        if self.pre_quantized:
            if param_value is not None:
                if tensor_name == "weight":
                    return_dtype = param_value.dtype
                elif self.quantization_config.dequantize_fp32:
                    if param_value.dtype != torch.float64 and self.torch_dtype != torch.float64:
                        return_dtype = torch.float32
                    else:
                        return_dtype = torch.float64
                else:
                    return_dtype = kwargs.get("dtype", param_value.dtype if self.torch_dtype is None else self.torch_dtype)

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
        quant_kwargs = get_quant_kwargs(layer, self.quantization_config, torch_dtype=torch_dtype, param_name=param_name)
        if quant_kwargs["return_device"] is None:
            quant_kwargs["return_device"] = target_device
        if quant_kwargs["quantization_device"] is not None:
            target_device = quant_kwargs["quantization_device"]
            quant_kwargs["quantization_device"] = None

        if param_value.dtype in {torch.float32, torch.float64} and devices.same_device(param_value.device, target_device):
            param_value = param_value.clone()
        else:
            param_value = param_value.to(target_device, non_blocking=self.quantization_config.non_blocking).to(dtype=torch.float32 if param_value.dtype != torch.float64 else torch.float64)

        layer.weight = torch.nn.Parameter(param_value, requires_grad=False)
        layer, self.quantization_config = sdnq_quantize_layer(layer, self.quantization_config, torch_dtype=torch_dtype, param_name=param_name, quant_kwargs=quant_kwargs)

        parent_module, tensor_name = get_module_from_name(model, param_name.removesuffix(tensor_name).removesuffix("."))
        setattr(parent_module, tensor_name, layer)

    def get_quantize_ops(self):
        return SDNQQuantize(self)

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        max_memory = {key: val * 0.80 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: torch.dtype) -> torch.dtype: # pylint: disable=unused-argument,arguments-renamed
        return dtype_dict[self.quantization_config.weights_dtype]["target_dtype"]

    def _process_model_before_weight_loading( # pylint: disable=arguments-differ
        self,
        model,
        device_map, # pylint: disable=unused-argument
        keep_in_fp32_modules: list[str] | None = None,
        **kwargs, # pylint: disable=unused-argument
    ):
        if self.pre_quantized:
            self.quantization_config.quantization_device = None
            self.quantization_config.return_device = None
            self.quantization_config.non_blocking = False
            self.quantization_config.add_skip_keys = False

            with init_empty_weights():
                model = sdnq_post_load_quant(model, torch_dtype=self.torch_dtype, add_skip_keys=False, use_dynamic_quantization=False, **get_quant_args_from_config(self.quantization_config))

        if self.quantization_config.add_skip_keys:
            if keep_in_fp32_modules is not None:
                self.quantization_config.modules_to_not_convert.extend(keep_in_fp32_modules)
            if hasattr(self, "get_modules_to_not_convert") and hasattr(model, "tie_weights"):
                self.quantization_config.modules_to_not_convert.extend(self.get_modules_to_not_convert(model, add_default_skips=True))
            model, self.quantization_config = add_module_skip_keys(model, self.quantization_config)


    def _process_model_after_weight_loading(self, model, **kwargs): # pylint: disable=unused-argument
        model.quantization_config = self.quantization_config
        model.quantization_method = QuantizationMethod.SDNQ
        if hasattr(model, "config"):
            try:
                model.config.quantization_config = self.quantization_config
            except Exception:
                pass
            try:
                model.config["quantization_config"] = self.quantization_config.to_dict()
            except Exception:
                pass

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
            The target dtype for the weights after quantization.
            Check out `sdnq.common.accepted_weight_dtypes` for all the supported values.
            These are some of the recommended values to use: ("int8", "int7", "int6", "uint5", "uint4", "uint3", "uint2", "float8_e4m3fn", "float7_e3m3fn", "float6_e3m2fn", "float5_e2m2fn", "float4_e2m1fn", "float3_e1m1fn", "float2_e1m0fn")
        quantized_matmul_dtype (`str`, *optional*, defaults to `None`):
            The target dtype for quantized matmul.
            `None` will use "int8" with integer weight dtypes and "float8_e4m3fn" or "float16" with float weight dtypes.
            Supported values are: ("int8", "float8_e4m3fn", "float16")
        hadamard_group_size (`int`, *optional*, defaults to `0`):
            Used to decide how many elements of a tensor will share the same Hadamard rotation group.
        group_size (`int`, *optional*, defaults to `0`):
            Used to decide how many elements of a tensor will share the same quantization group.
            group_size = 0 will automatically select a group size based on weights_dtype.
        svd_rank (`int`, *optional*, defaults to `32`):
            The rank size used for the SVDQuant algorithm.
        dynamic_loss_threshold (`float`, *optional*, defaults to `None`):
            The target quantization mse loss threshold to use for dynamic quantization.
            The value `None` or negative values means auto select a threshold based on the weights_dtype.
        svd_steps (`int`, *optional*, defaults to `8`):
            The number of iterations to use in svd lowrank estimation.
        use_svd (`bool`, *optional*, defaults to `False`):
            Enabling this option will use SVDQuant algorithm on top of SDNQ quantization.
        use_hadamard (`bool`, *optional*, defaults to `False`):
            Enabling this option will use Hadamard rotation on top of SDNQ quantization.
        quant_conv (`bool`, *optional*, defaults to `False`):
            Enabling this option will quantize the convolutional layers in UNet models too.
        quant_embedding (`bool`, *optional*, defaults to `False`):
            Enabling this option will quantize the embedding layers in text models too.
        use_quantized_matmul (`bool`, *optional*, defaults to `False`):
            Enabling this option will use quantized INT8 or FP8 MatMul instead of BF16 / FP16.
        use_quantized_matmul_conv (`bool`, *optional*, defaults to `False`):
            Same as use_quantized_matmul_conv but for the convolutional layers with UNets like SDXL.
        use_stochastic_rounding (`bool`, *optional*, defaults to `False`):
            Enabling this option will use stochastic rounding on the quantization step.
        use_dynamic_quantization (`bool`, *optional*, defaults to `False`):
            Enabling this option will dynamically select a per layer quantization type based on the dynamic_loss_threshold.
            weights_dtype will be used as the minimum allowed quantization type when this option is enabled.
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
            The list of modules to not quantize. Useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
        modules_dtype_dict (`dict`, *optional*, default to `None`):
            The dict of dtypes and list of modules. Useful for quantizing some modules with a different dtype.
        modules_quant_config (`dict`, *optional*, default to `None`):
            The dict of modules and a dict of quantization kwargs to use for that module.
            Useful for quantizing some modules with a different quantization config.
    """

    def __init__( # pylint: disable=super-init-not-called
        self,
        weights_dtype: str = "int8",
        quantized_matmul_dtype: str | None = None,
        hadamard_group_size: int = 128,
        group_size: int = 0,
        svd_rank: int = 32,
        svd_steps: int = 8,
        dynamic_loss_threshold: float | None = None,
        use_svd: bool = False,
        use_hadamard: bool = False,
        use_grad_ckpt: bool = True,
        quant_conv: bool = False,
        quant_embedding: bool = False,
        use_quantized_matmul: bool = False,
        use_quantized_matmul_conv: bool = False,
        use_static_quantization: bool = True,
        use_dynamic_quantization: bool = False,
        use_stochastic_rounding: bool = False,
        dequantize_fp32: bool = True,
        non_blocking: bool = False,
        add_skip_keys: bool = True,
        quantization_device: torch.device | None = None,
        return_device: torch.device | None = None,
        modules_to_not_convert: list[str] | None = None,
        modules_to_not_use_matmul: list[str] | None = None,
        modules_dtype_dict: dict[str, list[str]] | None = None,
        modules_quant_config: dict[str, dict] | None = None,
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
        self.hadamard_group_size = hadamard_group_size
        self.group_size = group_size
        self.svd_rank = svd_rank
        self.dynamic_loss_threshold = dynamic_loss_threshold
        self.svd_steps = svd_steps
        self.use_svd = use_svd
        self.use_hadamard = use_hadamard
        self.use_grad_ckpt = use_grad_ckpt
        self.quant_conv = quant_conv
        self.quant_embedding = quant_embedding
        self.use_quantized_matmul = use_quantized_matmul
        self.use_quantized_matmul_conv = use_quantized_matmul_conv
        self.use_static_quantization = use_static_quantization
        self.use_dynamic_quantization = use_dynamic_quantization
        self.use_stochastic_rounding = use_stochastic_rounding
        self.dequantize_fp32 = dequantize_fp32
        self.non_blocking = non_blocking
        self.add_skip_keys = add_skip_keys
        self.quantization_device = quantization_device
        self.return_device = return_device
        self.modules_to_not_convert = modules_to_not_convert
        self.modules_to_not_use_matmul = modules_to_not_use_matmul
        self.modules_dtype_dict = modules_dtype_dict
        self.modules_quant_config = modules_quant_config
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

        if self.modules_to_not_use_matmul is None:
            self.modules_to_not_use_matmul = []
        elif isinstance(self.modules_to_not_use_matmul, str):
            self.modules_to_not_use_matmul = [self.modules_to_not_use_matmul]
        elif isinstance(self.modules_to_not_use_matmul, tuple):
            self.modules_to_not_use_matmul = list(self.modules_to_not_use_matmul)
        elif not isinstance(self.modules_to_not_use_matmul, list):
            raise ValueError(f"modules_to_not_use_matmul must be a list but got {type(self.modules_to_not_use_matmul)}")

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

        if self.modules_quant_config is None:
            self.modules_quant_config = {}

        self.modules_to_not_convert = self.modules_to_not_convert.copy()
        self.modules_to_not_use_matmul = self.modules_to_not_use_matmul.copy()
        self.modules_dtype_dict = self.modules_dtype_dict.copy()
        self.modules_quant_config = self.modules_quant_config.copy()

        # dedupe
        self.modules_to_not_convert = list(set(self.modules_to_not_convert))
        self.modules_to_not_use_matmul = list(set(self.modules_to_not_use_matmul))
        for key, value in self.modules_dtype_dict.items():
            self.modules_dtype_dict[key] = list(set(value))

    def to_dict(self):
        quantization_config_dict = self.__dict__.copy() # make serializable
        quantization_config_dict["quantization_device"] = str(quantization_config_dict["quantization_device"]) if quantization_config_dict["quantization_device"] is not None else None
        quantization_config_dict["return_device"] = str(quantization_config_dict["return_device"]) if quantization_config_dict["return_device"] is not None else None
        return quantization_config_dict


import diffusers.quantizers.auto # noqa: E402,RUF100 # pylint: disable=wrong-import-order
diffusers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq"] = SDNQQuantizer
diffusers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq"] = SDNQConfig

diffusers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq_training"] = SDNQQuantizer
diffusers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq_training"] = SDNQConfig

import transformers.quantizers.auto # noqa: E402,RUF100 # pylint: disable=wrong-import-order
transformers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq"] = SDNQQuantizer
transformers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq"] = SDNQConfig

transformers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq_training"] = SDNQQuantizer
transformers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq_training"] = SDNQConfig

sdnq_quantize_layer_weight_compiled = compile_func(sdnq_quantize_layer_weight)
