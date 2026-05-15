import re
import torch

from .common import dtype_dict, common_skip_keys, module_skip_keys_dict, conv_types, conv_transpose_types


def check_param_name_in(param_name: str, param_list: list[str]) -> str:
    split_param_name = param_name.split(".")
    for param in param_list:
        if param.startswith("."):
            if param_name.startswith(param[1:]):
                return param
            else:
                continue
        if (
            param_name == param
            or param in split_param_name
            or ("*" in param and re.match(param.replace(".*", "\\.*").replace("*", ".*"), param_name))
        ):
            return param
    return None


def get_quant_args_from_config(quantization_config: dict) -> dict:
    from .quantizer import SDNQConfig
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
    quantization_config_dict.pop("use_dynamic_quantization", None)
    quantization_config_dict.pop("use_static_quantization", None)
    quantization_config_dict.pop("use_stochastic_rounding", None)
    quantization_config_dict.pop("use_grad_ckpt", None)
    quantization_config_dict.pop("is_training", None)
    quantization_config_dict.pop("sdnq_version", None)
    if quantization_config_dict.get("modules_quant_config", None) is not None:
        for key in quantization_config_dict["modules_quant_config"].keys():
            quantization_config_dict["modules_quant_config"][key] = get_quant_args_from_config(quantization_config_dict["modules_quant_config"][key])
    return quantization_config_dict


def get_minimum_dtype(weights_dtype: str, param_name: str, modules_dtype_dict: dict[str, list[str]]):
    if len(modules_dtype_dict.keys()) > 0:
        for key, value in modules_dtype_dict.items():
            if check_param_name_in(param_name, value) is not None:
                key = key.lower()
                if key.startswith("minimum") or key.endswith("bit") or key.endswith("bits"):
                    minimum_bits_str = key.removeprefix("minimum").removeprefix("-").removeprefix("_").removesuffix("bits").removesuffix("bit").removesuffix("-").removesuffix("_")
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


def get_quant_kwargs(layer: torch.nn.Module, quantization_config, torch_dtype: torch.dtype | None = None, param_name: str = "", **kwargs) -> dict:
    from .quantizer import SDNQConfig
    if not isinstance(quantization_config, SDNQConfig):
        quantization_config = SDNQConfig(**quantization_config)
    layer_class_name = layer.__class__.__name__

    quant_kwargs = {
        "weights_dtype": quantization_config.weights_dtype,
        "quantized_matmul_dtype": quantization_config.quantized_matmul_dtype,
        "group_size": quantization_config.group_size,
        "svd_rank": quantization_config.svd_rank,
        "svd_steps": quantization_config.svd_steps,
        "dynamic_loss_threshold": quantization_config.dynamic_loss_threshold,
        "use_svd": quantization_config.use_svd,
        "use_quantized_matmul": quantization_config.use_quantized_matmul,
        "use_quantized_matmul_conv": quantization_config.use_quantized_matmul_conv,
        "use_dynamic_quantization": quantization_config.use_dynamic_quantization,
        "use_stochastic_rounding": quantization_config.use_stochastic_rounding,
        "dequantize_fp32": quantization_config.dequantize_fp32,
        "non_blocking": quantization_config.non_blocking,
        "quantization_device": quantization_config.quantization_device,
        "return_device": quantization_config.return_device,
        "layer_class_name": layer_class_name,
        "torch_dtype": torch_dtype,
        "param_name": param_name,
    }

    for key, value in kwargs.items():
        quant_kwargs[key] = value

    param_key = check_param_name_in(quant_kwargs["param_name"], quantization_config.modules_quant_config.keys())
    if param_key is not None:
        for key, value in quantization_config.modules_quant_config[param_key].items():
            quant_kwargs[key] = value

    if layer_class_name in conv_transpose_types or layer_class_name in conv_types:
        quant_kwargs["use_quantized_matmul"] = quant_kwargs.pop("use_quantized_matmul_conv")
    else:
        quant_kwargs.pop("use_quantized_matmul_conv")

    if not quant_kwargs["use_dynamic_quantization"]:
        quant_kwargs.pop("dynamic_loss_threshold")

    quant_kwargs["weights_dtype"] = get_minimum_dtype(quant_kwargs["weights_dtype"], quant_kwargs["param_name"], quantization_config.modules_dtype_dict)
    if check_param_name_in(quant_kwargs["param_name"], quantization_config.modules_to_not_use_matmul) is not None:
        quant_kwargs["use_quantized_matmul"] = False

    return quant_kwargs


def add_module_skip_keys(model: torch.nn.Module, quantization_config):
    if getattr(model, "_keep_in_fp32_modules", None) is not None:
        quantization_config.modules_to_not_convert.extend(model._keep_in_fp32_modules) # pylint: disable=protected-access
    if getattr(model, "_tied_weights_keys", None) is not None:
        if isinstance(model._tied_weights_keys, dict): # pylint: disable=protected-access
            quantization_config.modules_to_not_convert.extend(model._tied_weights_keys.keys()) # pylint: disable=protected-access
            quantization_config.modules_to_not_convert.extend(model._tied_weights_keys.values()) # pylint: disable=protected-access
        else:
            quantization_config.modules_to_not_convert.extend(model._tied_weights_keys) # pylint: disable=protected-access

    skip_key_list = module_skip_keys_dict.get(model.__class__.__name__, None)
    if skip_key_list is not None:
        quantization_config.modules_to_not_convert.extend(skip_key_list[0])
        for key, value in skip_key_list[1].items():
            if key in quantization_config.modules_dtype_dict.keys():
                quantization_config.modules_dtype_dict[key].extend(value)
            else:
                quantization_config.modules_dtype_dict[key] = value

        if quantization_config.quantized_matmul_dtype is None:
            if dtype_dict[quantization_config.weights_dtype]["is_integer"]:
                quantized_matmul_dtype = "int8"
            elif dtype_dict[quantization_config.weights_dtype]["num_bits"] < 16:
                quantized_matmul_dtype = "float8_e4m3fn"
            else:
                quantized_matmul_dtype = "float16"
        else:
            quantized_matmul_dtype = quantization_config.quantized_matmul_dtype
        quantization_config.modules_to_not_use_matmul.extend(skip_key_list[2].get(quantized_matmul_dtype, []))
    else:
        quantization_config.modules_to_not_convert.extend(common_skip_keys)
        if getattr(model, "_skip_layerwise_casting_patterns", None) is not None:
            quantization_config.modules_to_not_convert.extend(model._skip_layerwise_casting_patterns) # pylint: disable=protected-access

    # dedupe
    quantization_config.modules_to_not_convert = list(set(quantization_config.modules_to_not_convert))
    quantization_config.modules_to_not_use_matmul = list(set(quantization_config.modules_to_not_use_matmul))
    for key, value in quantization_config.modules_dtype_dict.items():
        quantization_config.modules_dtype_dict[key] = list(set(value))

    return model, quantization_config
