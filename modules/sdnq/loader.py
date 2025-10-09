import os
import json
import torch
from diffusers.models.modeling_utils import ModelMixin
from .common import use_tensorwise_fp8_matmul, use_contiguous_mm
from .quantizer import SDNQConfig, sdnq_post_load_quant
from .dequantizer import dequantize_symmetric, re_quantize_int8, re_quantize_fp8


def get_module_names(model: ModelMixin) -> list:
    modules_names = model._internal_dict.keys() # pylint: disable=protected-access
    modules_names = [m for m in modules_names if not m.startswith('_')]
    modules_names = [m for m in modules_names if isinstance(getattr(model, m, None), torch.nn.Module)]
    modules_names = sorted(set(modules_names))
    return modules_names


def save_sdnq_model(model: ModelMixin, model_path: str, max_shard_size: str = "10GB", is_pipeline: bool = False, sdnq_config: SDNQConfig = None) -> None:
    model.save_pretrained(model_path, max_shard_size=max_shard_size) # actual save

    quantization_config_path = os.path.join(model_path, "quantization_config.json")
    if sdnq_config is not None: # if provided, save global config
        sdnq_config.to_json_file(quantization_config_path)

    if is_pipeline:
        for module_name in get_module_names(model): # save per-module config if available
            module = getattr(model, module_name, None)
            if module is None:
                continue
            module_quantization_config_path = os.path.join(model_path, module_name, "quantization_config.json")
            if hasattr(module, "quantization_config") and isinstance(module.quantization_config, SDNQConfig):
                module.quantization_config.to_json_file(module_quantization_config_path)
            elif hasattr(module, "config") and hasattr(module.config, "quantization_config") and isinstance(module.config.quantization_config, SDNQConfig):
                module.config.quantization_config.to_json_file(module_quantization_config_path)
    elif sdnq_config is None:
        if hasattr(model, "quantization_config") and isinstance(model.quantization_config, SDNQConfig):
            model.quantization_config.to_json_file(quantization_config_path)
        elif hasattr(model, "config") and hasattr(model.config, "quantization_config") and isinstance(model.config.quantization_config, SDNQConfig):
            model.config.quantization_config.to_json_file(quantization_config_path)


def load_sdnq_model(model_path: str, model_cls: ModelMixin = None, file_name: str = None, dtype: torch.dtype = None, device: torch.device = 'cpu', dequantize_fp32: bool = None, use_quantized_matmul: bool = None, model_config: dict = None, quantization_config: dict = None) -> ModelMixin:
    from accelerate import init_empty_weights
    from safetensors.torch import safe_open

    with init_empty_weights():
        if quantization_config is None:
            try:
                with open(os.path.join(model_path, "quantization_config.json"), "r", encoding="utf-8") as f:
                    quantization_config = json.load(f)
            except Exception:
                quantization_config = {}

        if model_config is None:
            try:
                with open(os.path.join(model_path, 'config.json'), "r", encoding="utf-8") as f:
                    model_config = json.load(f)
            except Exception:
                model_config = {}

        if model_cls is None:
            import transformers
            import diffusers
            class_name = model_config.get("_class_name", None) or model_config.get("architectures", None)
            if isinstance(class_name, list):
                class_name = class_name[0]
            if class_name is not None:
                model_cls = getattr(diffusers, class_name, None) or getattr(transformers, class_name, None)
        if model_cls is None:
            raise ValueError(f"Cannot determine model class for {model_path}, please provide model_cls argument")

        quantization_config.pop("is_integer", None)
        quantization_config.pop("quant_method", None)
        quantization_config.pop("quantization_device", None)
        quantization_config.pop("return_device", None)
        quantization_config.pop("non_blocking", None)
        quantization_config.pop("add_skip_keys", None)

        if hasattr(model_cls, "load_config"):
            config = model_cls.load_config(model_path)
            model = model_cls.from_config(config)
        elif hasattr(model_cls, "_from_config"):
            config = transformers.PretrainedConfig.from_dict(model_config)
            model = model_cls(config)
        else:
            raise ValueError(f"Dont know how to load model for {model_cls}")

        model = sdnq_post_load_quant(model, add_skip_keys=False, **quantization_config)

    state_dict = {}
    files = []
    if file_name:
        files.append(os.path.join(model_path, file_name))
    else:
        all_files = os.listdir(model_path)
        files = sorted([os.path.join(model_path, f) for f in all_files if f.endswith(".safetensors")])

    for fn in files:
        with safe_open(fn, framework="pt", device=str(device)) as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    if model.__class__.__name__ in {"T5EncoderModel", "UMT5EncoderModel"} and "encoder.embed_tokens.weight" not in state_dict.keys():
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

    model.load_state_dict(state_dict, assign=True)
    del state_dict

    if (dtype is not None) or (dequantize_fp32 is not None) or (use_quantized_matmul is not None):
        model = apply_options_to_model(model, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul)
    return model


def apply_options_to_model(model, dtype: torch.dtype = None, dequantize_fp32: bool = None, use_quantized_matmul: bool = None):
    has_children = list(model.children())
    if not has_children:
        return model
    for module in model.children():
        if hasattr(module, "sdnq_dequantizer"):
            if dtype is not None:
                module.sdnq_dequantizer.result_dtype = dtype

            current_scale_dtype = module.svd_up.dtype if module.svd_up is not None else module.scale.dtype
            scale_dtype = torch.float32 if dequantize_fp32 is None and current_scale_dtype == torch.float32 else torch.float32 if dequantize_fp32 else module.sdnq_dequantizer.result_dtype
            upcast_scale = bool(not use_tensorwise_fp8_matmul and module.sdnq_dequantizer.weights_dtype in {"float8_e4m3fn", "float8_e5m2"} and (use_quantized_matmul or (use_quantized_matmul is None and module.sdnq_dequantizer.use_quantized_matmul)))

            if upcast_scale:
                module.scale.data = module.scale.to(dtype=torch.float32)
            else:
                module.scale.data = module.scale.to(dtype=scale_dtype)
            if module.zero_point is not None:
                module.zero_point.data = module.zero_point.to(dtype=scale_dtype)
            if module.svd_up is not None:
                module.svd_up.data = module.svd_up.to(dtype=scale_dtype)
                module.svd_down.data = module.svd_down.to(dtype=scale_dtype)

            if use_quantized_matmul is not None and use_quantized_matmul != module.sdnq_dequantizer.use_quantized_matmul:
                if module.sdnq_dequantizer.weights_dtype in {"int8", "float8_e4m3fn", "float8_e5m2"}:
                    if use_quantized_matmul and module.sdnq_dequantizer.re_quantize_for_matmul:
                        scale_dtype = module.scale.dtype
                        if module.sdnq_dequantizer.weights_dtype == "int8":
                            module.weight.data, module.scale.data = re_quantize_int8(dequantize_symmetric(module.weight, module.scale, torch.float32, module.sdnq_dequantizer.result_shape))
                            module.scale.data = module.scale.to(dtype=scale_dtype)
                        else:
                            is_e5 = bool(module.sdnq_dequantizer.weights_dtype == "float8_e5m2")
                            module.weight.data, module.scale.data = re_quantize_fp8(dequantize_symmetric(module.weight, module.scale, torch.float32, module.sdnq_dequantizer.result_shape), is_e5=is_e5)
                            if use_tensorwise_fp8_matmul:
                                module.scale.data = module.scale.to(dtype=scale_dtype)
                    elif not module.sdnq_dequantizer.re_quantize_for_matmul:
                        module.weight.data, module.scale.data = module.weight.t_(), module.scale.t_()
                    if use_quantized_matmul:
                        if use_contiguous_mm:
                            module.weight.data = module.weight.contiguous()
                        elif module.weight.is_contiguous():
                            module.weight.data = module.weight.t_().contiguous().t_()
                if module.svd_up is not None:
                    module.svd_up.data = module.svd_up.t_()
                    module.svd_down.data = module.svd_down.t_()
                module.sdnq_dequantizer.use_quantized_matmul = use_quantized_matmul
        module = apply_options_to_model(module, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul)
    return model
