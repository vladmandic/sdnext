import os
import json
import torch
from safetensors import safe_open
from diffusers.models.modeling_utils import ModelMixin

from .common import use_tensorwise_fp8_matmul, use_contiguous_mm
from .quantizer import SDNQConfig, apply_sdnq_to_module
from .dequantizer import dequantize_symmetric, re_quantize_int8, re_quantize_fp8


def save_sdnq_model(model: ModelMixin, sdnq_config: SDNQConfig, model_path: str, max_shard_size: str = "10GB") -> None:
    model.save_pretrained(model_path, max_shard_size=max_shard_size)
    sdnq_config.to_json_file(os.path.join(model_path, "quantization_config.json"))


def load_sdnq_model(model_cls: ModelMixin, model_path: str, file_name: str = "diffusion_pytorch_model.safetensors", dtype: torch.dtype = None, dequantize_fp32: bool = None, use_quantized_matmul: bool = False) -> ModelMixin:
    with torch.device("meta"):
        with open(os.path.join(model_path, "quantization_config.json"), "r", encoding="utf-8") as f:
            quantization_config = json.load(f)
        quantization_config.pop("is_integer", None)
        quantization_config.pop("quant_method", None)
        quantization_config.pop("quantization_device", None)
        quantization_config.pop("return_device", None)
        quantization_config.pop("non_blocking", None)
        config = model_cls.load_config(model_path)
        model = model_cls.from_config(config)
        model = apply_sdnq_to_module(model, **quantization_config)

    state_dict = {}
    with safe_open(os.path.join(model_path, file_name), framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    model.load_state_dict(state_dict, assign=True)
    del state_dict
    model = apply_options_to_model(model, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul)
    return model


def apply_options_to_model(model, dtype: torch.dtype = None, dequantize_fp32: bool = None, use_quantized_matmul: bool = False):
    has_children = list(model.children())
    if not has_children:
        return model
    for module in model.children():
        if hasattr(module, "sdnq_dequantizer"):
            if dtype is not None:
                module.sdnq_dequantizer.result_dtype = dtype

            current_scale_dtype = module.svd_up.dtype if module.svd_up is not None else module.scale.dtype
            scale_dtype = torch.float32 if dequantize_fp32 is None and current_scale_dtype == torch.float32 else torch.float32 if dequantize_fp32 else module.sdnq_dequantizer.result_dtype
            upcast_scale = bool(use_quantized_matmul and use_tensorwise_fp8_matmul and module.sdnq_dequantizer.weights_dtype in {"float8_e4m3fn", "float8_e5m2"})

            if upcast_scale:
                module.scale.data = module.scale.to(dtype=torch.float32)
            else:
                module.scale.data = module.scale.to(dtype=scale_dtype)
            if module.zero_point is not None:
                module.zero_point.data = module.zero_point.to(dtype=scale_dtype)
            if module.svd_up is not None:
                module.svd_up.data = module.svd_up.to(dtype=scale_dtype)
                module.svd_down.data = module.svd_down.to(dtype=scale_dtype)

            if use_quantized_matmul != module.sdnq_dequantizer.use_quantized_matmul:
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
