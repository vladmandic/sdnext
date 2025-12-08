# pylint: disable=redefined-builtin,no-member,protected-access

import os
import torch

from modules import shared, devices

sdnq_version = "0.1.2"

dtype_dict = {
    "int32": {"min": -2147483648, "max": 2147483647, "num_bits": 32, "sign": 1, "exponent": 0, "mantissa": 31, "target_dtype": torch.int32, "torch_dtype": torch.int32, "storage_dtype": torch.int32, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int16": {"min": -32768, "max": 32767, "num_bits": 16, "sign": 1, "exponent": 0, "mantissa": 15, "target_dtype": torch.int16, "torch_dtype": torch.int16, "storage_dtype": torch.int16, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int8": {"min": -128, "max": 127, "num_bits": 8, "sign": 1, "exponent": 0, "mantissa": 7, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.int8, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int7": {"min": -64, "max": 63, "num_bits": 7, "sign": 1, "exponent": 0, "mantissa": 6, "target_dtype": "int7", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int6": {"min": -32, "max": 31, "num_bits": 6, "sign": 1, "exponent": 0, "mantissa": 5, "target_dtype": "int6", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int5": {"min": -16, "max": 15, "num_bits": 5, "sign": 1, "exponent": 0, "mantissa": 4, "target_dtype": "int5", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int4": {"min": -8, "max": 7, "num_bits": 4, "sign": 1, "exponent": 0, "mantissa": 3, "target_dtype": "int4", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int3": {"min": -4, "max": 3, "num_bits": 3, "sign": 1, "exponent": 0, "mantissa": 2, "target_dtype": "int3", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int2": {"min": -2, "max": 1, "num_bits": 2, "sign": 1, "exponent": 0, "mantissa": 1, "target_dtype": "int2", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "uint32": {"min": 0, "max": 4294967295, "num_bits": 32, "sign": 0, "exponent": 0, "mantissa": 32, "target_dtype": torch.uint32, "torch_dtype": torch.uint32, "storage_dtype": torch.uint32, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint16": {"min": 0, "max": 65535, "num_bits": 16, "sign": 0, "exponent": 0, "mantissa": 16, "target_dtype": torch.uint16, "torch_dtype": torch.uint16, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint8": {"min": 0, "max": 255, "num_bits": 8, "sign": 0, "exponent": 0, "mantissa": 8, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint7": {"min": 0, "max": 127, "num_bits": 7, "sign": 0, "exponent": 0, "mantissa": 7, "target_dtype": "uint7", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint6": {"min": 0, "max": 63, "num_bits": 6, "sign": 0, "exponent": 0, "mantissa": 6, "target_dtype": "uint6", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint5": {"min": 0, "max": 31, "num_bits": 5, "sign": 0, "exponent": 0, "mantissa": 5, "target_dtype": "uint5", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint4": {"min": 0, "max": 15, "num_bits": 4, "sign": 0, "exponent": 0, "mantissa": 4, "target_dtype": "uint4", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint3": {"min": 0, "max": 7, "num_bits": 3, "sign": 0, "exponent": 0, "mantissa": 3, "target_dtype": "uint3", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint2": {"min": 0, "max": 3, "num_bits": 2, "sign": 0, "exponent": 0, "mantissa": 2, "target_dtype": "uint2", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint1": {"min": 0, "max": 1, "num_bits": 1, "sign": 0, "exponent": 0, "mantissa": 1, "target_dtype": torch.bool, "torch_dtype": torch.bool, "storage_dtype": torch.bool, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "float32": {"min": -3.40282e+38, "max": 3.40282e+38, "num_bits": 32, "sign": 1, "exponent": 8, "mantissa": 23, "target_dtype": torch.float32, "torch_dtype": torch.float32, "storage_dtype": torch.float32, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "bfloat16": {"min": -3.38953e+38, "max": 3.38953e+38, "num_bits": 16, "sign": 1, "exponent": 8, "mantissa": 7, "target_dtype": torch.bfloat16, "torch_dtype": torch.bfloat16, "storage_dtype": torch.bfloat16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float16": {"min": -65504, "max": 65504, "num_bits": 16, "sign": 1, "exponent": 5, "mantissa": 10, "target_dtype": torch.float16, "torch_dtype": torch.float16, "storage_dtype": torch.float16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e4m3fn": {"min": -448, "max": 448, "num_bits": 8, "sign": 1, "exponent": 4, "mantissa": 3, "target_dtype": torch.float8_e4m3fn, "torch_dtype": torch.float8_e4m3fn, "storage_dtype": torch.float8_e4m3fn, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e5m2": {"min": -57344, "max": 57344, "num_bits": 8, "sign": 1, "exponent": 5, "mantissa": 2, "target_dtype": torch.float8_e5m2, "torch_dtype": torch.float8_e5m2, "storage_dtype": torch.float8_e5m2, "is_unsigned": False, "is_integer": False, "is_packed": False},
}

if hasattr(torch, "float8_e4m3fnuz"):
    dtype_dict["float8_e4m3fnuz"] = {"min": -240, "max": 240, "num_bits": 8, "sign": 1, "exponent": 4, "mantissa": 3, "target_dtype": "fp8", "torch_dtype": torch.float8_e4m3fnuz, "storage_dtype": torch.float8_e4m3fnuz, "is_unsigned": False, "is_integer": False, "is_packed": False}
if hasattr(torch, "float8_e5m2fnuz"):
    dtype_dict["float8_e5m2fnuz"] = {"min": -57344, "max": 57344, "num_bits": 8, "sign": 1, "exponent": 5, "mantissa": 2, "target_dtype": "fp8", "torch_dtype": torch.float8_e5m2fnuz, "storage_dtype": torch.float8_e5m2fnuz, "is_unsigned": False, "is_integer": False, "is_packed": False}

dtype_dict["fp32"] = dtype_dict["float32"]
dtype_dict["bf16"] = dtype_dict["bfloat16"]
dtype_dict["fp16"] = dtype_dict["float16"]
dtype_dict["fp8"] = dtype_dict["float8_e4m3fn"]
dtype_dict["bool"] = dtype_dict["uint1"]

linear_types = {"Linear"}
conv_types = {"Conv1d", "Conv2d", "Conv3d"}
conv_transpose_types = {"ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"}
allowed_types = set.union(linear_types, conv_types, conv_transpose_types)
accepted_weight_dtypes = set(dtype_dict.keys())
accepted_matmul_dtypes = {"int8", "fp8", "fp16", "float8_e4m3fnuz", "float16"}

is_rdna2 = bool(devices.backend == "rocm" and int(getattr(torch.cuda.get_device_properties(devices.device), "gcnArchName", "gfx0000")[3:]) < 1100)
use_torch_compile = shared.opts.sdnq_dequantize_compile # this setting requires a full restart of the webui to apply

def check_torch_compile(): # dynamo can be disabled after startup
    return use_torch_compile and not torch._dynamo.config.disable # pylint: disable=protected-access


if os.environ.get("SDNQ_USE_TENSORWISE_FP8_MM", None) is None:
    # row-wise FP8 only exist on H100 hardware, sdnq will use software row-wise with tensorwise hardware with this setting
    use_tensorwise_fp8_matmul = bool(devices.backend != "cuda" or (devices.backend == "cuda" and torch.cuda.get_device_capability(devices.device) < (9,0)))
else:
    use_tensorwise_fp8_matmul = os.environ.get("SDNQ_USE_TENSORWISE_FP8_MM", "0").lower() not in {"0", "false", "no"}

if os.environ.get("SDNQ_USE_CONTIGUOUS_MM", None) is None:
    use_contiguous_mm = bool(is_rdna2 or devices.backend in {"ipex", "mps", "cpu", "openvino", "zluda"})
else:
    use_contiguous_mm = bool(os.environ.get("SDNQ_USE_CONTIGUOUS_MM", "0").lower() not in {"0", "false", "no"})

if os.environ.get("SDNQ_USE_TRITON_MM", None) is None:
    use_triton_mm = bool(is_rdna2 or devices.backend == "zluda")
else:
    use_triton_mm = bool(os.environ.get("SDNQ_USE_TRITON_MM", "0").lower() not in {"0", "false", "no"})


if use_triton_mm:
    try:
        from .triton_mm import int_mm
        int_mm_func = int_mm
    except Exception:
        int_mm_func = torch._int_mm
else:
    int_mm_func = torch._int_mm


def fp_mm_torch(x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
    return torch.mm(x,y, out_dtype=torch.float32)

fp_mm_func = None
if os.environ.get("SDNQ_USE_TRITON_MM", "1").lower() not in {"0", "false", "no"}:
    try:
        from .triton_mm import fp_mm
        fp_mm_func = fp_mm
    except Exception:
        fp_mm_func = None

if fp_mm_func is None:
    fp_mm_func = fp_mm_torch


if use_torch_compile:
    torch._dynamo.config.cache_size_limit = max(8192, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.accumulated_recompile_limit = max(8192, torch._dynamo.config.accumulated_recompile_limit)
    def compile_func(fn, **kwargs):
        if kwargs.get("fullgraph", None) is None:
            kwargs["fullgraph"] = True
        if kwargs.get("dynamic", None) is None:
            kwargs["dynamic"] = False
        return torch.compile(fn, **kwargs)
else:
    def compile_func(fn, **kwargs): # pylint: disable=unused-argument
        return fn


common_skip_keys = (
    ".time_embed",
    ".context_embedder",
    ".condition_embedder",
    ".x_embedder",
    ".t_embedder",
    ".y_embedder",
    ".emb_in",
    ".txt_in",
    ".img_in",
    ".vid_in",
    ".proj_out",
    ".norm_out",
    ".emb_out",
    ".txt_out",
    ".img_out",
    ".vid_out",
    ".final_layer",
    "multi_modal_projector",
    "time_text_embed",
    "patch_embedding",
    "patch_embed",
    "patch_emb",
    "lm_head",
    "wte",
)

module_skip_keys_dict = {
    "FluxTransformer2DModel": [
        ["single_transformer_blocks.0.norm.linear.weight", "time_text_embed", "time_embed", "context_embedder", "x_embedder", ".proj_out", "norm_out"],
        {}
    ],
    "Flux2Transformer2DModel": [
        ["double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation", "time_guidance_embed", "context_embedder", "x_embedder", ".proj_out", "norm_out"],
        {}
    ],
    "ChromaTransformer2DModel": [
        ["distilled_guidance_layer", "time_text_embed", "context_embedder", "x_embedder", ".proj_out", "norm_out"],
        {}
    ],
    "QwenImageTransformer2DModel": [
        ["transformer_blocks.0.img_mod.1.weight", "time_text_embed", "txt_in", "img_in", "proj_out", "norm_out"],
        {}
    ],
    "WanTransformer3DModel": [
        ["scale_shift_table", "patch_embedding", "condition_embedder", "proj_out", "norm_out"],
        {}
    ],
    "LongCatVideoTransformer3DModel": [
        ["blocks.0.adaLN_modulation.1.weight", "x_embedder", "t_embedder", "y_embedder", "final_layer"],
        {}
    ],
    "Lumina2Transformer2DModel": [
        ["layers.0.norm1.linear.weight", "time_caption_embed", "x_embedder", "norm_out"],
        {}
    ],
    "ZImageTransformer2DModel": [
        ["layers.0.adaLN_modulation.0.weight", "t_embedder", "cap_embedder", "all_x_embedder", "all_final_layer"],
        {}
    ],
    "HunyuanImage3ForCausalMM": [
        ["lm_head", "patch_embed", "time_embed", "time_embed_2", "final_layer", "wte", "ln_f", "timestep_emb", "vae", "vision_aligner", "head", "post_layernorm", "embeddings"],
        {}
    ],
    "Emu3ForCausalLM": [
        ["lm_head", "vq_model", "tokenizer"],
        {}
    ],
    "Gemma3nForCausalLM": [
        ["lm_head", "correction_coefs", "prediction_coefs", "embedding_projection"],
        {}
    ],
    "MoondreamModel": [
        ["lm_head", "region", "wte", "post_ln", "proj_mlp", "patch_emb", "pos_emb"],
        {}
    ],
    "NaDiT": [
        [".emb_in", ".txt_in", ".vid_in", ".emb_scale", ".vid_out", ".vid_out_norm", ".vid_out_ada"],
        {}
    ],
}

module_skip_keys_dict["LongCatImageTransformer2DModel"] = module_skip_keys_dict["FluxTransformer2DModel"]
module_skip_keys_dict["ChronoEditTransformer3DModel"] = module_skip_keys_dict["WanTransformer3DModel"]
module_skip_keys_dict["Gemma3nForConditionalGeneration"] = module_skip_keys_dict["Gemma3nForCausalLM"]
module_skip_keys_dict["HfMoondream"] = module_skip_keys_dict["MoondreamModel"]
module_skip_keys_dict["NaDiTUpscaler"] = module_skip_keys_dict["NaDiT"]
