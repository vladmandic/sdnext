from __future__ import annotations
import io
import os
import sys
import time
import contextlib
from enum import Enum
from typing import TYPE_CHECKING

import gradio as gr

from installer import (
    log as log,
    print_dict,
    console as console,
    get_version as get_version,
)

log.debug("Initializing: shared module")

import modules.memmon
import modules.paths as paths
from modules.json_helpers import (
    readfile as readfile,
    writefile as writefile,
)
from modules.shared_helpers import (
    listdir as listdir,
    walk_files as walk_files,
    html_path as html_path,
    html as html,
    req as req,
    total_tqdm as total_tqdm,
)
from modules import errors, devices, shared_state, cmd_args, theme, history, files_cache
from modules.shared_defaults import get_default_modes
from modules.paths import (
    models_path as models_path, # For compatibility, do not modify from here...
    script_path as script_path,
    data_path as data_path,
    sd_configs_path as sd_configs_path,
    sd_default_config as sd_default_config,
    sd_model_file as sd_model_file,
    default_sd_model_file as default_sd_model_file,
    extensions_dir as extensions_dir,
    extensions_builtin_dir as extensions_builtin_dir, # ... to here.
)
from modules.memstats import (
    memory_stats,
    ram_stats as ram_stats,
)

log.debug("Initializing: pipelines")
from modules import shared_items
from modules.interrogate.openclip import caption_models, caption_types, get_clip_models, refresh_clip_models
from modules.interrogate.vqa import vlm_models, vlm_prompts, vlm_system, vlm_default


if TYPE_CHECKING:
    # Behavior modified by __future__.annotations
    from diffusers import DiffusionPipeline
    from modules.shared_legacy import LegacyOption
    from modules.ui_extra_networks import ExtraNetworksPage


class Backend(Enum):
    ORIGINAL = 1
    DIFFUSERS = 2


errors.install([gr])
demo: gr.Blocks | None = None
api = None
url = 'https://github.com/vladmandic/sdnext'
cmd_opts = cmd_args.parse_args()
parser = cmd_args.parser
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
listfiles = listdir
xformers_available = False
compiled_model_state = None
sd_upscalers = []
detailers = []
face_restorers = []
yolo = None
tab_names = []
extra_networks: list[ExtraNetworksPage] = []
hypernetworks = {}
settings_components = {}
restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_control_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}
resize_modes = ["None", "Fixed", "Crop", "Fill", "Outpaint", "Context aware"]
max_workers = 12
sdnq_quant_modes = ["int8", "int7", "int6", "uint5", "uint4", "uint3", "uint2", "float8_e4m3fn", "float7_e3m3fn", "float6_e3m2fn", "float5_e2m2fn", "float4_e2m1fn", "float3_e1m1fn", "float2_e1m0fn"]
sdnq_matmul_modes = ["auto", "int8", "float8_e4m3fn", "float16"]
default_hfcache_dir = os.environ.get("SD_HFCACHEDIR", None) or os.path.join(paths.models_path, 'huggingface')
state = shared_state.State()


# early select backend
backend = Backend.DIFFUSERS
if cmd_opts.use_openvino: # override for openvino
    os.environ.setdefault('PYTORCH_TRACING_MODE', 'TORCHFX')
    from modules.intel.openvino import get_device_list as get_openvino_device_list # pylint: disable=ungrouped-imports
elif cmd_opts.use_ipex or devices.has_xpu():
    from modules.intel.ipex import ipex_init
    ok, e = ipex_init()
    if not ok:
        log.error(f'IPEX initialization failed: {e}')
        if os.environ.get('SD_DEVICE_DEBUG', None) is not None:
            errors.display(e, 'IPEX')
elif cmd_opts.use_directml:
    from modules.dml import directml_init
    ok, e = directml_init()
    if not ok:
        log.error(f'DirectML initialization failed: {e}')
        if os.environ.get('SD_DEVICE_DEBUG', None) is not None:
            errors.display(e, 'DirectML')
elif cmd_opts.use_rocm or devices.has_rocm():
    from modules.rocm import rocm_init
    ok, e = rocm_init()
    if not ok:
        log.error(f'ROCm initialization failed: {e}')
        if os.environ.get('SD_DEVICE_DEBUG', None) is not None:
            errors.display(e, 'ROCm')
devices.backend = devices.get_backend(cmd_opts)
devices.device = devices.get_optimal_device()
mem_stat = memory_stats()
cpu_memory = round(mem_stat['ram']['total'] if "ram" in mem_stat else 0)
gpu_memory = round(mem_stat['gpu']['total'] if "gpu" in mem_stat else 0)
if gpu_memory == 0:
    gpu_memory = cpu_memory
native = backend == Backend.DIFFUSERS
if not files_cache.do_cache_folders:
    log.warning('File cache disabled: ')


def list_checkpoint_titles():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.checkpoint_titles()


list_checkpoint_tiles = list_checkpoint_titles # alias for legacy typo
default_checkpoint = list_checkpoint_titles()[0] if len(list_checkpoint_titles()) > 0 else "model.safetensors"


def is_url(string):
    from urllib.parse import urlparse
    parsed_url = urlparse(string)
    return all([parsed_url.scheme, parsed_url.netloc])


def refresh_checkpoints():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.list_models()


def refresh_vaes():
    import modules.sd_vae # pylint: disable=W0621
    modules.sd_vae.refresh_vae_list()


def refresh_upscalers():
    import modules.modelloader # pylint: disable=W0621
    modules.modelloader.load_upscalers()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    modules.sd_samplers.set_samplers()
    return modules.sd_samplers.all_samplers

log.debug('Initializing: default modes')
startup_offload_mode, startup_offload_min_gpu, startup_offload_max_gpu, startup_cross_attention, startup_sdp_options, startup_sdp_choices, startup_sdp_override_options, startup_sdp_override_choices, startup_offload_always, startup_offload_never = get_default_modes(cmd_opts=cmd_opts, mem_stat=mem_stat)
from modules.dml import memory_providers, default_memory_provider, directml_do_hijack
from modules.onnx_impl import execution_providers

log.debug('Initializing: settings')
from modules.ui_components import DropdownEditable
from modules.options import OptionInfo, options_section
options_templates: dict[str, OptionInfo | LegacyOption] = {}

options_templates.update(options_section(('sd', "Model Loading"), {
    "sd_backend": OptionInfo('diffusers', "Execution backend", gr.Radio, {"choices": ['diffusers', 'original'], "visible": False }),
    "diffusers_pipeline": OptionInfo('Autodetect', 'Model pipeline', gr.Dropdown, lambda: {"choices": list(shared_items.get_pipelines())}),
    "sd_model_checkpoint": OptionInfo(default_checkpoint, "Base model", DropdownEditable, lambda: {"choices": list_checkpoint_titles()}, refresh=refresh_checkpoints),
    "sd_model_refiner": OptionInfo('None', "Refiner model", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_titles()}, refresh=refresh_checkpoints),
    "sd_unet": OptionInfo("Default", "UNET model", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list),
    "latent_history": OptionInfo(16, "Latent history size", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),

    "advanced_sep": OptionInfo("<h2>Advanced Options</h2>", "", gr.HTML),
    "sd_checkpoint_autoload": OptionInfo(True, "Model auto-load on start"),
    "sd_parallel_load": OptionInfo(True, "Model load using multiple threads"),
    "sd_checkpoint_autodownload": OptionInfo(True, "Model auto-download on demand"),
    "stream_load": OptionInfo(False, "Model load using streams", gr.Checkbox),
    "diffusers_to_gpu": OptionInfo(False, "Model load model direct to GPU"),
    "runai_streamer_diffusers": OptionInfo(False, "Diffusers load using Run:ai streamer", gr.Checkbox),
    "runai_streamer_transformers": OptionInfo(False, "Transformers load using Run:ai streamer", gr.Checkbox),
    "diffusers_eval": OptionInfo(False, "Force model eval", gr.Checkbox, {"visible": True }),
    "device_map": OptionInfo('default', "Model load device map", gr.Radio, {"choices": ['default', 'gpu', 'cpu'] }),
    "disable_accelerate": OptionInfo(False, "Disable accelerate", gr.Checkbox, {"visible": False }),
    "sd_checkpoint_cache": OptionInfo(0, "Cached models", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": False }),
}))

options_templates.update(options_section(('model_options', "Model Options"), {
    "model_modular_sep": OptionInfo("<h2>Modular Pipelines</h2>", "", gr.HTML),
    "model_modular_enable": OptionInfo(False, "Enable modular pipelines (experimental)"),
    "model_google_sep": OptionInfo("<h2>Google GenAI</h2>", "", gr.HTML),
    "google_use_vertexai": OptionInfo(False, "Google cloud use VertexAI endpoints"),
    "google_api_key": OptionInfo("", "Google cloud API key", gr.Textbox),
    "google_project_id": OptionInfo("", "Google Cloud project ID", gr.Textbox),
    "google_location_id": OptionInfo("", "Google Cloud location ID", gr.Textbox),
    "model_sd3_sep": OptionInfo("<h2>Stable Diffusion 3.x</h2>", "", gr.HTML),
    "model_sd3_disable_te5": OptionInfo(False, "Disable T5 text encoder"),
    "model_h1_sep": OptionInfo("<h2>HiDream</h2>", "", gr.HTML),
    "model_h1_llama_repo": OptionInfo("Default", "LLama repo", gr.Textbox),
    "model_wan_sep": OptionInfo("<h2>WanAI</h2>", "", gr.HTML),
    "model_wan_stage": OptionInfo("low noise", "Processing stage", gr.Radio, {"choices": ['high noise', 'low noise', 'combined'] }),
    "model_wan_boundary": OptionInfo(0.85, "Stage boundary ratio", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.05 }),
    "model_chrono_sep": OptionInfo("<h2>ChronoEdit</h2>", "", gr.HTML),
    "model_chrono_temporal_steps": OptionInfo(0, "Temporal steps", gr.Slider, {"minimum": 0, "maximum": 50, "step": 1 }),
    "model_qwen_layer_sep": OptionInfo("<h2>Qwen layered</h2>", "", gr.HTML),
    "model_qwen_layers": OptionInfo(2, "Qwen layered number of layers", gr.Slider, {"minimum": 2, "maximum": 9, "step": 1 }),
}))

options_templates.update(options_section(('offload', "Model Offloading"), {
    "offload_sep": OptionInfo("<h2>Model Offloading</h2>", "", gr.HTML),
    "diffusers_offload_mode": OptionInfo(startup_offload_mode, "Model offload mode", gr.Radio, {"choices": ['none', 'balanced', 'group', 'model', 'sequential']}),
    "diffusers_offload_nonblocking": OptionInfo(False, "Non-blocking move operations"),
    "interrogate_offload": OptionInfo(True, "Offload caption models"),
    "offload_balanced_sep": OptionInfo("<h2>Balanced Offload</h2>", "", gr.HTML),
    "diffusers_offload_pre": OptionInfo(True, "Offload during pre-forward"),
    "diffusers_offload_streams": OptionInfo(False, "Offload using streams"),
    "diffusers_offload_min_gpu_memory": OptionInfo(startup_offload_min_gpu, "Offload low watermark", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01 }),
    "diffusers_offload_max_gpu_memory": OptionInfo(startup_offload_max_gpu, "Offload GPU high watermark", gr.Slider, {"minimum": 0.1, "maximum": 1, "step": 0.01 }),
    "diffusers_offload_max_cpu_memory": OptionInfo(0.90, "Offload CPU high watermark", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": False }),
    "models_not_to_offload": OptionInfo("", "Model types not to offload"),
    "diffusers_offload_always": OptionInfo(startup_offload_always, "Modules to always offload"),
    "diffusers_offload_never": OptionInfo(startup_offload_never, "Modules to never offload"),
    "offload_group_sep": OptionInfo("<h2>Group Offload</h2>", "", gr.HTML),
    "group_offload_type": OptionInfo("leaf_level", "Group offload type", gr.Radio, {"choices": ['leaf_level', 'block_level']}),
    "group_offload_stream": OptionInfo(False, "Use torch streams", gr.Checkbox),
    'group_offload_record': OptionInfo(False, "Record torch streams", gr.Checkbox),
    'group_offload_blocks': OptionInfo(1, "Offload blocks", gr.Number),
}))

options_templates.update(options_section(("quantization", "Model Quantization"), {
    "models_not_to_quant": OptionInfo("", "Model types not to quantize"),

    "sdnq_quantize_sep": OptionInfo("<h2>SDNQ: SD.Next Quantization</h2>", "", gr.HTML),
    "sdnq_quantize_weights": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "LLM", "Control", "VAE"]}),
    "sdnq_quantize_mode": OptionInfo("auto", "Quantization mode", gr.Dropdown, {"choices": ["auto", "pre", "post"]}),
    "sdnq_quantize_weights_mode": OptionInfo("int8", "Quantization type", gr.Dropdown, {"choices": sdnq_quant_modes}),
    "sdnq_quantize_matmul_mode": OptionInfo("auto", "Quantized MatMul type", gr.Dropdown, {"choices": sdnq_matmul_modes}),
    "sdnq_quantize_weights_mode_te": OptionInfo("Same as model", "Quantization type for Text Encoders", gr.Dropdown, {"choices": ['Same as model'] + sdnq_quant_modes}),
    "sdnq_quantize_matmul_mode_te": OptionInfo("Same as model", "Quantized MatMul type for Text Encoders", gr.Dropdown, {"choices": ['Same as model'] + sdnq_matmul_modes}),
    "sdnq_modules_to_not_convert": OptionInfo("", "Modules to not convert"),
    "sdnq_modules_dtype_dict": OptionInfo("{}", "Modules dtype dict"),
    "sdnq_quantize_weights_group_size": OptionInfo(0, "Group size", gr.Slider, {"minimum": -1, "maximum": 4096, "step": 1}),
    "sdnq_svd_rank": OptionInfo(32, "SVD rank size", gr.Slider, {"minimum": 1, "maximum": 512, "step": 1}),
    "sdnq_svd_steps": OptionInfo(8, "SVD steps", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "sdnq_dynamic_loss_threshold": OptionInfo(1e-2, "Dynamic loss threshold", gr.Slider, {"minimum": 1e-4, "maximum": 1e-1, "step": 1e-4}),
    "sdnq_use_svd": OptionInfo(False, "Use SVD quantization", gr.Checkbox),
    "sdnq_use_dynamic_quantization": OptionInfo(False, "Use Dynamic quantization", gr.Checkbox),
    "sdnq_quantize_conv_layers": OptionInfo(False, "Quantize convolutional layers", gr.Checkbox),
    "sdnq_dequantize_compile": OptionInfo(devices.has_triton(early=True), "Dequantize using torch.compile", gr.Checkbox),
    "sdnq_use_quantized_matmul": OptionInfo(False, "Use quantized MatMul", gr.Checkbox),
    "sdnq_use_quantized_matmul_conv": OptionInfo(False, "Use quantized MatMul with conv", gr.Checkbox),
    "sdnq_quantize_with_gpu": OptionInfo(True, "Quantize using GPU", gr.Checkbox),
    "sdnq_dequantize_fp32": OptionInfo(False, "Dequantize using full precision", gr.Checkbox),
    "sdnq_quantize_shuffle_weights": OptionInfo(False, "Shuffle weights in post mode", gr.Checkbox),

    "nunchaku_sep": OptionInfo("<h2>Nunchaku Engine</h2>", "", gr.HTML),
    "nunchaku_attention": OptionInfo(False, "Nunchaku attention", gr.Checkbox),
    "nunchaku_offload": OptionInfo(False, "Nunchaku offloading", gr.Checkbox),

    "bnb_quantization_sep": OptionInfo("<h2>BitsAndBytes</h2>", "", gr.HTML),
    "bnb_quantization": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "LLM", "VAE"]}),
    "bnb_quantization_type": OptionInfo("nf4", "Quantization type", gr.Dropdown, {"choices": ["nf4", "fp8", "fp4"]}),
    "bnb_quantization_storage": OptionInfo("uint8", "Backend storage", gr.Dropdown, {"choices": ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]}),

    "quanto_quantization_sep": OptionInfo("<h2>Optimum Quanto</h2>", "", gr.HTML),
    "quanto_quantization": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "LLM"]}),
    "quanto_quantization_type": OptionInfo("int8", "Quantization weights type", gr.Dropdown, {"choices": ["float8", "int8", "int4", "int2"]}),

    "optimum_quanto_sep": OptionInfo("<h2>Optimum Quanto: post-load</h2>", "", gr.HTML),
    "optimum_quanto_weights": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "Control", "VAE"]}),
    "optimum_quanto_weights_type": OptionInfo("qint8", "Quantization weights type", gr.Dropdown, {"choices": ["qint8", "qfloat8_e4m3fn", "qfloat8_e5m2", "qint4", "qint2"]}),
    "optimum_quanto_activations_type": OptionInfo("none", "Quantization activations type ", gr.Dropdown, {"choices": ["none", "qint8", "qfloat8_e4m3fn", "qfloat8_e5m2"]}),
    "optimum_quanto_shuffle_weights": OptionInfo(False, "Shuffle weights in post mode", gr.Checkbox),

    "torchao_sep": OptionInfo("<h2>TorchAO</h2>", "", gr.HTML),
    "torchao_quantization": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "LLM", "Control", "VAE"]}),
    "torchao_quantization_mode": OptionInfo("auto", "Quantization mode", gr.Dropdown, {"choices": ["auto", "pre", "post"]}),
    "torchao_quantization_type": OptionInfo("int8_weight_only", "Quantization type", gr.Dropdown, {"choices": ["int4_weight_only", "int8_dynamic_activation_int4_weight", "int8_weight_only", "int8_dynamic_activation_int8_weight", "float8_weight_only", "float8_dynamic_activation_float8_weight", "float8_static_activation_float8_weight"]}),

    "layerwise_quantization_sep": OptionInfo("<h2>Layerwise Casting</h2>", "", gr.HTML),
    "layerwise_quantization": OptionInfo([], "Layerwise casting enabled", gr.CheckboxGroup, {"choices": ["Model", "TE"]}),
    "layerwise_quantization_storage": OptionInfo("float8_e4m3fn", "Layerwise casting storage", gr.Dropdown, {"choices": ["float8_e4m3fn", "float8_e5m2"]}),
    "layerwise_quantization_nonblocking": OptionInfo(False, "Layerwise non-blocking operations", gr.Checkbox),

    "trt_quantization_sep": OptionInfo("<h2>TensorRT</h2>", "", gr.HTML),
    "trt_quantization": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model"]}),
    "trt_quantization_type": OptionInfo("int8", "Quantization type", gr.Dropdown, {"choices": ["int8", "int4", "fp8", "nf4", "nvfp4"]}),

    "nncf_compress_sep": OptionInfo("<h2>NNCF: Neural Network Compression Framework</h2>", "", gr.HTML, {"visible": cmd_opts.use_openvino}),
    "nncf_compress_weights": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "VAE"], "visible": cmd_opts.use_openvino}),
    "nncf_compress_weights_mode": OptionInfo("INT8_SYM", "Quantization type", gr.Dropdown, {"choices": ["INT8", "INT8_SYM", "FP8", "MXFP8", "INT4_ASYM", "INT4_SYM", "FP4", "MXFP4", "NF4"], "visible": cmd_opts.use_openvino}),
    "nncf_compress_weights_raito": OptionInfo(0, "Compress ratio", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": cmd_opts.use_openvino}),
    "nncf_compress_weights_group_size": OptionInfo(0, "Group size", gr.Slider, {"minimum": -1, "maximum": 4096, "step": 1, "visible": cmd_opts.use_openvino}),
    "nncf_quantize": OptionInfo([], "Static Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "TE", "VAE"], "visible": cmd_opts.use_openvino}),
    "nncf_quantize_mode": OptionInfo("INT8", "OpenVINO activations mode", gr.Dropdown, {"choices": ["INT8", "FP8_E4M3", "FP8_E5M2"], "visible": cmd_opts.use_openvino}),
}))

options_templates.update(options_section(('vae_encoder', "Variational Auto Encoder"), {
    "sd_vae": OptionInfo("Automatic", "VAE model", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "diffusers_vae_upcast": OptionInfo("default", "VAE upcasting", gr.Radio, {"choices": ['default', 'true', 'false']}),
    "no_half_vae": OptionInfo(False if not cmd_opts.use_openvino else True, "Full precision (--no-half-vae)"),
    "diffusers_vae_slicing": OptionInfo(True, "VAE slicing", gr.Checkbox),
    "diffusers_vae_tiling": OptionInfo(cmd_opts.lowvram, "VAE tiling", gr.Checkbox),
    "diffusers_vae_tile_size": OptionInfo(0, "VAE tile size", gr.Slider, {"minimum": 0, "maximum": 4096, "step": 8 }),
    "diffusers_vae_tile_overlap": OptionInfo(0.25, "VAE tile overlap", gr.Slider, {"minimum": 0, "maximum": 0.95, "step": 0.05 }),
    "remote_vae_type": OptionInfo('raw', "Remote VAE image type", gr.Dropdown, {"choices": ['raw', 'jpg', 'png']}),
    "remote_vae_encode": OptionInfo(False, "Remote VAE for encode"),
}))

options_templates.update(options_section(('text_encoder', "Text Encoder"), {
    "sd_text_encoder": OptionInfo('Default', "Text encoder model", DropdownEditable, lambda: {"choices": shared_items.sd_te_items()}, refresh=shared_items.refresh_te_list),
    "prompt_attention": OptionInfo("native", "Prompt attention parser", gr.Radio, {"choices": ["native", "compel", "xhinker", "a1111", "fixed"] }),
    "prompt_mean_norm": OptionInfo(False, "Prompt attention normalization", gr.Checkbox),
    "sd_textencoder_cache_size": OptionInfo(4, "Text encoder cache size", gr.Slider, {"minimum": 0, "maximum": 16, "step": 1}),
    "sd_textencder_linebreak": OptionInfo(True, "Use line break as prompt segment marker", gr.Checkbox),
    "diffusers_zeros_prompt_pad": OptionInfo(False, "Use zeros for prompt padding", gr.Checkbox),
    "te_optional_sep": OptionInfo("<h2>Optional</h2>", "", gr.HTML),
    "te_shared_t5": OptionInfo(True, "T5: Use shared instance of text encoder"),
    "te_pooled_embeds": OptionInfo(False, "SDXL: Use weighted pooled embeds"),
    "te_complex_human_instruction": OptionInfo(True, "Sana: Use complex human instructions"),
    "te_use_mask": OptionInfo(True, "Lumina: Use mask in transformers"),
}))

options_templates.update(options_section(('cuda', "Compute Settings"), {
    "math_sep": OptionInfo("<h2>Execution Precision</h2>", "", gr.HTML),
    "precision": OptionInfo("Autocast", "Precision type", gr.Radio, {"choices": ["Autocast", "Full"], "visible": False}),
    "cuda_dtype": OptionInfo("Auto", "Device precision type", gr.Radio, {"choices": ["Auto", "FP32", "FP16", "BF16"]}),
    "no_half": OptionInfo(False if not cmd_opts.use_openvino else True, "Force full precision (--no-half)", None, None, None),
    "upcast_sampling": OptionInfo(False if sys.platform != "darwin" else True, "Upcast sampling", gr.Checkbox, {"visible": False}),

    "generator_sep": OptionInfo("<h2>Noise Options</h2>", "", gr.HTML),
    "diffusers_generator_device": OptionInfo("GPU", "Generator device", gr.Radio, {"choices": ["GPU", "CPU", "Unset"]}),

    "cross_attention_sep": OptionInfo("<h2>Cross Attention</h2>", "", gr.HTML),
    "cross_attention_optimization": OptionInfo(startup_cross_attention, "Attention method", gr.Radio, lambda: {"choices": shared_items.list_crossattention()}),
    "sdp_options": OptionInfo(startup_sdp_options, "SDP kernels", gr.CheckboxGroup, {"choices": startup_sdp_choices}),
    "sdp_overrides": OptionInfo(startup_sdp_override_options, "SDP overrides", gr.CheckboxGroup, {"choices": startup_sdp_override_choices}),
    "attention_slicing": OptionInfo('Default', "Attention slicing", gr.Radio, {"choices": ['Default', 'Enabled', 'Disabled']}),
    "xformers_options": OptionInfo(['Flash attention'], "xFormers options", gr.CheckboxGroup, {"choices": ['Flash attention'] }),
    "dynamic_attention_slice_rate": OptionInfo(0.5, "Dynamic Attention slicing rate", gr.Slider, {"minimum": 0.01, "maximum": max(gpu_memory,4), "step": 0.01}),
    "dynamic_attention_trigger_rate": OptionInfo(1, "Dynamic Attention trigger rate", gr.Slider, {"minimum": 0.01, "maximum": max(gpu_memory,4)*2, "step": 0.01}),
}))

options_templates.update(options_section(('backends', "Backend Settings"), {
    "other_sep": OptionInfo("<h2>Torch Options</h2>", "", gr.HTML),
    "opt_channelslast": OptionInfo(False, "Channels last "),
    "cudnn_deterministic": OptionInfo(False, "Deterministic mode"),
    "diffusers_fuse_projections": OptionInfo(False, "Fused projections"),
    "torch_expandable_segments": OptionInfo(False, "Expandable segments"),
    "cudnn_enabled": OptionInfo("default", "cuDNN enabled", gr.Radio, {"choices": ["default", "true", "false"]}),
    "cudnn_benchmark": OptionInfo(devices.backend != "rocm", "cuDNN full-depth benchmark"),
    "cudnn_benchmark_limit": OptionInfo(10, "cuDNN benchmark limit", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),
    "torch_tunable_ops": OptionInfo("default", "Tunable ops", gr.Radio, {"choices": ["default", "true", "false"]}),
    "torch_tunable_limit": OptionInfo(30, "Tunable ops limit", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "cuda_mem_fraction": OptionInfo(0.0, "Memory limit", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.05}),
    "torch_gc_threshold": OptionInfo(70, "GC threshold", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "inference_mode": OptionInfo("no-grad", "Inference mode", gr.Radio, {"choices": ["no-grad", "inference-mode", "none"]}),
    "torch_malloc": OptionInfo("native", "Memory allocator", gr.Radio, {"choices": ['native', 'cudaMallocAsync'] }),

    "onnx_sep": OptionInfo("<h2>ONNX</h2>", "", gr.HTML),
    "onnx_execution_provider": OptionInfo(execution_providers.get_default_execution_provider().value, 'ONNX Execution Provider', gr.Dropdown, lambda: {"choices": execution_providers.available_execution_providers }),
    "onnx_cpu_fallback": OptionInfo(True, 'ONNX allow fallback to CPU'),
    "onnx_cache_converted": OptionInfo(True, 'ONNX cache converted models'),
    "onnx_unload_base": OptionInfo(False, 'ONNX unload base model when processing refiner'),

    "olive_sep": OptionInfo("<h2>Olive</h2>", "", gr.HTML),
    "olive_float16": OptionInfo(True, 'Olive use FP16 on optimization'),
    "olive_vae_encoder_float32": OptionInfo(False, 'Olive force FP32 for VAE Encoder'),
    "olive_static_dims": OptionInfo(True, 'Olive use static dimensions'),
    "olive_cache_optimized": OptionInfo(True, 'Olive cache optimized models'),

    "ipex_sep": OptionInfo("<h2>IPEX</h2>", "", gr.HTML, {"visible": devices.backend == "ipex"}),
    "ipex_optimize": OptionInfo([], "IPEX Optimize", gr.CheckboxGroup, {"choices": ["Model", "TE", "VAE", "Upscaler"], "visible": devices.backend == "ipex"}),

    "openvino_sep": OptionInfo("<h2>OpenVINO</h2>", "", gr.HTML, {"visible": cmd_opts.use_openvino}),
    "openvino_devices": OptionInfo([], "OpenVINO devices to use", gr.CheckboxGroup, {"choices": get_openvino_device_list() if cmd_opts.use_openvino else [], "visible": cmd_opts.use_openvino}), # pylint: disable=E0606
    "openvino_accuracy": OptionInfo("performance", "OpenVINO accuracy mode", gr.Radio, {"choices": ["performance", "accuracy"], "visible": cmd_opts.use_openvino}),
    "openvino_disable_model_caching": OptionInfo(True, "OpenVINO disable model caching", gr.Checkbox, {"visible": cmd_opts.use_openvino}),
    "openvino_disable_memory_cleanup": OptionInfo(True, "OpenVINO disable memory cleanup after compile", gr.Checkbox, {"visible": cmd_opts.use_openvino}),

    "directml_sep": OptionInfo("<h2>DirectML</h2>", "", gr.HTML, {"visible": devices.backend == "directml"}),
    "directml_memory_provider": OptionInfo(default_memory_provider, "DirectML memory stats provider", gr.Radio, {"choices": memory_providers, "visible": devices.backend == "directml"}),
    "directml_catch_nan": OptionInfo(False, "DirectML retry ops for NaN", gr.Checkbox, {"visible": devices.backend == "directml"}),
}))

options_templates.update(options_section(('advanced', "Pipeline Modifiers"), {
    "clip_skip_sep": OptionInfo("<h2>CLiP Skip</h2>", "", gr.HTML),
    "clip_skip_enabled": OptionInfo(False, "CLiP skip enabled"),

    "token_merging_sep": OptionInfo("<h2>Token Merging</h2>", "", gr.HTML),
    "token_merging_method": OptionInfo("None", "Token merging enabled", gr.Radio, {"choices": ['None', 'ToMe', 'ToDo']}),
    "tome_ratio": OptionInfo(0.0, "ToMe token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}),
    "todo_ratio": OptionInfo(0.0, "ToDo token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}),

    "freeu_sep": OptionInfo("<h2>FreeU</h2>", "", gr.HTML),
    "freeu_enabled": OptionInfo(False, "FreeU enabled"),
    "freeu_b1": OptionInfo(1.2, "FreeU 1st stage backbone", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_b2": OptionInfo(1.4, "FreeU 2nd stage backbone", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_s1": OptionInfo(0.9, "FreeU 1st stage skip", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "freeu_s2": OptionInfo(0.2, "FreeU 2nd stage skip", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "pag_sep": OptionInfo("<h2>PAG: Perturbed attention guidance</h2>", "", gr.HTML),
    "pag_apply_layers": OptionInfo("m0", "PAG layer names"),

    "pab_sep": OptionInfo("<h2>PAB: Pyramid attention broadcast </h2>", "", gr.HTML),
    "pab_enabled": OptionInfo(False, "PAB cache enabled"),
    "pab_spacial_skip_range": OptionInfo(2, "PAB spacial skip range", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}),
    "pab_spacial_skip_start": OptionInfo(100, "PAB spacial skip start", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}),
    "pab_spacial_skip_end": OptionInfo(800, "PAB spacial skip end", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}),

    "cache_dit_sep": OptionInfo("<h2>Cache-DiT</h2>", "", gr.HTML),
    "cache_dit_enabled": OptionInfo(False, "Cache-DiT enabled"),
    "cache_dit_calibrator": OptionInfo("None", "Cache-DiT calibrator", gr.Radio, {"choices": ["None", "TaylorSeer", "FoCa"]}),
    "cache_dit_fcompute": OptionInfo(-1, "Cache-DiT F-compute blocks", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "cache_dit_bcompute": OptionInfo(-1, "Cache-DiT B-compute blocks", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "cache_dit_threshold": OptionInfo(-1, "Cache-DiT residual diff threshold", gr.Slider, {"minimum": -1.0, "maximum": 1.0, "step": 0.01}),
    "cache_dit_warmup": OptionInfo(-1, "Cache-DiT warmup steps", gr.Slider, {"minimum": -1, "maximum": 50, "step": 1}),

    "faster_cache__sep": OptionInfo("<h2>Faster Cache</h2>", "", gr.HTML),
    "faster_cache_enabled": OptionInfo(False, "FasterCache cache enabled"),
    "fc_spacial_skip_range": OptionInfo(2, "FasterCache spacial skip range", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}),
    "fc_spacial_skip_start": OptionInfo(0, "FasterCache spacial skip start", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}),
    "fc_spacial_skip_end": OptionInfo(681, "FasterCache spacial skip end", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.01}),
    "fc_uncond_skip_range": OptionInfo(5, "FasterCache uncond skip range", gr.Slider, {"minimum": 1, "maximum": 4, "step": 1}),
    "fc_uncond_skip_start": OptionInfo(0, "FasterCache uncond skip start", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "fc_uncond_skip_end": OptionInfo(781, "FasterCache uncond skip end", gr.Slider, {"minimum": 0, "maximum": 1, "step": 1}),
    "fc_attention_weight": OptionInfo(0.5, "FasterCache spacial skip range", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}),
    "fc_tensor_format": OptionInfo("BCFHW", "FasterCache tensor format", gr.Radio, {"choices": ["BCFHW", "BFCHW", "BCHW"]}),
    "fc_guidance_distilled": OptionInfo(False, "FasterCache guidance distilled", gr.Checkbox),

    "para_sep": OptionInfo("<h2>Para-attention</h2>", "", gr.HTML),
    "para_cache_enabled": OptionInfo(False, "ParaAttention first-block cache enabled"),
    "para_diff_threshold": OptionInfo(0.1, "ParaAttention residual diff threshold", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "teacache_sep": OptionInfo("<h2>TeaCache</h2>", "", gr.HTML),
    "teacache_enabled": OptionInfo(False, "TeaCache cache enabled"),
    "teacache_thresh": OptionInfo(0.15, "TeaCache L1 threshold", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "hypertile_sep": OptionInfo("<h2>HyperTile</h2>", "", gr.HTML),
    "hypertile_unet_enabled": OptionInfo(False, "Hypertile UNet Enabled"),
    "hypertile_hires_only": OptionInfo(False, "Hypertile HiRes pass only"),
    "hypertile_unet_tile": OptionInfo(0, "Hypertile UNet max tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_unet_min_tile": OptionInfo(0, "Hypertile UNet min tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_unet_swap_size": OptionInfo(1, "Hypertile UNet swap size", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "hypertile_unet_depth": OptionInfo(0, "Hypertile UNet depth", gr.Slider, {"minimum": 0, "maximum": 4, "step": 1}),
    "hypertile_vae_enabled": OptionInfo(False, "Hypertile VAE Enabled", gr.Checkbox),
    "hypertile_vae_tile": OptionInfo(128, "Hypertile VAE tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_vae_swap_size": OptionInfo(1, "Hypertile VAE swap size", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),

    "hidiffusion_sep": OptionInfo("<h2>HiDiffusion</h2>", "", gr.HTML),
    "hidiffusion_raunet": OptionInfo(True, "HiDiffusion apply RAU-Net"),
    "hidiffusion_attn": OptionInfo(True, "HiDiffusion apply MSW-MSA"),
    "hidiffusion_steps": OptionInfo(8, "HiDiffusion aggressive at step", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "hidiffusion_t1": OptionInfo(-1, "HiDiffusion override T1 ratio", gr.Slider, {"minimum": -1, "maximum": 1.0, "step": 0.05}),
    "hidiffusion_t2": OptionInfo(-1, "HiDiffusion override T2 ratio", gr.Slider, {"minimum": -1, "maximum": 1.0, "step": 0.05}),

    "linfusion_sep": OptionInfo("<h2>LinFusion</h2>", "", gr.HTML),
    "enable_linfusion": OptionInfo(False, "LinFusion apply distillation on load"),

    "ras_sep": OptionInfo("<h2>RAS: Region-Adaptive Sampling</h2>", "", gr.HTML),
    "ras_enable": OptionInfo(False, "RAS enabled"),

    "cfgzero_sep": OptionInfo("<h2>CFG-Zero</h2>", "", gr.HTML),
    "cfgzero_enabled": OptionInfo(False, "CFG-Zero enabled"),
    "cfgzero_star": OptionInfo(False, "CFG-Zero star"),
    "cfgzero_steps": OptionInfo(0, "CFG-Zero steps", gr.Slider, {"minimum": 0, "maximum": 3, "step": 1}),

    "inference_batch_sep": OptionInfo("<h2>Batch</h2>", "", gr.HTML),
    "sequential_seed": OptionInfo(True, "Batch mode uses sequential seeds"),
    "batch_frame_mode": OptionInfo(False, "Parallel process images in batch"),
}))

options_templates.update(options_section(('compile', "Model Compile"), {
    "cuda_compile_sep": OptionInfo("<h2>Model Compile</h2>", "", gr.HTML),
    "cuda_compile": OptionInfo([] if not cmd_opts.use_openvino else ["Model", "VAE", "Upscaler", "Control"], "Compile Model", gr.CheckboxGroup, {"choices": ["Model", "TE", "VAE", "LLM", "Control", "Upscaler"]}),
    "cuda_compile_backend": OptionInfo("inductor" if not cmd_opts.use_openvino else "openvino_fx", "Model compile backend", gr.Radio, {"choices": ['none', 'inductor', 'cudagraphs', 'aot_ts_nvfuser', 'hidet', 'migraphx', 'ipex', 'onediff', 'stable-fast', 'deep-cache', 'olive-ai', 'openvino_fx']}),
    "cuda_compile_mode": OptionInfo("default", "Model compile mode", gr.Radio, {"choices": ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']}),
    "cuda_compile_options": OptionInfo(["repeated", "fullgraph", "dynamic"] if not cmd_opts.use_openvino else [], "Model compile options", gr.CheckboxGroup, {"choices": ["precompile", "repeated", "fullgraph", "dynamic", "verbose"]}),
    "deep_cache_interval": OptionInfo(3, "DeepCache cache interval", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
}))

options_templates.update(options_section(('system-paths', "System Paths"), {
    "models_paths_sep_options": OptionInfo("<h2>Models Paths</h2>", "", gr.HTML),
    "models_dir": OptionInfo('models', "Root model folder", folder=True),
    "model_paths_sep_options": OptionInfo("<h2>Paths for specific models</h2>", "", gr.HTML),
    "ckpt_dir": OptionInfo(os.path.join(paths.models_path, 'Stable-diffusion'), "Folder with stable diffusion models", folder=True),
    "diffusers_dir": OptionInfo(os.path.join(paths.models_path, 'Diffusers'), "Folder with Huggingface models", folder=True),
    "hfcache_dir": OptionInfo(default_hfcache_dir, "Folder for Huggingface cache", folder=True),
    "tunable_dir": OptionInfo(os.path.join(paths.models_path, 'tunable'), "Folder for Tunable ops cache", folder=True),
    "vae_dir": OptionInfo(os.path.join(paths.models_path, 'VAE'), "Folder with VAE files", folder=True),
    "unet_dir": OptionInfo(os.path.join(paths.models_path, 'UNET'), "Folder with UNET files", folder=True),
    "te_dir": OptionInfo(os.path.join(paths.models_path, 'Text-encoder'), "Folder with Text encoder files", folder=True),
    "lora_dir": OptionInfo(os.path.join(paths.models_path, 'Lora'), "Folder with LoRA network(s)", folder=True),
    "styles_dir": OptionInfo(os.path.join(paths.models_path, 'styles'), "File or Folder with user-defined styles", folder=True),
    "wildcards_dir": OptionInfo(os.path.join(paths.models_path, 'wildcards'), "Folder with user-defined wildcards", folder=True),
    "embeddings_dir": OptionInfo(os.path.join(paths.models_path, 'embeddings'), "Folder with textual inversion embeddings", folder=True),
    "control_dir": OptionInfo(os.path.join(paths.models_path, 'control'), "Folder with Control models", folder=True),
    "yolo_dir": OptionInfo(os.path.join(paths.models_path, 'yolo'), "Folder with Yolo models", folder=True),
    "codeformer_models_path": OptionInfo(os.path.join(paths.models_path, 'Codeformer'), "Folder with codeformer models", folder=True),
    "gfpgan_models_path": OptionInfo(os.path.join(paths.models_path, 'GFPGAN'), "Folder with GFPGAN models", folder=True),
    "esrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'ESRGAN'), "Folder with ESRGAN models", folder=True),
    "bsrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'BSRGAN'), "Folder with BSRGAN models", folder=True),
    "realesrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'RealESRGAN'), "Folder with RealESRGAN models", folder=True),
    "scunet_models_path": OptionInfo(os.path.join(paths.models_path, 'SCUNet'), "Folder with SCUNet models", folder=True),
    "swinir_models_path": OptionInfo(os.path.join(paths.models_path, 'SwinIR'), "Folder with SwinIR models", folder=True),
    "clip_models_path": OptionInfo(os.path.join(paths.models_path, 'CLIP'), "Folder with CLIP models", folder=True),
    "other_paths_sep_options": OptionInfo("<h2>Cache folders</h2>", "", gr.HTML),
    "clean_temp_dir_at_start": OptionInfo(True, "Cleanup temporary folder on startup"),
    "temp_dir": OptionInfo("", "Directory for temporary images; leave empty for default", folder=True),
    "accelerate_offload_path": OptionInfo('cache/accelerate', "Folder for disk offload", folder=True),
    "openvino_cache_path": OptionInfo('cache', "Folder for OpenVINO cache", folder=True),
    "onnx_cached_models_path": OptionInfo(os.path.join(paths.models_path, 'ONNX', 'cache'), "Folder for ONNX cached models", folder=True),
    "onnx_temp_dir": OptionInfo(os.path.join(paths.models_path, 'ONNX', 'temp'), "Folder for ONNX conversion", folder=True),
}))

options_templates.update(options_section(('saving-images', "Image Options"), {
    "samples_save": OptionInfo(True, "Save all generated images"),
    "keep_incomplete": OptionInfo(True, "Save interrupted images"),
    "samples_format": OptionInfo('jpg', 'File format', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2", "jxl"]}),
    "jpeg_quality": OptionInfo(90, "Image quality", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "img_max_size_mp": OptionInfo(1000, "Maximum image size (MP)", gr.Slider, {"minimum": 100, "maximum": 2000, "step": 1}),
    "webp_lossless": OptionInfo(False, "WebP lossless compression"),
    "save_selected_only": OptionInfo(True, "UI save only saves selected image"),
    "include_mask": OptionInfo(False, "Include mask in outputs"),
    "samples_save_zip": OptionInfo(False, "Create ZIP archive for multiple images"),
    "image_background": OptionInfo("#000000", "Resize background color", gr.ColorPicker, {}),

    "image_sep_grid": OptionInfo("<h2>Grid Options</h2>", "", gr.HTML),
    "grid_save": OptionInfo(True, "Save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2", "jxl"]}),
    "n_rows": OptionInfo(-1, "Grid max rows count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "n_cols": OptionInfo(-1, "Grid max columns count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "grid_background": OptionInfo("#000000", "Grid background color", gr.ColorPicker, {}),
    "font": OptionInfo("", "Font file"),
    "font_color": OptionInfo("#FFFFFF", "Font color", gr.ColorPicker, {}),

    "image_sep_browser": OptionInfo("<h2>Image Gallery</h2>", "", gr.HTML),
    "browser_cache": OptionInfo(True, "Use image gallery cache"),
    "browser_folders": OptionInfo("", "Additional image browser folders"),
    "browser_gallery_autoupdate": OptionInfo(False, "Automatically update when switching to the gallery"),
    "browser_fixed_width": OptionInfo(False, "Use fixed width thumbnails"),
    "viewer_show_metadata": OptionInfo(True, "Show metadata in full screen image browser"),

    "save_sep_options": OptionInfo("<h2>Intermediate Image Saving</h2>", "", gr.HTML),
    "save_init_img": OptionInfo(False, "Save init images"),
    "save_images_before_highres_fix": OptionInfo(False, "Save image before hires"),
    "save_images_before_refiner": OptionInfo(False, "Save image before refiner"),
    "save_images_before_detailer": OptionInfo(False, "Save image before detailer"),
    "save_images_before_color_correction": OptionInfo(False, "Save image before color correction"),
    "save_mask": OptionInfo(False, "Save inpainting mask"),
    "save_mask_composite": OptionInfo(False, "Save inpainting masked composite"),
    "gradio_skip_video": OptionInfo(False, "Do not display video output in UI"),

    "image_sep_watermark": OptionInfo("<h2>Watermarking</h2>", "", gr.HTML),
    "image_watermark_enabled": OptionInfo(False, "Include invisible watermark"),
    "image_watermark": OptionInfo('', "Invisible watermark string"),
    "image_watermark_position": OptionInfo('none', 'Image watermark position', gr.Dropdown, {"choices": ["none", "top/left", "top/right", "bottom/left", "bottom/right", "center", "random"]}),
    "image_watermark_image": OptionInfo('', "Image watermark file"),
}))

options_templates.update(options_section(('saving-paths', "Image Paths"), {
    "saving_sep_images": OptionInfo("<h2>Save Options</h2>", "", gr.HTML),
    "save_images_add_number": OptionInfo(True, "Numbered filenames", component_args=hide_dirs),
    "use_original_name_batch": OptionInfo(True, "Batch uses original name"),
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),
    "samples_filename_pattern": OptionInfo("[seq]-[date]-[model_name]", "Images filename pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max words", gr.Slider, {"minimum": 1, "maximum": 99, "step": 1, **hide_dirs}),

    "outdir_sep_dirs": OptionInfo("<h2>Folders</h2>", "", gr.HTML),
    "outdir_samples": OptionInfo("", "Base images folder", component_args=hide_dirs, folder=True),
    "outdir_txt2img_samples": OptionInfo("outputs/text", 'Folder for text generate', component_args=hide_dirs, folder=True),
    "outdir_img2img_samples": OptionInfo("outputs/image", 'Folder for image generate', component_args=hide_dirs, folder=True),
    "outdir_control_samples": OptionInfo("outputs/control", 'Folder for control generate', component_args=hide_dirs, folder=True),
    "outdir_extras_samples": OptionInfo("outputs/extras", 'Folder for processed images', component_args=hide_dirs, folder=True),
    "outdir_save": OptionInfo("outputs/save", "Folder for manually saved images", component_args=hide_dirs, folder=True),
    "outdir_video": OptionInfo("outputs/video", "Folder for videos", component_args=hide_dirs, folder=True),
    "outdir_init_images": OptionInfo("outputs/inputs", "Folder for init images", component_args=hide_dirs, folder=True),

    "outdir_sep_grids": OptionInfo("<h2>Grids</h2>", "", gr.HTML),
    "outdir_grids": OptionInfo("", "Base grids folder", component_args=hide_dirs, folder=True),
    "outdir_txt2img_grids": OptionInfo("outputs/grids", 'Folder for txt2img grids', component_args=hide_dirs, folder=True),
    "outdir_img2img_grids": OptionInfo("outputs/grids", 'Folder for img2img grids', component_args=hide_dirs, folder=True),
    "outdir_control_grids": OptionInfo("outputs/grids", 'Folder for control grids', component_args=hide_dirs, folder=True),
}))

options_templates.update(options_section(('image-metadata', "Image Metadata"), {
    "image_metadata": OptionInfo(True, "Save metadata in image"),
    "save_txt": OptionInfo(False, "Save metadata to text file"),
    "save_log_fn": OptionInfo("", "Save metadata to JSON file", component_args=hide_dirs),
    "disable_apply_params": OptionInfo('', "Restore from metadata: skip params", gr.Textbox),
    "disable_apply_metadata": OptionInfo(['sd_model_checkpoint', 'sd_vae', 'sd_unet', 'sd_text_encoder'], "Restore from metadata: skip settings", gr.Dropdown, lambda: {"multiselect":True, "choices": opts.list()}),
}))

options_templates.update(options_section(('ui', "User Interface"), {
    "themes_sep_ui": OptionInfo("<h2>Theme options</h2>", "", gr.HTML),
    "theme_type": OptionInfo("Modern", "Theme type", gr.Radio, {"choices": ["Modern", "Standard", "None"]}),
    "theme_style": OptionInfo("Auto", "Theme mode", gr.Radio, {"choices": ["Auto", "Dark", "Light"]}),
    "gradio_theme": OptionInfo("black-teal", "UI theme", gr.Dropdown, lambda: {"choices": theme.list_themes()}, refresh=theme.refresh_themes),

    "quicksetting_sep_images": OptionInfo("<h2>Quicksettings</h2>", "", gr.HTML),
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"], "Quicksettings list", gr.Dropdown, lambda: {"multiselect":True, "choices": opts.list()}),

    "server_sep_ui": OptionInfo("<h2>Startup & Server Options</h2>", "", gr.HTML),
    "autolaunch": OptionInfo(False, "Autolaunch browser upon startup"),
    "motd": OptionInfo(False, "Show MOTD"),
    "subpath": OptionInfo("", "Mount URL subpath"),
    "ui_request_timeout": OptionInfo(120000, "UI request timeout", gr.Slider, {"minimum": 1000, "maximum": 300000, "step": 10}),

    "ui_tabs": OptionInfo("<h2>UI Tabs</h2>", "", gr.HTML),
    "ui_disabled": OptionInfo([], "Disabled UI tabs", gr.Dropdown, { 'visible': False }),

    "cards_sep_ui": OptionInfo("<h2>Networks panel</h2>", "", gr.HTML),
    "extra_networks_card_size": OptionInfo(140, "Network card size (px)", gr.Slider, {"minimum": 20, "maximum": 2000, "step": 1}),
    "extra_networks_card_cover": OptionInfo("sidebar", "Network panel position", gr.Radio, {"choices": ["cover", "inline", "sidebar"]}),
    "extra_networks_card_square": OptionInfo(True, "Disable variable aspect ratio"),

    "other_sep_ui": OptionInfo("<h2>Other...</h2>", "", gr.HTML),
    "ui_locale": OptionInfo("Auto", "UI locale", gr.Dropdown, lambda: {"choices": theme.list_locales()}),
    "font_size": OptionInfo(14, "Font size", gr.Slider, {"minimum": 8, "maximum": 32, "step": 1}),
    "gpu_monitor": OptionInfo(3000, "GPU monitor interval", gr.Slider, {"minimum": 100, "maximum": 60000, "step": 100}),
    "aspect_ratios": OptionInfo("1:1, 4:3, 3:2, 16:9, 16:10, 21:9, 2:3, 3:4, 9:16, 10:16, 9:21", "Allowed aspect ratios"),
    "compact_view": OptionInfo(False, "Compact view"),
    "ui_columns": OptionInfo(4, "Gallery view columns", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1}),

    "images_sep_log": OptionInfo("<h2>Log Display</h2>", "", gr.HTML),
    "logmonitor_show": OptionInfo(True, "Show log view"),
    "logmonitor_refresh_period": OptionInfo(5000, "Log view update period", gr.Slider, {"minimum": 0, "maximum": 30000, "step": 25}),

    "images_sep_ui": OptionInfo("<h2>Outputs & Images</h2>", "", gr.HTML),
    "return_grid": OptionInfo(True, "Show grid in results"),
    "return_mask": OptionInfo(False, "Inpainting include greyscale mask in results"),
    "return_mask_composite": OptionInfo(False, "Inpainting include masked composite in results"),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface", gr.Checkbox, {"visible": False}),
    "send_size": OptionInfo(False, "Send size when sending prompt or image to another interface", gr.Checkbox, {"visible": False}),
}))

options_templates.update(options_section(('live-preview', "Live Previews"), {
    "show_progress_every_n_steps": OptionInfo(1, "Live preview display period", gr.Slider, {"minimum": 0, "maximum": 20, "step": 1}),
    "show_progress_type": OptionInfo("TAESD", "Live preview method", gr.Radio, {"choices": ["Simple", "Approximate", "TAESD", "Full VAE"]}),
    "live_preview_refresh_period": OptionInfo(500, "Progress update period", gr.Slider, {"minimum": 0, "maximum": 5000, "step": 25}),
    "taesd_variant": OptionInfo(shared_items.sd_taesd_items()[0], "TAESD variant", gr.Dropdown, {"choices": shared_items.sd_taesd_items()}),
    "taesd_layers": OptionInfo(3, "TAESD decode layers", gr.Slider, {"minimum": 1, "maximum": 3, "step": 1}),
    "live_preview_downscale": OptionInfo(True, "Downscale high resolution live previews"),

    "notification_audio_enable": OptionInfo(False, "Play a notification upon completion"),
    "notification_audio_path": OptionInfo("html/notification.mp3","Path to notification sound", component_args=hide_dirs, folder=True),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Additional postprocessing operations", gr.Dropdown, lambda: {"multiselect":True, "choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", gr.Dropdown, lambda: {"multiselect":True, "choices": [x.name for x in shared_items.postprocessing_scripts()], "visible": False }),

    "postprocessing_sep_img2img": OptionInfo("<h2>Inpaint</h2>", "", gr.HTML),
    "img2img_color_correction": OptionInfo(False, "Apply color correction"),
    "mask_apply_overlay": OptionInfo(True, "Apply mask as overlay"),
    "img2img_background_color": OptionInfo("#ffffff", "Image transparent color fill", gr.ColorPicker, {}),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for image processing", gr.Slider, {"minimum": 0.1, "maximum": 1.5, "step": 0.01, "visible": False}),

    "postprocessing_sep_detailer": OptionInfo("<h2>Detailer</h2>", "", gr.HTML),
    "detailer_unload": OptionInfo(False, "Move detailer model to CPU when complete"),
    "detailer_augment": OptionInfo(False, "Detailer use model augment"),

    "postprocessing_sep_seedvt": OptionInfo("<h2>SeedVT</h2>", "", gr.HTML),
    "seedvt_cfg_scale": OptionInfo(3.5, "SeedVR CFG Scale", gr.Slider, {"minimum": 1, "maximum": 15, "step": 1}),

    "postprocessing_sep_face_restore": OptionInfo("<h2>Face Restore</h2>", "", gr.HTML),
    "face_restoration_model": OptionInfo("None", "Face restoration", gr.Radio, lambda: {"choices": ['None'] + [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.2, "CodeFormer weight parameter", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),

    "postprocessing_sep_upscalers": OptionInfo("<h2>Upscaling</h2>", "", gr.HTML),
    "upscaler_unload": OptionInfo(False, "Unload upscaler after processing"),
    "upscaler_latent_steps": OptionInfo(20, "Upscaler latent steps", gr.Slider, {"minimum": 4, "maximum": 100, "step": 1}),
    "upscaler_tile_size": OptionInfo(192, "Upscaler tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "upscaler_tile_overlap": OptionInfo(8, "Upscaler tile overlap", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),
}))


options_templates.update(options_section(('huggingface', "Huggingface"), {
    "huggingface_sep": OptionInfo("<h2>Huggingface</h2>", "", gr.HTML),
    "diffuser_cache_config": OptionInfo(True, "Use cached model config when available"),
    "huggingface_token": OptionInfo('', 'HuggingFace token', gr.Textbox, {"lines": 2}),
    "hf_transfer_mode": OptionInfo("rust", "HuggingFace download method", gr.Radio, {"choices": ['requests', 'rust', 'xet']}),
    "huggingface_mirror": OptionInfo('', 'HuggingFace mirror', gr.Textbox),
    "offline_mode": OptionInfo(False, 'Force offline mode', gr.Checkbox),

    "diffusers_model_load_variant": OptionInfo("default", "Preferred Model variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "diffusers_vae_load_variant": OptionInfo("default", "Preferred VAE variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "custom_diffusers_pipeline": OptionInfo('', 'Load custom Diffusers pipeline'),
    "civitai_token": OptionInfo('', 'HuggingFace token', gr.Textbox, {"lines": 2, "visible": False}),
}))

options_templates.update(options_section(('extra_networks', "Networks"), {
    "extra_networks_sep1": OptionInfo("<h2>Networks UI</h2>", "", gr.HTML),
    "extra_networks_show": OptionInfo(True, "UI show on startup"),
    "extra_networks": OptionInfo(["All"], "Available networks", gr.Dropdown, lambda: {"multiselect":True, "choices": ['All'] + [en.title for en in extra_networks]}),
    "extra_networks_sort": OptionInfo("Default", "Sort order", gr.Dropdown, {"choices": ['Default', 'Name [A-Z]', 'Name [Z-A]', 'Date [Newest]', 'Date [Oldest]', 'Size [Largest]', 'Size [Smallest]']}),
    "extra_networks_view": OptionInfo("gallery", "UI view", gr.Radio, {"choices": ["gallery", "list"]}),
    "extra_networks_sidebar_width": OptionInfo(35, "UI sidebar width (%)", gr.Slider, {"minimum": 10, "maximum": 80, "step": 1}),
    "extra_networks_height": OptionInfo(0, "UI height (%)", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}), # set in ui_javascript
    "extra_networks_fetch": OptionInfo(True, "UI fetch network info on mouse-over"),
    "extra_network_skip_indexing": OptionInfo(False, "Build info on first access", gr.Checkbox),

    "extra_networks_scan_sep": OptionInfo("<h2>Networks Scan</h2>", "", gr.HTML),
    "extra_networks_scan_skip": OptionInfo("", "Skip CivitAI scan for regex pattern(s)", gr.Textbox),

    "extra_networks_model_sep": OptionInfo("<h2>Rerefence models</h2>", "", gr.HTML),
    "extra_network_reference_enable": OptionInfo(True, "Enable use of reference models", gr.Checkbox),
    "extra_network_reference_values": OptionInfo(False, "Use reference values when available", gr.Checkbox),

    "extra_networks_lora_sep": OptionInfo("<h2>LoRA</h2>", "", gr.HTML),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default strength", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "lora_force_reload": OptionInfo(False, "LoRA force reload always"),
    "lora_force_diffusers": OptionInfo(False if not cmd_opts.use_openvino else True, "LoRA load using Diffusers method"),

    "lora_apply_te": OptionInfo(False, "LoRA native apply to text encoder"),
    "lora_fuse_native": OptionInfo(True, "LoRA native fuse with model"),
    "lora_fuse_diffusers": OptionInfo(False, "LoRA diffusers fuse with model"),
    "lora_apply_tags": OptionInfo(0, "LoRA auto-apply tags", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "lora_in_memory_limit": OptionInfo(1, "LoRA memory cache", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
    "lora_add_hashes_to_infotext": OptionInfo(False, "LoRA add hash info to metadata"),

    "extra_networks_styles_sep": OptionInfo("<h2>Styles</h2>", "", gr.HTML),
    "extra_networks_styles": OptionInfo(True, "Show reference styles"),
    "extra_networks_apply_unparsed": OptionInfo(True, "Restore unparsed prompt"),

    "extra_networks_embed_sep": OptionInfo("<h2>Embeddings</h2>", "", gr.HTML),
    "diffusers_enable_embed": OptionInfo(True, "Enable embeddings support", gr.Checkbox),
    "diffusers_convert_embed": OptionInfo(False, "Auto-convert SD15 embeddings to SDXL", gr.Checkbox),

    "extra_networks_wildcard_sep": OptionInfo("<h2>Wildcards</h2>", "", gr.HTML),
    "wildcards_enabled": OptionInfo(True, "Enable file wildcards support"),
}))

options_templates.update(options_section(('extensions', "Extensions"), {
    "disable_all_extensions": OptionInfo("none", "Disable all extensions", gr.Radio, {"choices": ["none", "user", "all"]}),
}))


options_templates.update(options_section(('hidden_options', "Hidden options"), {
    # internal options
    "diffusers_version": OptionInfo("", "Diffusers version", gr.Textbox, {"visible": False}),
    "disabled_extensions": OptionInfo([], "Disable these extensions", gr.Textbox, {"visible": False}),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint", gr.Textbox, {"visible": False}),
    "tooltips": OptionInfo("UI Tooltips", "UI tooltips", gr.Radio, {"choices": ["None", "Browser default", "UI tooltips"], "visible": False}),

    # Caption/Interrogate settings (controlled via Caption Tab UI)
    "interrogate_default_type": OptionInfo("VLM", "Default caption type", gr.Radio, {"choices": ["VLM", "OpenCLiP", "Tagger"], "visible": False}),
    "tagger_show_scores": OptionInfo(False, "Tagger: show confidence scores in results", gr.Checkbox, {"visible": False}),
    "interrogate_clip_model": OptionInfo("ViT-L-14/openai", "OpenCLiP: default model", gr.Dropdown, lambda: {"choices": get_clip_models(), "visible": False}, refresh=refresh_clip_models),
    "interrogate_clip_mode": OptionInfo(caption_types[0], "OpenCLiP: default mode", gr.Dropdown, {"choices": caption_types, "visible": False}),
    "interrogate_blip_model": OptionInfo(list(caption_models)[0], "OpenCLiP: default captioner", gr.Dropdown, {"choices": list(caption_models), "visible": False}),
    "interrogate_clip_num_beams": OptionInfo(1, "OpenCLiP: num beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1, "visible": False}),
    "interrogate_clip_min_length": OptionInfo(32, "OpenCLiP: min length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1, "visible": False}),
    "interrogate_clip_max_length": OptionInfo(74, "OpenCLiP: max length", gr.Slider, {"minimum": 1, "maximum": 512, "step": 1, "visible": False}),
    "interrogate_clip_min_flavors": OptionInfo(2, "OpenCLiP: min flavors", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1, "visible": False}),
    "interrogate_clip_max_flavors": OptionInfo(16, "OpenCLiP: max flavors", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1, "visible": False}),
    "interrogate_clip_flavor_count": OptionInfo(1024, "OpenCLiP: intermediate flavors", gr.Slider, {"minimum": 256, "maximum": 4096, "step": 64, "visible": False}),
    "interrogate_clip_chunk_size": OptionInfo(1024, "OpenCLiP: chunk size", gr.Slider, {"minimum": 256, "maximum": 4096, "step": 64, "visible": False}),
    "interrogate_vlm_model": OptionInfo(vlm_default, "VLM: default model", gr.Dropdown, {"choices": list(vlm_models), "visible": False}),
    "interrogate_vlm_prompt": OptionInfo(vlm_prompts[2], "VLM: default prompt", DropdownEditable, {"choices": vlm_prompts, "visible": False}),
    "interrogate_vlm_system": OptionInfo(vlm_system, "VLM: system prompt", gr.Textbox, {"visible": False}),
    "interrogate_vlm_num_beams": OptionInfo(1, "VLM: num beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1, "visible": False}),
    "interrogate_vlm_max_length": OptionInfo(512, "VLM: max length", gr.Slider, {"minimum": 1, "maximum": 4096, "step": 1, "visible": False}),
    "interrogate_vlm_do_sample": OptionInfo(True, "VLM: use sample method", gr.Checkbox, {"visible": False}),
    "interrogate_vlm_temperature": OptionInfo(0.8, "VLM: temperature", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.01, "visible": False}),
    "interrogate_vlm_top_k": OptionInfo(0, "VLM: top-k", gr.Slider, {"minimum": 0, "maximum": 99, "step": 1, "visible": False}),
    "interrogate_vlm_top_p": OptionInfo(0, "VLM: top-p", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.01, "visible": False}),
    "interrogate_vlm_keep_prefill": OptionInfo(False, "VLM: keep prefill text in output", gr.Checkbox, {"visible": False}),
    "interrogate_vlm_keep_thinking": OptionInfo(False, "VLM: keep reasoning trace in output", gr.Checkbox, {"visible": False}),
    "interrogate_vlm_thinking_mode": OptionInfo(False, "VLM: enable thinking/reasoning mode", gr.Checkbox, {"visible": False}),
    "tagger_threshold": OptionInfo(0.50, "Tagger: general tag threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": False}),
    "tagger_include_rating": OptionInfo(False, "Tagger: include rating tags", gr.Checkbox, {"visible": False}),
    "tagger_max_tags": OptionInfo(74, "Tagger: max tags", gr.Slider, {"minimum": 1, "maximum": 512, "step": 1, "visible": False}),
    "tagger_sort_alpha": OptionInfo(False, "Tagger: sort alphabetically", gr.Checkbox, {"visible": False}),
    "tagger_use_spaces": OptionInfo(False, "Tagger: use spaces for tags", gr.Checkbox, {"visible": False}),
    "tagger_escape_brackets": OptionInfo(True, "Tagger: escape brackets", gr.Checkbox, {"visible": False}),
    "tagger_exclude_tags": OptionInfo("", "Tagger: exclude tags", gr.Textbox, {"visible": False}),
    "waifudiffusion_model": OptionInfo("wd-eva02-large-tagger-v3", "WaifuDiffusion: default model", gr.Dropdown, {"choices": [], "visible": False}),
    "waifudiffusion_character_threshold": OptionInfo(0.85, "WaifuDiffusion: character tag threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": False}),

    # control settings are handled separately
    "control_hires": OptionInfo(False, "Hires use Control", gr.Checkbox, {"visible": False}),
    "control_aspect_ratio": OptionInfo(False, "Aspect ratio resize", gr.Checkbox, {"visible": False}),
    "control_max_units": OptionInfo(4, "Maximum number of units", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1, "visible": False}),
    "control_tiles": OptionInfo("1x1, 1x2, 1x3, 1x4, 2x1, 2x1, 2x2, 2x3, 2x4, 3x1, 3x2, 3x3, 3x4, 4x1, 4x2, 4x3, 4x4", "Tiling options", gr.Textbox, {"visible": False}),
    "control_move_processor": OptionInfo(False, "Processor move to CPU after use", gr.Checkbox, {"visible": False}),
    "control_unload_processor": OptionInfo(False, "Processor unload after use", gr.Checkbox, {"visible": False}),

    # sampler settings are handled separately
    "show_samplers": OptionInfo([], "Show samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()], "visible": False}),
    'eta_noise_seed_delta': OptionInfo(0, "Noise seed delta (eta)", gr.Number, {"precision": 0, "visible": False}),
    "scheduler_eta": OptionInfo(1.0, "Noise multiplier (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": False}),
    "schedulers_solver_order": OptionInfo(0, "Solver order (where", gr.Slider, {"minimum": 0, "maximum": 5, "step": 1, "visible": False}),
    "schedulers_use_loworder": OptionInfo(True, "Use simplified solvers in final steps", gr.Checkbox, {"visible": False}),
    "schedulers_prediction_type": OptionInfo("default", "Override model prediction type", gr.Radio, {"choices": ['default', 'epsilon', 'sample', 'v_prediction', 'flow_prediction'], "visible": False}),
    "schedulers_sigma": OptionInfo("default", "Sigma algorithm", gr.Radio, {"choices": ['default', 'karras', 'exponential', 'polyexponential'], "visible": False}),
    "schedulers_beta_schedule": OptionInfo("default", "Beta schedule", gr.Dropdown, {"choices": ['default', 'linear', 'scaled_linear', 'squaredcos_cap_v2', 'sigmoid'], "visible": False}),
    "schedulers_use_thresholding": OptionInfo(False, "Use dynamic thresholding", gr.Checkbox, {"visible": False}),
    "schedulers_timestep_spacing": OptionInfo("default", "Timestep spacing", gr.Dropdown, {"choices": ['default', 'linspace', 'leading', 'trailing'], "visible": False}),
    'schedulers_timesteps': OptionInfo('', "Timesteps", gr.Textbox, {"visible": False}),
    "schedulers_rescale_betas": OptionInfo(False, "Rescale betas with zero terminal SNR", gr.Checkbox, {"visible": False}),
    'schedulers_beta_start': OptionInfo(0, "Beta start", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001}),
    'schedulers_beta_end': OptionInfo(0, "Beta end", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001}),
    'schedulers_timesteps_range': OptionInfo(1000, "Timesteps range", gr.Slider, {"minimum": 250, "maximum": 4000, "step": 1}),
    'schedulers_shift': OptionInfo(3, "Sampler shift", gr.Slider, {"minimum": 0.1, "maximum": 10, "step": 0.1, "visible": False}),
    'schedulers_dynamic_shift': OptionInfo(False, "Sampler dynamic shift", gr.Checkbox, {"visible": False}),
    'schedulers_sigma_adjust': OptionInfo(1.0, "Sigma adjust", gr.Slider, {"minimum": 0.5, "maximum": 1.5, "step": 0.01, "visible": False}),
    'schedulers_sigma_adjust_min': OptionInfo(0.2, "Sigma adjust start", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": False}),
    'schedulers_sigma_adjust_max': OptionInfo(0.8, "Sigma adjust end", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": False}),
    'schedulers_base_shift': OptionInfo(0.5, "Sampler base shift", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": False}),
    'schedulers_max_shift': OptionInfo(1.15, "Sampler max shift", gr.Slider, {"minimum": 0.0, "maximum": 4.0, "step": 0.01, "visible": False}),
    'uni_pc_variant': OptionInfo("bh2", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"], "visible": False}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"], "visible": False}),

    # detailer settings are handled separately
    "detailer_model": OptionInfo("Detailer", "Detailer model", gr.Radio, lambda: {"choices": [x.name() for x in detailers], "visible": False}),
    "detailer_classes": OptionInfo("", "Detailer classes", gr.Textbox, { "visible": False}),
    "detailer_conf": OptionInfo(0.6, "Min confidence", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_max": OptionInfo(2, "Max detected", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1, "visible": False}),
    "detailer_iou": OptionInfo(0.5, "Max overlap", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_sigma_adjust": OptionInfo(1.0, "Detailer sigma adjust", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_sigma_adjust_max": OptionInfo(1.0, "Detailer sigma end", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_min_size": OptionInfo(0.0, "Min object size", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_max_size": OptionInfo(1.0, "Max object size", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_padding": OptionInfo(20, "Item padding", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1, "visible": False}),
    "detailer_blur": OptionInfo(10, "Item edge blur", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1, "visible": False}),
    "detailer_models": OptionInfo(['face-yolo8n'], "Detailer models", gr.Dropdown, lambda: {"multiselect":True, "choices": list(yolo.list), "visible": False}),
    "detailer_args": OptionInfo("", "Detailer args", gr.Textbox, { "visible": False}),
    "detailer_merge": OptionInfo(False, "Merge multiple results from each detailer model", gr.Checkbox, {"visible": False}),
    "detailer_sort": OptionInfo(False, "Sort detailer output by location", gr.Checkbox, {"visible": False}),
    "detailer_save": OptionInfo(False, "Include detection results", gr.Checkbox, {"visible": False}),
    "detailer_seg": OptionInfo(False, "Use segmentation", gr.Checkbox, {"visible": False}),
}))


from modules.shared_legacy import get_legacy_options
options_templates.update(get_legacy_options())
from modules.options_handler import Options
config_filename = cmd_opts.config
opts = Options(options_templates, restricted_opts, filename=config_filename)
cmd_opts = cmd_args.settings_args(opts, cmd_opts)
if cmd_opts.locale is not None:
    opts.data['ui_locale'] = cmd_opts.locale
if cmd_opts.use_xformers:
    opts.data['cross_attention_optimization'] = 'xFormers'
opts.data['uni_pc_lower_order_final'] = opts.schedulers_use_loworder # compatibility
opts.data['uni_pc_order'] = max(2, opts.schedulers_solver_order) # compatibility
log.info(f'Engine: backend={backend} compute={devices.backend} device={devices.get_optimal_device_name()} attention="{opts.cross_attention_optimization}" mode={devices.inference_context.__name__}')

profiler = None
import modules.styles
prompt_styles = modules.styles.StyleDatabase(opts)
reference_models = readfile(os.path.join('data', 'reference.json'), as_type="dict") if opts.extra_network_reference_enable else {}
reference_models.update(readfile(os.path.join('data', 'reference-nunchaku.json'), as_type="dict") if opts.extra_network_reference_enable else {})
cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or (cmd_opts.server_name or False)) and not cmd_opts.insecure

log.debug('Initializing: devices')
devices.args = cmd_opts
devices.opts = opts
devices.onnx = [opts.onnx_execution_provider]
devices.set_cuda_params()
if opts.onnx_cpu_fallback and 'CPUExecutionProvider' not in devices.onnx:
    devices.onnx.append('CPUExecutionProvider')
device = devices.device
parallel_processing_allowed = not cmd_opts.lowvram
mem_mon = modules.memmon.MemUsageMonitor("MemMon", devices.device)
history = history.History()
if devices.backend == "directml":
    directml_do_hijack()
from modules import sdnq # pylint: disable=unused-import # register to diffusers and transformers
log.debug('Quantization: registered=SDNQ')

try:
    log.info(f'Device: {print_dict(devices.get_gpu_info())}')
except Exception as ex:
    log.error(f'Device: {ex}')


def restart_server(restart=True):
    if demo is None:
        return
    log.critical('Server shutdown requested')
    try:
        sys.tracebacklimit = 0
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stdout(stderr):
            demo.server.wants_restart = restart
            demo.server.should_exit = True
            demo.server.force_exit = True
            demo.close(verbose=False)
            demo.server.close()
            demo.fns = []
        time.sleep(1)
        sys.tracebacklimit = 100
        # os._exit(0)
    except (Exception, BaseException) as err:
        log.error(f'Server shutdown error: {err}')
    if restart:
        log.info('Server will restart')


def restore_defaults(restart=True):
    if os.path.exists(cmd_opts.config):
        log.info('Restoring server defaults')
        os.remove(cmd_opts.config)
    restart_server(restart)


# startup def of shared.sd_model before its redefined in modeldata
sd_model: DiffusionPipeline | None = None # dummy and overwritten by class
sd_refiner: DiffusionPipeline | None = None # dummy and overwritten by class
sd_model_type: str = '' # dummy and overwritten by class
sd_refiner_type: str = '' # dummy and overwritten by class
sd_loaded: bool = False # dummy and overwritten by class

from modules.modeldata import Shared # pylint: disable=ungrouped-imports
sys.modules[__name__].__class__ = Shared
