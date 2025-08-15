import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules import shared, errors, devices, sd_models, model_quant


debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_flux_quanto(checkpoint_info):
    transformer, text_encoder_2 = None, None
    quanto = model_quant.load_quanto('Load model: type=FLUX')

    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path

    try:
        quantization_map = os.path.join(repo_path, "transformer", "quantization_map.json")
        debug(f'Load model: type=FLUX quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="transformer"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info)
            quantization_map = hf_hub_download(repo_id, subfolder='transformer', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        state_dict = load_file(os.path.join(repo_path, "transformer", "diffusion_pytorch_model.safetensors"))
        dtype = state_dict['context_embedder.bias'].dtype
        with torch.device("meta"):
            transformer = diffusers.FluxTransformer2DModel.from_config(os.path.join(repo_path, "transformer", "config.json")).to(dtype=dtype)
        quanto.requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
        transformer_dtype = transformer.dtype
        if transformer_dtype != devices.dtype:
            try:
                transformer = transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=FLUX Failed to cast transformer to {devices.dtype}, set dtype to {transformer_dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=FLUX failed to load Quanto transformer: {e}")
        if debug:
            errors.display(e, 'FLUX Quanto:')

    try:
        quantization_map = os.path.join(repo_path, "text_encoder_2", "quantization_map.json")
        debug(f'Load model: type=FLUX quantization map="{quantization_map}" repo="{checkpoint_info.name}" component="text_encoder_2"')
        if not os.path.exists(quantization_map):
            repo_id = sd_models.path_to_repo(checkpoint_info)
            quantization_map = hf_hub_download(repo_id, subfolder='text_encoder_2', filename='quantization_map.json', cache_dir=shared.opts.diffusers_dir)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        with open(os.path.join(repo_path, "text_encoder_2", "config.json"), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        state_dict = load_file(os.path.join(repo_path, "text_encoder_2", "model.safetensors"))
        dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
        with torch.device("meta"):
            text_encoder_2 = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
        quanto.requantize(text_encoder_2, state_dict, quantization_map, device=torch.device("cpu"))
        text_encoder_2_dtype = text_encoder_2.dtype
        if text_encoder_2_dtype != devices.dtype:
            try:
                text_encoder_2 = text_encoder_2.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"Load model: type=FLUX Failed to cast text encoder to {devices.dtype}, set dtype to {text_encoder_2_dtype}")
    except Exception as e:
        shared.log.error(f"Load model: type=FLUX failed to load Quanto text encoder: {e}")
        if debug:
            errors.display(e, 'FLUX Quanto:')

    return transformer, text_encoder_2
