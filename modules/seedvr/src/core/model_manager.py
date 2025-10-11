import os
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors_file
from huggingface_hub import hf_hub_download

from src.optimization.memory_manager import preinitialize_rope_cache
from src.common.config import load_config, create_object
from src.core.infer import VideoDiffusionInfer


def configure_runner(model_name, cache_dir, device:str='cpu', dtype:torch.dtype=None):
    repo_id = "vladmandic/SeedVR2"
    script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(script_directory, './config_7b.yaml') if "7b" in model_name else os.path.join(script_directory, './config_3b.yaml')
    config = load_config(config_path)

    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    vae_config = OmegaConf.load(vae_config_path)
    vae_config.spatial_downsample_factor = vae_config.get('spatial_downsample_factor', 8)
    vae_config.temporal_downsample_factor = vae_config.get('temporal_downsample_factor', 4)
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)

    runner = VideoDiffusionInfer(config, device=device, dtype=dtype)
    OmegaConf.set_readonly(runner.config, False)

    # load dit
    with torch.device("meta"):
        runner.dit = create_object(config.dit.model)
        runner.dit.eval().to(dtype)
        runner.dit.to_empty(device="cpu")

    model_file = hf_hub_download(repo_id=repo_id, filename=model_name, cache_dir=cache_dir)
    state_dict = load_safetensors_file(model_file)
    runner.dit.load_state_dict(state_dict, assign=True)
    del state_dict
    runner.dit = runner.dit.to(device=device, dtype=dtype)

    # load vae
    vae_file = hf_hub_download(repo_id=repo_id, filename=config.vae.checkpoint, cache_dir=cache_dir)
    with torch.device("meta"):
        runner.vae = create_object(config.vae.model)
        runner.vae.requires_grad_(False).eval()
        runner.vae.to_empty(device="cpu")

    state_dict = load_safetensors_file(vae_file)
    runner.vae.load_state_dict(state_dict)
    del state_dict
    runner.vae = runner.vae.to(device=device, dtype=dtype)
    runner.config.vae.dtype = str(dtype)
    runner.vae.set_causal_slicing(**config.vae.slicing)
    runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

    # load embeds
    pos_embeds_file = hf_hub_download(repo_id=repo_id, filename='pos_emb.pt', cache_dir=cache_dir)
    neg_embeds_file = hf_hub_download(repo_id=repo_id, filename='neg_emb.pt', cache_dir=cache_dir)
    runner.text_pos_embeds = torch.load(pos_embeds_file).to(device=device, dtype=dtype)
    runner.text_neg_embeds = torch.load(neg_embeds_file).to(device=device, dtype=dtype)

    preinitialize_rope_cache(runner)
    return runner
