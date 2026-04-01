import os
import torch
from safetensors.torch import load_file as load_safetensors_file
from huggingface_hub import hf_hub_download
from ..optimization.memory_manager import preinitialize_rope_cache
from ..common.config import load_config, create_object
from ..core.infer import VideoDiffusionInfer


def configure_runner(model_name, cache_dir, device:str='cpu', dtype:torch.dtype=None):
    from installer import install
    install('omegaconf')
    from omegaconf import OmegaConf

    repo_id = "vladmandic/SeedVR2"
    script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(script_directory, './config_7b.yaml') if "7b" in model_name else os.path.join(script_directory, './config_3b.yaml')
    config = load_config(config_path)

    runner = VideoDiffusionInfer(config, device=device, dtype=dtype)
    OmegaConf.set_readonly(runner.config, False)

    # load dit
    with torch.device("meta"):
        runner.dit = create_object(config.dit.model)
        runner.dit.requires_grad_(False).eval()
        runner.dit.to_empty(device="cpu")
    model_file = hf_hub_download(repo_id=repo_id, filename=model_name, cache_dir=cache_dir)
    state_dict = load_safetensors_file(model_file)
    runner.dit.load_state_dict(state_dict, assign=True)
    runner.dit = runner.dit.to(device="cpu", dtype=dtype)
    del state_dict

    # load vae
    vae_config_path = os.path.join(script_directory, 'src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml')
    vae_config = OmegaConf.load(vae_config_path)
    config.vae.model = OmegaConf.merge(config.vae.model, vae_config)

    vae_file = hf_hub_download(repo_id=repo_id, filename=config.vae.checkpoint, cache_dir=cache_dir)
    with torch.device("meta"):
        runner.vae = create_object(config.vae.model)
        runner.vae.requires_grad_(False).eval()
        runner.vae.to_empty(device="cpu")
    state_dict = load_safetensors_file(vae_file)
    runner.vae.load_state_dict(state_dict)
    runner.vae = runner.vae.to(device="cpu", dtype=dtype)
    runner.config.vae.dtype = str(dtype)
    runner.config.vae.slicing = {'split_size': 8, 'memory_device': 'same'}
    runner.config.vae.memory_limit = {'conv_max_mem': 0.2, 'norm_max_mem': 0.2}
    runner.vae.set_causal_slicing(**runner.config.vae.slicing)
    runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    del state_dict

    # load embeds
    pos_embeds_file = hf_hub_download(repo_id=repo_id, filename='pos_emb.pt', cache_dir=cache_dir)
    neg_embeds_file = hf_hub_download(repo_id=repo_id, filename='neg_emb.pt', cache_dir=cache_dir)
    runner.text_pos_embeds = torch.load(pos_embeds_file).to(device=device, dtype=dtype)
    runner.text_neg_embeds = torch.load(neg_embeds_file).to(device=device, dtype=dtype)

    return runner
