import time
import torch
import diffusers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from modules import shared, devices
from modules.logger import log


decoder_id = "ostris/vae-kl-f8-d16"
adapter_id = "ostris/16ch-VAE-Adapters"


def load_vae(pipe):
    if shared.sd_model_type == 'sd':
        adapter_file = "16ch-VAE-Adapter-SD15-alpha.safetensors"
    elif shared.sd_model_type == 'sdxl':
        adapter_file = "16ch-VAE-Adapter-SDXL-alpha_v02.safetensors"
    else:
        log.error('VAE: type=osiris unsupported model type')
        return
    t0 = time.time()
    ckpt_file = hf_hub_download(adapter_id, adapter_file, cache_dir=shared.opts.hfcache_dir)
    ckpt = load_file(ckpt_file)
    lora_state_dict = {k: v for k, v in ckpt.items() if "lora" in k}
    unet_state_dict = {k.replace("unet_", ""): v for k, v in ckpt.items() if "unet_" in k}

    pipe.unet.conv_in = torch.nn.Conv2d(16, 320, 3, 1, 1)
    pipe.unet.conv_out = torch.nn.Conv2d(320, 16, 3, 1, 1)
    pipe.unet.load_state_dict(unet_state_dict, strict=False)
    pipe.unet.conv_in.to(devices.dtype)
    pipe.unet.conv_out.to(devices.dtype)
    pipe.unet.config.in_channels = 16
    pipe.unet.config.out_channels = 16

    pipe.load_lora_weights(lora_state_dict, adapter_name=adapter_id)
    # pipe.set_adapters(adapter_names=[adapter_id], adapter_weights=[0.8])
    pipe.fuse_lora(adapter_names=[adapter_id], lora_scale=0.8, fuse_unet=True)

    pipe.vae = diffusers.AutoencoderKL.from_pretrained(decoder_id, torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
    t1 = time.time()
    log.info(f'VAE load: type=osiris decoder="{decoder_id}" adapter="{adapter_id}" time={t1-t0:.2f}s')
