# converted from <https://github.com/city96/SD-Latent-Interposer>

import os
import time
import torch
import torch.nn as nn
from safetensors.torch import load_file


# v1 = Stable Diffusion 1.x
# xl = Stable Diffusion Extra Large (SDXL)
# v3 = Stable Diffusion Version Three (SD3)
# fx = Black Forest Labs Flux dot One
# cc = Stable Cascade (Stage C) [not used]
# ca = Stable Cascade (Stage A/B)
config = {
    "v1-to-xl": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "v1-to-v3": {"ch_in": 4, "ch_out":16, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "xl-to-v1": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "xl-to-v3": {"ch_in": 4, "ch_out":16, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "v3-to-v1": {"ch_in":16, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "v3-to-xl": {"ch_in":16, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "fx-to-v1": {"ch_in":16, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "fx-to-xl": {"ch_in":16, "ch_out": 4, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "fx-to-v3": {"ch_in":16, "ch_out":16, "ch_mid": 64, "scale": 1.0, "blocks": 12},
    "ca-to-v1": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 0.5, "blocks": 12},
    "ca-to-xl": {"ch_in": 4, "ch_out": 4, "ch_mid": 64, "scale": 0.5, "blocks": 12},
    "ca-to-v3": {"ch_in": 4, "ch_out":16, "ch_mid": 64, "scale": 0.5, "blocks": 12},
}


class ResBlock(nn.Module):
    """Block with residuals"""
    def __init__(self, ch):
        super().__init__()
        self.join = nn.ReLU()
        self.norm = nn.BatchNorm2d(ch)
        self.long = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        x = self.norm(x)
        return self.join(self.long(x) + x)


class ExtractBlock(nn.Module):
    """Increase no. of channels by [out/in]"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.join  = nn.ReLU()
        self.short = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.long  = nn.Sequential(
            nn.Conv2d( ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.join(self.long(x) + self.short(x))


class InterposerModel(nn.Module):
    """
    NN layout, ported from:
    https://github.com/city96/SD-Latent-Interposer/blob/main/interposer.py
    """
    def __init__(self, ch_in=4, ch_out=4, ch_mid=64, scale=1.0, blocks=12):
        super().__init__()
        self.ch_in  = ch_in
        self.ch_out = ch_out
        self.ch_mid = ch_mid
        self.blocks = blocks
        self.scale  = scale

        self.head = ExtractBlock(self.ch_in, self.ch_mid)
        self.core = nn.Sequential(
            nn.Upsample(scale_factor=self.scale, mode="nearest"),
            *[ResBlock(self.ch_mid) for _ in range(blocks)],
            nn.BatchNorm2d(self.ch_mid),
            nn.SiLU(),
        )
        self.tail = nn.Conv2d(self.ch_mid, self.ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y = self.head(x)
        z = self.core(y)
        return self.tail(z)


def map_model_name(name: str):
    if name == 'sd':
        return 'v1'
    if name == 'sdxl':
        return 'xl'
    if name == 'sd3':
        return 'v3'
    if name in ['f1', 'chroma']:
        return 'fx'
    return name


class Interposer:
    def __init__(self):
        self.version = 4.0 # network revision
        self.loaded = None # current model name
        self.model = None  # current model
        self.vae = None # current VAE

    def convert(self, src: str, dst: str, latents: torch.Tensor):
        from diffusers import AutoencoderKL
        from huggingface_hub import hf_hub_download
        from modules import shared, devices

        src = map_model_name(src)
        dst = map_model_name(dst)
        if src == dst:
            return None
        model_name = f"{src}-to-{dst}"
        if model_name not in config:
            shared.log.error(f'Interposer: model="{model_name}" unknown')
            return None
        if (self.loaded != model_name) or (self.model is None):
            model_fn = hf_hub_download(
                repo_id="city96/SD-Latent-Interposer",
                subfolder=f"v{self.version}",
                filename=f"{model_name}_interposer-v{self.version}.safetensors",
                cache_dir=shared.opts.hfcache_dir,
            )
            self.model = InterposerModel(**config[model_name])
            self.model = self.model.to(device=devices.cpu, dtype=torch.float32)
            self.model.eval()
            self.model.load_state_dict(load_file(model_fn))
            self.loaded = model_name
            if dst == 'v1':
                vae_repo = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
                self.vae = AutoencoderKL.from_pretrained(vae_repo, subfolder='vae', cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype)
            elif dst == 'xl':
                vae_repo = 'madebyollin/sdxl-vae-fp16-fix'
                self.vae = AutoencoderKL.from_pretrained(vae_repo, cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype)
            elif dst == 'v3':
                vae_repo = 'stabilityai/stable-diffusion-3.5-large'
                self.vae = AutoencoderKL.from_pretrained(vae_repo, subfolder='vae', cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype)
            elif dst == 'fx':
                vae_repo = 'black-forest-labs/FLUX.1-dev'
                self.vae = AutoencoderKL.from_pretrained(vae_repo, subfolder='vae', cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype)

        t0 = time.time()
        if self.model is None or self.vae is None:
            return None
        with torch.no_grad():
            latent = latents.clone().cpu().float() # force fp32, always run on CPU
            output = self.model(latent)
            output = output.to(device=latents.device, dtype=latents.dtype)
        t1 = time.time()
        shared.log.debug(f'Interposer: src={src}/{list(latents.shape)} dst={dst}/{list(output.shape)} model="{os.path.basename(model_fn)}" vae="{vae_repo}" time={t1-t0:.2f}')
        # shared.log.debug(f'Interposer: src={latents.aminmax()} dst={output.aminmax()}')
        return output
