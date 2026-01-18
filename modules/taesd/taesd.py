import torch
import torch.nn as nn
from modules import devices


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )

def Decoder(latent_channels=4):
    from modules import shared
    if shared.opts.taesd_layers == 1:
        return nn.Sequential(
            Clamp(), conv(latent_channels, 64), nn.ReLU(),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Identity(), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Identity(), conv(64, 64, bias=False),
            Block(64, 64), conv(64, 3),
        )
    elif shared.opts.taesd_layers == 2:
        return nn.Sequential(
            Clamp(), conv(latent_channels, 64), nn.ReLU(),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Identity(), conv(64, 64, bias=False),
            Block(64, 64), conv(64, 3),
        )
    else:
        return nn.Sequential(
            Clamp(), conv(latent_channels, 64), nn.ReLU(),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
            Block(64, 64), conv(64, 3),
        )


class TAESD(nn.Module): # pylint: disable=abstract-method
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=None):
        super().__init__()
        self.dtype = devices.dtype_vae if devices.dtype_vae != torch.bfloat16 else torch.float16 # taesd does not support bf16
        if latent_channels is None:
            latent_channels = self.guess_latent_channels(str(decoder_path), str(encoder_path))
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"), strict=False)
            self.encoder.eval()
            self.encoder = self.encoder.to(devices.device, dtype=self.dtype)
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"), strict=False)
            self.decoder.eval()
            self.decoder = self.decoder.to(devices.device, dtype=self.dtype)

    def guess_latent_channels(self, decoder_path, encoder_path):
        if "f2" in encoder_path or "f2" in decoder_path:
            return 32  # FLUX.2 uses 32 latent channels
        if ("f1" in encoder_path or "f1" in decoder_path) or ("sd3" in encoder_path or "sd3" in decoder_path):
            return 16
        return 4

    @staticmethod
    def scale_latents(x):
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1) # raw latents -> [0, 1]

    @staticmethod
    def unscale_latents(x):
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude) # [0, 1] -> raw latents
