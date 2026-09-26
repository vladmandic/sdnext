#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1, n_gn = lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False), n_in*4 # pylint: disable=unnecessary-lambda-assignment
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))
    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder(latent_channels=4, use_midblock_gn=False, image_channels=3):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        conv(image_channels, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )

def Decoder(latent_channels=4, use_midblock_gn=False, image_channels=3):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, image_channels),
    )

def F32Encoder(latent_channels=32, image_channels=3):
    """Encoder variant with 32x spatial downscaling instead of 8x."""
    return nn.Sequential(
        conv(image_channels, 32, stride=2), nn.ReLU(inplace=True), conv(32, 64, stride=2), nn.ReLU(inplace=True), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )

def F32Decoder(latent_channels=32, image_channels=3):
    """Decoder variant with 32x spatial upscaling instead of 8x."""
    return nn.Sequential(
        Clamp(), conv(latent_channels, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, image_channels),
    )

def F16Encoder(latent_channels=64, image_channels=3):
    """Encoder variant with 16x spatial downscaling, 2x2 input patchify, and wider low-res stages."""
    return nn.Sequential(
        nn.PixelUnshuffle(2), conv(image_channels * 4, 64), nn.ReLU(inplace=True), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 128, stride=2, bias=False), Block(128, 128), Block(128, 128), Block(128, 128),
        conv(128, 256, stride=2, bias=False), Block(256, 256), Block(256, 256), Block(256, 256),
        conv(256, latent_channels),
    )

def F16Decoder(latent_channels=64, image_channels=3):
    """Decoder variant with 16x spatial upscaling, wider low-res stages, and 2x2 output patchify."""
    return nn.Sequential(
        Clamp(), conv(latent_channels, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, image_channels * 4), nn.PixelShuffle(2),
    )

class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=None, arch_variant=None, image_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        if latent_channels is None:
            latent_channels, arch_variant = self.guess_latent_channels_and_arch(str(encoder_path or decoder_path))
        if image_channels is None:
            image_channels = self.guess_image_channels(str(encoder_path or decoder_path))
        # 3 for RGB, 4 for RGBA (straight alpha)
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        # flux_2 required global pooling/norm for accurate distillation, enable conditionally
        self.encoder = Encoder(latent_channels, use_midblock_gn=(arch_variant in ["flux_2"]), image_channels=image_channels)
        self.decoder = Decoder(latent_channels, use_midblock_gn=(arch_variant in ["flux_2"]), image_channels=image_channels)
        # sana dcae requires 32x spatial downscaling, enable conditionally
        if arch_variant == "f32":
            self.encoder, self.decoder = F32Encoder(latent_channels, image_channels), F32Decoder(latent_channels, image_channels)
        # qwen image 2.1 vae requires 16x spatial downscaling, enable conditionally
        if arch_variant == "f16":
            self.encoder, self.decoder = F16Encoder(latent_channels, image_channels), F16Decoder(latent_channels, image_channels)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))

    def guess_latent_channels(self, encoder_path):
        """Guess latent channel count based on encoder filename"""
        return self.guess_latent_channels_and_arch(encoder_path)[0]

    def guess_latent_channels_and_arch(self, encoder_path):
        """Guess latent channel count and architecture variant based on encoder filename"""
        if "taef1" in encoder_path:
            return 16, None
        if "taef2" in encoder_path:
            return 32, "flux_2"
        if "taesd3" in encoder_path:
            return 16, None
        if "taesana" in encoder_path:
            return 32, "f32" # f32c32
        if "taeqi2_1" in encoder_path:
            return 64, "f16" # f16c64
        return 4, None

    def guess_image_channels(self, encoder_path):
        """Guess image channel count (3 for RGB, 4 for RGBA) based on encoder filename"""
        if "taeqi2_1" in encoder_path:
            return 4 # qwen image 2.1 vae is rgba
        return 3

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)


@torch.no_grad()
def main():
    from PIL import Image
    import sys
    import torchvision.transforms.functional as TF
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device", dev)
    taesd = TAESD().to(dev)
    # rgba models also accept rgb images, as rgba with alpha=1
    im_mode = "RGBA" if taesd.image_channels == 4 else "RGB"
    for im_path in sys.argv[1:]:
        im = TF.to_tensor(Image.open(im_path).convert(im_mode)).unsqueeze(0).to(dev)

        # encode image, quantize, and save to file
        im_enc = taesd.scale_latents(taesd.encoder(im)).mul_(255).round_().byte()
        # fold 4k channels into k vertically stacked 4-channel tiles (no-op for 4-channel latents)
        im_enc = im_enc[0].unflatten(0, (-1, 4)).transpose(0, 1).flatten(1, 2)
        enc_path = im_path + ".encoded.png"
        TF.to_pil_image(im_enc).save(enc_path)
        print(f"Encoded {im_path} to {enc_path}")

        # load the saved file, dequantize, and decode
        im_enc = TF.to_tensor(Image.open(enc_path))
        im_enc = im_enc.unflatten(1, (taesd.latent_channels // 4, -1)).transpose(0, 1).flatten(0, 1)
        im_enc = taesd.unscale_latents(im_enc.unsqueeze(0).to(dev))
        im_dec = taesd.decoder(im_enc).clamp(0, 1)
        dec_path = im_path + ".decoded.png"
        print(f"Decoded {enc_path} to {dec_path}")
        TF.to_pil_image(im_dec[0]).save(dec_path)


def bench():
    import sys
    import time
    model = sys.argv[1] if len(sys.argv) > 1 else ''
    if not model:
        print("Provide a model path as the first argument.")
        return
    device = torch.device("cuda")
    taesd = TAESD(decoder_path=model).to(device=device, dtype=torch.float16)
    taesd.eval()
    latent = torch.randn(1, taesd.latent_channels, 64, 64, device=device, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(10): # warmup
            decoded = taesd.decoder(latent)
        t0 = time.perf_counter()
        for _ in range(1000):
            torch.cuda.synchronize()
            decoded = taesd.decoder(latent)
        t1 = time.perf_counter()
        print(f"Perf: time={t1 - t0} input={latent.shape} output={decoded.shape} model={model}")

if __name__ == "__main__":
    bench()
    # main()
