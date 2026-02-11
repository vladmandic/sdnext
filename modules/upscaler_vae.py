import time
from PIL import Image
from modules.upscaler import Upscaler, UpscalerData


class UpscalerAsymmetricVAE(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Asymmetric VAE"
        self.vae = None
        self.selected = None
        self.scalers = [
            UpscalerData("Asymmetric VAE v1", None, self),
            UpscalerData("Asymmetric VAE v2", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        if selected_model is None:
            return img
        import diffusers
        from modules import shared, devices, images_sharpfin
        if self.vae is None or (selected_model != self.selected):
            if 'v1' in selected_model:
                repo_id = 'Heasterian/AsymmetricAutoencoderKLUpscaler'
            else:
                repo_id = 'Heasterian/AsymmetricAutoencoderKLUpscaler_v2'
            self.vae = diffusers.AsymmetricAutoencoderKL.from_pretrained(repo_id, cache_dir=shared.opts.hfcache_dir)
            self.vae.requires_grad_(False)
            self.vae = self.vae.to(device=devices.device, dtype=devices.dtype)
            self.vae.eval()
            self.selected = selected_model
            shared.log.debug(f'Upscaler load: selected="{self.selected}" vae="{repo_id}"')
        t0 = time.time()
        img = images_sharpfin.resize(img, (8 * (img.width // 8), 8 * (img.height // 8))).convert('RGB')
        tensor = images_sharpfin.to_tensor(img).unsqueeze(0).to(device=devices.device, dtype=devices.dtype)
        self.vae = self.vae.to(device=devices.device)
        tensor = self.vae(tensor).sample
        upscaled = images_sharpfin.to_pil(tensor.squeeze().clamp(0.0, 1.0).float().cpu())
        self.vae = self.vae.to(device=devices.cpu)
        t1 = time.time()
        shared.log.debug(f'Upscale: name="{self.selected}" input={img.size} output={upscaled.size} time={t1 - t0:.2f}')
        return upscaled


class UpscalerWanUpscale(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "WAN Upscale"
        self.vae_encode = None
        self.vae_decode = None
        self.selected = None
        self.scalers = [
            UpscalerData("WAN Asymmetric Upscale", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        if selected_model is None:
            return img
        import torch.nn.functional as FN
        import diffusers
        from modules import shared, devices, images_sharpfin
        if (self.vae_encode is None) or (self.vae_decode is None) or (selected_model != self.selected):
            repo_encode = 'Qwen/Qwen-Image-Edit-2509'
            subfolder_encode = 'vae'
            self.vae_encode = diffusers.AutoencoderKLWan.from_pretrained(repo_encode, subfolder=subfolder_encode, cache_dir=shared.opts.hfcache_dir)
            self.vae_encode.requires_grad_(False)
            self.vae_encode = self.vae_encode.to(device=devices.device, dtype=devices.dtype)
            self.vae_encode.eval()
            repo_decode = 'spacepxl/Wan2.1-VAE-upscale2x'
            subfolder_decode = "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1"
            self.vae_decode = diffusers.AutoencoderKLWan.from_pretrained(repo_decode, subfolder=subfolder_decode, cache_dir=shared.opts.hfcache_dir)
            self.vae_decode.requires_grad_(False)
            self.vae_decode = self.vae_decode.to(device=devices.device, dtype=devices.dtype)
            self.vae_decode.eval()
            self.selected = selected_model
            shared.log.debug(f'Upscaler load: selected="{self.selected}" encode="{repo_encode}" decode="{repo_decode}"')

        t0 = time.time()
        self.vae_encode = self.vae_encode.to(device=devices.device)
        tensor = images_sharpfin.to_tensor(img).unsqueeze(0).unsqueeze(2).to(device=devices.device, dtype=devices.dtype)
        tensor = self.vae_encode.encode(tensor).latent_dist.mode()
        self.vae_encode.to(device=devices.cpu)

        self.vae_decode = self.vae_decode.to(device=devices.device)
        tensor = self.vae_decode.decode(tensor).sample
        tensor = FN.pixel_shuffle(tensor.movedim(2, 1), upscale_factor=2).movedim(1, 2) # pixel shuffle needs [..., C, H, W] format
        self.vae_decode.to(device=devices.cpu)

        upscaled = images_sharpfin.to_pil(tensor.squeeze().clamp(0.0, 1.0).float().cpu())
        t1 = time.time()
        shared.log.debug(f'Upscale: name="{self.selected}" input={img.size} output={upscaled.size} time={t1 - t0:.2f}')
        return upscaled
