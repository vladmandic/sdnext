import time
import random
import numpy as np
import torch
from PIL import Image
from modules import devices, images_sharpfin
from modules.shared import opts, log
from modules.upscaler import Upscaler, UpscalerData


MODELS_MAP = {
    "SeedVR2 3B": "seedvr2_ema_3b_fp16.safetensors",
    "SeedVR2 7B": "seedvr2_ema_7b_fp16.safetensors",
    "SeedVR2 7B Sharp": "seedvr2_ema_7b_sharp_fp16.safetensors",
}
to_pil = images_sharpfin.to_pil


class UpscalerSeedVR(Upscaler):
    def __init__(self, dirname=None):
        self.name = "SeedVR2"
        super().__init__()
        self.scalers = [
            UpscalerData(name="SeedVR2 3B", path=None, upscaler=self, model=None, scale=1),
            UpscalerData(name="SeedVR2 7B", path=None, upscaler=self, model=None, scale=1),
            UpscalerData(name="SeedVR2 7B Sharp", path=None, upscaler=self, model=None, scale=1),
        ]
        self.model = None
        self.model_loaded = None

    def load_model(self, path: str):
        model_name = MODELS_MAP.get(path, None)
        if (self.model is None) or (self.model_loaded != model_name):
            log.debug(f'Upscaler loading: name="{self.name}" model="{model_name}"')
            t0 = time.time()
            from modules.seedvr.src.core.model_manager import configure_runner
            from modules.seedvr.src.core import generation
            self.model = configure_runner(
                model_name=model_name,
                cache_dir=opts.hfcache_dir,
                device=devices.device,
                dtype=devices.dtype,
            )
            self.model_loaded = model_name
            self.model.dit.device = devices.device
            self.model.dit.dtype = devices.dtype
            self.model.vae_encode = self.vae_encode
            self.model.vae_decode = self.vae_decode
            self.model.model_step = generation.generation_step
            generation.generation_step = self.model_step
            self.model._internal_dict = {
                'dit': self.model.dit,
                'vae': self.model.vae,
            }
            t1 = time.time()
            self.model.dit.config = self.model.config.dit
            self.model.vae.tile_sample_min_size = 1024
            self.model.vae.tile_latent_min_size = 128
            from modules.model_quant import do_post_load_quant
            self.model = do_post_load_quant(self.model, allow=True)
            # from modules.sd_offload import set_diffuser_offload
            # set_diffuser_offload(self.model)
            log.info(f'Upscaler loaded: name="{self.name}" model="{model_name}" time={t1 - t0:.2f}')

    def vae_encode(self, samples):
        log.debug(f'Upscaler encode: samples={samples[0].shape if len(samples) > 0 else None} tile={self.model.vae.tile_sample_min_size} overlap={self.model.vae.tile_overlap_factor}')
        latents = []
        if len(samples) == 0:
            return latents
        self.model.dit = self.model.dit.to(device="cpu")
        self.model.vae = self.model.vae.to(device=self.device)
        devices.torch_gc()
        from einops import rearrange
        from modules.seedvr.src.optimization import memory_manager
        memory_manager.clear_rope_cache(self.model)
        scale = self.model.config.vae.scaling_factor
        shift = self.model.config.vae.get("shifting_factor", 0.0)
        batches = [sample.unsqueeze(0) for sample in samples]
        for sample in batches:
            sample = sample.to(self.device, self.model.vae.dtype)
            sample = self.model.vae.preprocess(sample)
            latent = self.model.vae.encode(sample).latent
            latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
            latent = rearrange(latent, "b c ... -> b ... c")
            latent = (latent - shift) * scale
            latents.append(latent)
        latents = [latent.squeeze(0) for latent in latents]
        self.model.vae = self.model.vae.to(device="cpu")
        devices.torch_gc()
        return latents

    def vae_decode(self, latents, target_dtype: torch.dtype = None):
        log.debug(f'Upscaler decode: latents={latents[0].shape if len(latents) > 0 else None} tile={self.model.vae.tile_latent_min_size} overlap={self.model.vae.tile_overlap_factor}')
        samples = []
        if len(latents) == 0:
            return samples
        from einops import rearrange
        from modules.seedvr.src.optimization import memory_manager
        memory_manager.clear_rope_cache(self.model)
        self.model.dit = self.model.dit.to(device="cpu")
        self.model.vae = self.model.vae.to(device=self.device)
        devices.torch_gc()
        scale = self.model.config.vae.scaling_factor
        shift = self.model.config.vae.get("shifting_factor", 0.0)
        latents = [latent.unsqueeze(0) for latent in latents]
        with devices.inference_context():
            for _i, latent in enumerate(latents):
                latent = latent.to(self.device, self.model.vae.dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2)
                sample = self.model.vae.decode(latent).sample
                sample = self.model.vae.postprocess(sample)
                samples.append(sample)
        samples = [sample.squeeze(0) for sample in samples]
        self.model.vae = self.model.vae.to(device="cpu")
        devices.torch_gc()
        return samples

    def model_step(self, *args, **kwargs):
        from modules.seedvr.src.optimization import memory_manager
        self.model.vae = self.model.vae.to(device="cpu")
        self.model.dit = self.model.dit.to(device=self.device)
        devices.torch_gc()
        log.debug(f'Upscaler inference: args={len(args)} kwargs={list(kwargs.keys())}')
        memory_manager.preinitialize_rope_cache(self.model)
        with devices.inference_context():
            result = self.model.model_step(*args, **kwargs)
        self.model.dit = self.model.dit.to(device="cpu")
        devices.torch_gc()
        return result

    def do_upscale(self, img: Image.Image, selected_file):
        self.load_model(selected_file)
        if self.model is None:
            return img

        from modules.seedvr.src.core import generation

        width = int(self.scale * img.width) // 8 * 8
        image_tensor = np.array(img)
        image_tensor = torch.from_numpy(image_tensor).to(device=devices.device, dtype=devices.dtype).unsqueeze(0) / 255.0

        random.seed()
        seed = int(random.randrange(4294967294))

        t0 = time.time()
        with devices.inference_context():
            result_tensor = generation.generation_loop(
                runner=self.model,
                images=image_tensor,
                cfg_scale=opts.seedvt_cfg_scale,
                seed=seed,
                res_w=width,
                batch_size=1,
                temporal_overlap=0,
                device=devices.device,
            )
        t1 = time.time()
        log.info(f'Upscaler: type="{self.name}" model="{selected_file}" scale={self.scale} cfg={opts.seedvt_cfg_scale} seed={seed} time={t1 - t0:.2f}')
        img = to_pil(result_tensor.squeeze())

        if opts.upscaler_unload:
            self.model.dit = None
            self.model.vae = None
            self.model.cache = None
            self.model = None
            log.debug(f'Upscaler unload: type="{self.name}" model="{selected_file}"')
        devices.torch_gc(force=True)
        return img
