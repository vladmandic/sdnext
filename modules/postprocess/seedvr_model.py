import time
import os
import random
import numpy as np
import torch
from PIL import Image
from modules import devices, timer
from modules.shared import opts
from modules.upscaler import Upscaler, UpscalerData
from modules.image import convert
from modules.model_quant import do_post_load_quant
from modules.logger import log, console


MODELS_MAP = {
    "SeedVR2 3B": "seedvr2_ema_3b_fp16.safetensors",
    "SeedVR2 7B": "seedvr2_ema_7b_fp16.safetensors",
    "SeedVR2 7B Sharp": "seedvr2_ema_7b_sharp_fp16.safetensors",
}


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
        self.device = devices.device
        self.step = 1
        self.frames = 0
        self.offload = True
        self.pbar = None
        self.task = None
        self.fps = 24
        self.timer = None

    def set_vae_params(self, vae_memory: float, tile_size: int, tile_overlap: float, vae_tile_encode: bool = True, vae_tile_decode: bool = True):
        if vae_memory >= 0.99:
            vae_memory = None
        self.model.config.vae.memory_limit = {'conv_max_mem': vae_memory, 'norm_max_mem': vae_memory}
        self.model.vae.set_memory_limit(**self.model.config.vae.memory_limit)
        self.model.vae.tile_sample_min_size = tile_size
        self.model.vae.tile_latent_min_size = tile_size // 8
        self.model.vae.tile_overlap_factor = tile_overlap
        if vae_tile_encode:
            self.model.vae.use_slicing_encode = False
            self.model.vae.use_tiling_encode = True
        else:
            self.model.vae.use_slicing_encode = True
            self.model.vae.use_tiling_encode = False
        if vae_tile_decode:
            self.model.vae.use_slicing_decode = False
            self.model.vae.use_tiling_decode = True
        else:
            self.model.vae.use_slicing_decode = True
            self.model.vae.use_tiling_decode = False

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
            if not hasattr(generation, "generation_step_original"): # Patch generation_loop's generation_step() with our wrapper; stash the original once so reloads don't re-wrap the wrapper itself (infinite recursion).
                generation.generation_step_original = generation.generation_step
            generation.generation_step = self.model_step
            self.model._internal_dict = {
                'dit': self.model.dit,
                'vae': self.model.vae,
            }
            self.model.dit.config = self.model.config.dit
            self.model = do_post_load_quant(self.model, allow=True)
            t1 = time.time()
            log.info(f'Upscaler loaded: name="{self.name}" model="{model_name}" time={t1 - t0:.2f}')

    def vae_encode(self, samples):
        latents = []
        if len(samples) == 0:
            return latents
        self.pbar.update(self.task, description=f'encode: images={list(samples[0].shape) if len(samples) > 0 else None}')
        if self.offload:
            t0 = time.time()
            self.model.dit = self.model.dit.to(device="cpu")
            self.model.vae = self.model.vae.to(device=self.device)
            self.timer.ts('offload', t0)
        devices.torch_gc(fast=True)
        t0 = time.time()
        from einops import rearrange
        scale = self.model.config.vae.scaling_factor
        shift = self.model.config.vae.get("shifting_factor", 0.0)
        batches = [sample.unsqueeze(0) for sample in samples]
        with devices.inference_context():
            for sample in batches:
                sample = sample.to(self.device, self.model.vae.dtype)
                sample = self.model.vae.preprocess(sample)
                latent = self.model.vae.encode(sample).latent
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                latents.append(latent.contiguous())
        latents = [latent.squeeze(0) for latent in latents]
        self.timer.ts('encode', t0)
        if self.offload:
            t0 = time.time()
            self.model.vae = self.model.vae.to(device="cpu")
            self.timer.ts('offload', t0)
        devices.torch_gc(fast=True)
        return latents

    def vae_decode(self, latents, target_dtype: torch.dtype = None):
        self.pbar.update(self.task, description=f'decode: latents={list(latents[0].shape) if len(latents) > 0 else None}')
        samples = []
        if len(latents) == 0:
            return samples
        from einops import rearrange
        if self.offload:
            t0 = time.time()
            self.model.dit = self.model.dit.to(device="cpu")
            self.model.vae = self.model.vae.to(device=self.device)
            self.timer.ts('offload', t0)
        devices.torch_gc(fast=True)
        t0 = time.time()
        scale = self.model.config.vae.scaling_factor
        shift = self.model.config.vae.get("shifting_factor", 0.0)
        latents = [latent.unsqueeze(0) for latent in latents]
        with devices.inference_context():
            for _i, latent in enumerate(latents):
                latent = latent.to(self.device, self.model.vae.dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2).contiguous()
                sample = self.model.vae.decode(latent).sample
                sample = self.model.vae.postprocess(sample)
                samples.append(sample.squeeze(0).contiguous())
        self.timer.ts('decode', t0)
        if self.offload:
            t0 = time.time()
            self.model.vae = self.model.vae.to(device="cpu")
            self.timer.ts('offload', t0)
        devices.torch_gc(fast=True)
        return samples

    def model_step(self, *args, **kwargs):
        from modules.shared import state
        if state.interrupted or state.skipped:
            return None
        from modules.seedvr.src.core import generation
        if self.offload:
            t0 = time.time()
            self.model.vae = self.model.vae.to(device="cpu")
            self.model.dit = self.model.dit.to(device=self.device)
            self.timer.ts('offload', t0)
        devices.torch_gc(fast=True)
        t0 = time.time()
        with devices.inference_context():
            self.pbar.update(self.task, description=f'inference: batch={self.step}')
            result = generation.generation_step_original(*args, **kwargs)
            self.pbar.update(self.task, advance=self.step)
        self.timer.ts('step', t0)
        if self.offload:
            t0 = time.time()
            self.model.dit = self.model.dit.to(device="cpu")
            self.timer.ts('offload', t0)
        devices.torch_gc(fast=True)
        return result

    def read_image(self, image: str | Image.Image):
        try:
            if isinstance(image, str):
                image = Image.open(image)
            image = image.convert("RGB")
            width = image.width
            tensor = np.array(image)
            tensor = torch.from_numpy(tensor).to(device=devices.device, dtype=devices.dtype).unsqueeze(0) / 255.0
            self.frames = 1
            return tensor, width
        except Exception as e:
            log.error(f'Upscaler: name="SeedVR2" image="{image}" {e}')
            return None, None

    def read_video(self, video_path: str):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log.error(f'Upscaler: name="SeedVR2" video="{video_path}" failed to open')
                return None, None
            frames = []
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if len(frames) == 0:
                log.error(f'Upscaler: name="SeedVR2" video="{video_path}" no frames read')
                return None, None
            tensor = torch.from_numpy(np.array(frames)).to(device=devices.device, dtype=devices.dtype) / 255.0
            self.frames = tensor.shape[0]
            return tensor, width
        except Exception as e:
            log.error(f'Upscaler: name="SeedVR2" video="{video_path}" {e}')
            return None, None

    def create_video(self, tensor: torch.Tensor, codec: str = 'libx264', codec_opt: str = 'crf:16', interpolate: int = 0):
        t0 = time.time()
        from modules.video_models.video_save import save_video
        pixels = tensor.permute(3, 0, 1, 2).unsqueeze(0) # from (t, h, w, c) to (n, c, t, h, w)
        _frames, filename, _thumb = save_video(p=None,
                                               pixels=pixels,
                                               mp4_fps=self.fps,
                                               mp4_thumb=False,
                                               mp4_frames=False,
                                               reclamp=False,
                                               mp4_codec=codec,
                                               mp4_opt=codec_opt,
                                               mp4_interpolate=interpolate,
                                              )
        self.timer.ts('save', t0)
        return filename

    def do_upscale(self,
                   img: Image.Image | str,
                   selected_file,
                   cfg_scale: float = 1.5,
                   cfg_rescale: float = 0.0,
                   steps: int = 1,
                   seed: int = -1,
                   scale: float | None = None,
                   tile_size: int = 1024,
                   tile_overlap: float = 0.25,
                   batch_size: int = 1,
                   batch_overlap: int = 0,
                   offload: bool = True,
                   interpolate: int = 1,
                   codec: str = 'libx264',
                   codec_opt: str = 'crf:16',
                   vae_memory: float = 0.2,
                   vae_tile_encode: bool = True,
                   vae_tile_decode: bool = True,
                  ):
        self.timer = timer.Timer()
        self.offload = offload
        self.load_model(selected_file)
        self.set_vae_params(vae_memory=vae_memory, tile_size=tile_size, tile_overlap=tile_overlap, vae_tile_encode=vae_tile_encode, vae_tile_decode=vae_tile_decode)
        if self.model is None:
            return img
        if not self.offload:
            self.model.dit = self.model.dit.to(device=devices.device)
            self.model.vae = self.model.vae.to(device=devices.device)
            devices.torch_gc(fast=True)
        self.timer.record('load')

        from modules.seedvr.src.core import generation

        self.scale = self.scale if scale is None else scale
        if isinstance(img, Image.Image):
            tensor, width = self.read_image(img)
        elif isinstance(img, str):
            tensor, width = self.read_video(img)
        else:
            log.error(f'Upscaler: name="SeedVR2" image="{img}" unsupported type {type(img)}')
            return img
        self.timer.record('read')

        if tensor is None or width is None:
            log.error(f'Upscaler: name="SeedVR2" image="{img}" failed to read')
            return img
        width = int(self.scale * width) // 8 * 8
        random.seed()
        seed = int(random.randrange(4294967294)) if seed == -1 else int(seed)
        self.step = 1 if self.frames == 1 else batch_size - batch_overlap

        mode = "mode=image" if self.frames == 1 else f"mode=video frames={self.frames}"
        batch_info = f'batch=(size={batch_size} overlap={batch_overlap})'
        vae_info = f'vae=(tiled={vae_tile_encode}/{vae_tile_decode} memory={vae_memory} size={tile_size} overlap={tile_overlap})'
        log.info(f'Upscaler: type="{self.name}" model="{selected_file}" {mode} scale={self.scale} cfg={cfg_scale}:{cfg_rescale} seed={seed} steps={steps} offload={self.offload} {batch_info} {vae_info}')

        import rich.progress as rp
        self.pbar = rp.Progress(rp.TextColumn('[cyan]SeedVR:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=console)
        self.task = self.pbar.add_task(total=self.frames, description='starting...')
        with devices.inference_context(), self.pbar:
            self.pbar.update(self.task, description='initialize rope')
            from modules.seedvr.src.optimization import memory_manager
            memory_manager.clear_rope_cache(self.model)
            memory_manager.preinitialize_rope_cache(self.model)
            self.timer.record('init')
            result_tensor = generation.generation_loop(
                runner=self.model,
                images=tensor,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                steps=steps, # TODO SeedVR steps
                batch_size=batch_size, # TODO SeedVR batch size
                temporal_overlap=batch_overlap, # TODO SeedVR temporal overlap
                seed=seed,
                res_w=width,
                device=devices.device,
                color_reconstruct=True,
            )
            memory_manager.clear_rope_cache(self.model)

        self.pbar.update(self.task, completed=self.frames)
        t1 = time.time()
        self.frames = result_tensor.shape[0] if result_tensor is not None else 0
        self.timer.add('inference', self.timer.get('step') - self.timer.get('encode') - self.timer.get('decode'))
        self.timer.rm('step')

        t0 = time.time()
        self.model.dit = self.model.dit.to(device="cpu")
        self.model.vae = self.model.vae.to(device="cpu")
        self.timer.ts('offload', t0)
        if opts.upscaler_unload:
            self.model.dit = None
            self.model.vae = None
            self.model.cache = None
            self.model = None
            log.debug(f'Upscaler unload: type="{self.name}" model="{selected_file}"')
        devices.torch_gc(force=True)
        self.timer.ts('cleanup', t1)

        if self.frames == 1:
            result = convert.to_pil(result_tensor.squeeze())
        elif self.frames > 1:
            result = self.create_video(result_tensor, codec=codec, codec_opt=codec_opt, interpolate=interpolate)
        else:
            log.error(f'Upscaler: name="SeedVR2" model="{selected_file}" no frames generated')
            result = img
        log.info(f'Upscaler: type="{self.name}" model="{selected_file}" frames={self.frames} {self.timer.summary()}')
        return result
