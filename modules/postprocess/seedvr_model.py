import time
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from modules import devices
from modules.shared import opts, log
from modules.upscaler import Upscaler, UpscalerData


MODELS_MAP = {
    "SeedVR2 3B": "seedvr2_ema_3b_fp16.safetensors",
    "SeedVR2 7B": "seedvr2_ema_7b_fp16.safetensors",
    "SeedVR2 7B Sharp": "seedvr2_ema_7b_sharp_fp16.safetensors",
}
to_pil = ToPILImage()


class UpscalerSeedVR(Upscaler):
    def __init__(self, dirname=None):
        self.name = "SeedVR"
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
            log.debug(f'Upscaler load: name="{self.name}" model="{model_name}"')
            from modules.seedvr.src.core.model_manager import configure_runner
            self.model = configure_runner(
                model_name=model_name,
                cache_dir=opts.hfcache_dir,
                device=devices.device,
                dtype=devices.dtype,
            )

    def do_upscale(self, img: Image.Image, selected_file):
        devices.torch_gc()
        self.load_model(selected_file)
        if self.model is None:
            return img

        from modules.seedvr.src.core.generation import generation_loop

        width = int(self.scale * img.width) // 8 * 8
        image_tensor = np.array(img)
        image_tensor = torch.from_numpy(image_tensor).to(device=devices.device, dtype=devices.dtype).unsqueeze(0) / 255.0

        t0 = time.time()
        result_tensor = generation_loop(
            runner=self.model,
            images=image_tensor,
            cfg_scale=1.0,
            seed=42,
            res_w=width,
            batch_size=1,
            temporal_overlap=0,
            device=devices.device,
        )
        t1 = time.time()
        log.info(f'Upscaler: type="{self.name}" model="{selected_file}" scale={self.scale} time={t1 - t0:.2f}')
        img = to_pil(result_tensor.squeeze().permute((2, 0, 1)))
        devices.torch_gc()

        if opts.upscaler_unload:
            self.model = None
            log.debug(f'Upscaler unload: type="{self.name}" model="{selected_file}"')
            devices.torch_gc(force=True)
        return img
