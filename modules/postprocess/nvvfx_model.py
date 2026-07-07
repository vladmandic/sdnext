import torch
import numpy as np
from PIL import Image
from modules import shared, devices
from modules.logger import log
from modules.upscaler import Upscaler, UpscalerData


class UpscalerDiffusion(Upscaler):
    def __init__(self, dirname): # pylint: disable=super-init-not-called
        self.name = "nVidia VFX"
        self.user_path = dirname
        """
        self.scalers = [
            UpscalerData(name="nVidia VFX 1x Denoise Ultra", path="", upscaler=self, model=None, scale=1),
            UpscalerData(name="nVidia VFX 1x Deblur Ultra", path="", upscaler=self, model=None, scale=1),
            UpscalerData(name="nVidia VFX 1x Denoise High", path="", upscaler=self, model=None, scale=1),
            UpscalerData(name="nVidia VFX 1x Deblur High", path="", upscaler=self, model=None, scale=1),
            UpscalerData(name="nVidia VFX 2x Ultra", path="", upscaler=self, model=None, scale=2),
            UpscalerData(name="nVidia VFX 4x Ultra", path="", upscaler=self, model=None, scale=4),
            UpscalerData(name="nVidia VFX 2x High", path="", upscaler=self, model=None, scale=2),
            UpscalerData(name="nVidia VFX 4x High", path="", upscaler=self, model=None, scale=4),
        ]
        """
        self.scalers = []
        self.models = {}

    def load_model(self, path: str):
        scaler: UpscalerData = [x for x in self.scalers if x.data_path == path or x.name == path]
        if len(scaler) == 0:
            log.error(f"Upscaler cannot match model: type={self.name} model={path}")
            return None
        scaler = scaler[0]
        if self.models.get(path, None) is not None:
            log.debug(f"Upscaler cached: type={scaler.name} model={path}")
            return self.models[path]
        from installer import install
        install('nvidia-vfx')

    def callback(self, _step: int, _timestep: int, _latents: torch.FloatTensor):
        pass

    def do_upscale(self, img: Image.Image, selected_model):
        devices.torch_gc()
        self.load_model(selected_model)

        frame = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().to(devices.device) / 255.0
        frame = frame.to(devices.device)

        try:
            import nvvfx
        except Exception as e:
            log.error(f"Upscaler: failed to import nvvfx: {e}")
            return img

        config_map = {
            "nVidia VFX 1x Denoise Ultra": nvvfx.VideoSuperRes.QualityLevel.DENOISE_ULTRA,
            "nVidia VFX 1x Deblur Ultra": nvvfx.VideoSuperRes.QualityLevel.DEBLUR_ULTRA,
            "nVidia VFX 1x Denoise High": nvvfx.VideoSuperRes.QualityLevel.DENOISE_HIGH,
            "nVidia VFX 1x Deblur High": nvvfx.VideoSuperRes.QualityLevel.DEBLUR_HIGH,
            "nVidia VFX 2x Ultra": nvvfx.VideoSuperRes.QualityLevel.ULTRA,
            "nVidia VFX 4x Ultra": nvvfx.VideoSuperRes.QualityLevel.ULTRA,
            "nVidia VFX 2x High": nvvfx.VideoSuperRes.QualityLevel.HIGH,
            "nVidia VFX 4x High": nvvfx.VideoSuperRes.QualityLevel.HIGH,
        }
        quality = config_map.get(selected_model, None)
        log.info(f'Upscaler: type="{self.name}" model="{selected_model}" version={nvvfx.__version__} sdk={nvvfx.get_sdk_version()} quality={quality}')
        if self.models.get(selected_model, None) is not None:
            vsr = self.models[selected_model]
        else:
            vsr = nvvfx.VideoSuperRes(quality=quality)
            self.models[selected_model] = vsr
        if '2x' in selected_model:
            vsr.output_width = img.width * 2
            vsr.output_height = img.height * 2
        elif '4x' in selected_model:
            vsr.output_width = img.width * 4
            vsr.output_height = img.height * 4
        elif 'Denoise' in selected_model or 'Deblur' in selected_model or '1x' in selected_model:
            vsr.output_width = img.width
            vsr.output_height = img.height
        else:
            log.error(f"Upscaler: unknown model: {selected_model}")
            return img
        vsr.input_width = img.width
        vsr.input_height = img.height

        log.debug(f"Upscaler: {vsr}")
        try:
            vsr.load()
        except Exception as e:
            log.error(f"Upscaler: failed to load model: {selected_model} error={e}")
            return img
        self.models[selected_model] = vsr

        result = vsr.run(frame)
        result = torch.from_dlpack(result.image).clone()
        image = Image.fromarray((result.permute(1, 2, 0).contiguous().cpu().numpy() * 255).astype(np.uint8))

        if shared.opts.upscaler_unload and selected_model in self.models:
            del self.models[selected_model]
            log.debug(f"Upscaler unloaded: type={self.name} model={selected_model}")
            devices.torch_gc(force=True)
        return image
