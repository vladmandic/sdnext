import os
import time
import numpy as np
import torch
from PIL import Image
from modules.upscaler import Upscaler, UpscalerData
from modules import devices, shared, errors
from modules.logger import log


class UpscalerNVVFX(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "nVidia VFX"
        self.scalers = [
            UpscalerData("nVidia VFX bicubic", None, self, scale=0),
            UpscalerData("nVidia VFX low", None, self, scale=1),
            UpscalerData("nVidia VFX medium", None, self, scale=2),
            UpscalerData("nVidia VFX high", None, self, scale=3),
            UpscalerData("nVidia VFX ultra", None, self, scale=4),
            UpscalerData("nVidia VFX denoise low", None, self, scale=8),
            UpscalerData("nVidia VFX denoise medium", None, self, scale=9),
            UpscalerData("nVidia VFX denoise high", None, self, scale=10),
            UpscalerData("nVidia VFX denoise ultra", None, self, scale=11),
            UpscalerData("nVidia VFX deblur low", None, self, scale=12),
            UpscalerData("nVidia VFX deblur medium", None, self, scale=13),
            UpscalerData("nVidia VFX deblur high", None, self, scale=14),
            UpscalerData("nVidia VFX deblur ultra", None, self, scale=15),
            UpscalerData("nVidia VFX highbitrate low", None, self, scale=16),
            UpscalerData("nVidia VFX highbitrate medium", None, self, scale=17),
            UpscalerData("nVidia VFX highbitrate high", None, self, scale=18),
            UpscalerData("nVidia VFX highbitrate ultra", None, self, scale=19),
        ]

    # nvvfx overrides upscale instead of do_upscale because it handles scale directly
    def upscale(self, img: Image.Image | torch.Tensor, scale, selected_model: str | None = None):
        if selected_model is None:
            return img
        from installer import install
        install('nvidia-vfx')
        os.environ["NV_VFX_LOG_LEVEL"] = "4"
        os.environ["NV_VFX_DEBUG"] = "1"
        try:
            import nvvfx
        except Exception as e:
            log.error(f"Upscaler: nvvfx {e}")
            errors.display(e, "Upscaler: nvvfx error")
            return img

        jobid = shared.state.begin('Upscale')
        try:
            t0 = time.time()
            upscaler = self.find_model(selected_model)

            quality = nvvfx.VideoSuperRes.QualityLevel(upscaler.scale)
            vsr = nvvfx.VideoSuperRes(quality=quality)
            vsr.input_width = img.width
            vsr.input_height = img.height
            _scale = 1.0 if 'DEBLUR' in quality.name or 'DENOISE' in quality.name else scale
            vsr.output_width = int(img.width * _scale)
            vsr.output_height = int(img.height * _scale)
            log.debug(f'Upscaler: id={upscaler.scale} scale={_scale} version={nvvfx.__version__} sdk={nvvfx.get_sdk_version()} vsr={vsr}')
            vsr.load()

            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().contiguous().to(devices.device) / 255.0
            result = vsr.run(tensor)
            tensor = torch.from_dlpack(result.image).clone()
            tensor = 255.0 * tensor.permute(1, 2, 0).contiguous().cpu()
            upscaled = Image.fromarray(tensor.numpy().astype(np.uint8))

            vsr.close()
            t1 = time.time()
            log.debug(f'Upscale: name="{selected_model}" input={img.size} output={upscaled.size} time={t1 - t0:.2f}')
        except nvvfx.NvVFXError as e:
            log.error(f"Upscaler: nvvfx {e}")
            errors.display(e, "Upscaler: nvvfx error")
            upscaled = img

        shared.state.end(jobid)
        return upscaled
