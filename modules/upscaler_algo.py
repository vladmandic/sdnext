import time
from PIL import Image
from modules.upscaler import Upscaler, UpscalerData
from modules.shared import log


class UpscalerDCC(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "DCC Interpolation"
        self.vae = None
        self.scalers = [
            UpscalerData("DCC Interpolation", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import math
        import numpy as np
        from modules.postprocess.dcc import DCC
        t0 = time.time()
        normalized = np.array(img).astype(np.float32) / 255.0
        scale = math.ceil(self.scale)
        upscaled = DCC(normalized, scale)
        upscaled = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min())
        upscaled = (255.0 * upscaled).astype(np.uint8)
        upscaled = Image.fromarray(upscaled)
        t1 = time.time()
        log.debug(f"Upscale: name=DCC input={img.size} output={upscaled.size} time={t1 - t0:.2f}")
        return upscaled


class UpscalerVIPS(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "VIPS"
        self.scalers = [
            UpscalerData("VIPS Lanczos 2", None, self),
            UpscalerData("VIPS Lanczos 3", None, self),
            UpscalerData("VIPS Mitchell", None, self),
            UpscalerData("VIPS MagicKernelSharp 2013", None, self),
            UpscalerData("VIPS MagicKernelSharp 2021", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        if selected_model is None:
            return img
        from installer import install
        install('pyvips')
        try:
            import pyvips
        except Exception as e:
            log.error(f"Upscaler: vips {e}")
            return img
        t0 = time.time()
        vips_image = pyvips.Image.new_from_array(img)
        try:
            if selected_model is None:
                return img
            elif selected_model == "VIPS Lanczos 2":
                vips_image = vips_image.resize(2, kernel='lanczos2')
            elif selected_model == "VIPS Lanczos 3":
                vips_image = vips_image.resize(2, kernel='lanczos3')
            elif selected_model == "VIPS Mitchell":
                vips_image = vips_image.resize(2, kernel='mitchell')
            elif selected_model == "VIPS MagicKernelSharp 2013":
                vips_image = vips_image.resize(2, kernel='mks2013')
            elif selected_model == "VIPS MagicKernelSharp 2021":
                vips_image = vips_image.resize(2, kernel='mks2021')
            else:
                return img
        except Exception as e:
            log.error(f"Upscaler: vips {e}")
            return img
        upscaled = Image.fromarray(vips_image.numpy())
        t1 = time.time()
        log.debug(f"Upscale: name=VIPS input={img.size} output={upscaled.size} time={t1 - t0:.2f}")
        return upscaled

class UpscalerHQX(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "HQX"
        self.scalers = [
            UpscalerData("HQX Interpolation", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import numpy as np
        from modules.postprocess.hqx import hqx
        t0 = time.time()
        np_img = np.array(img).astype(np.uint32)
        upscaled = hqx(np_img, 2)
        upscaled = (upscaled).astype(np.uint8)
        upscaled = Image.fromarray(upscaled)
        t1 = time.time()
        log.debug(f"Upscale: name=HQX input={img.size} output={upscaled.size} time={t1 - t0:.2f}")
        return upscaled

class UpscalerICBI(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "ICB"
        self.scalers = [
            UpscalerData("ICB Interpolation", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import numpy as np
        from modules.postprocess.icbi import icbi
        t0 = time.time()
        np_img = np.array(img)
        upscaled = icbi(np_img)
        upscaled = Image.fromarray(upscaled)
        t1 = time.time()
        log.debug(f"Upscale: name=ICB input={img.size} output={upscaled.size} time={t1 - t0:.2f}")
        return upscaled
