from PIL import Image
from modules.upscaler import Upscaler, UpscalerData
from modules.shared import log


class UpscalerNone(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "None"
        self.scalers = [UpscalerData("None", None, self)]

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img


class UpscalerResize(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Resize"
        self.scalers = [
            UpscalerData("Resize Nearest", None, self),
            UpscalerData("Resize Lanczos", None, self),
            UpscalerData("Resize Bicubic", None, self),
            UpscalerData("Resize Bilinear", None, self),
            UpscalerData("Resize Hamming", None, self),
            UpscalerData("Resize Box", None, self),
            UpscalerData("Resize Sharpfin MKS2021", None, self),
            UpscalerData("Resize Sharpfin Lanczos3", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        if selected_model is None:
            return img
        elif selected_model == "Resize Nearest":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.NEAREST)
        elif selected_model == "Resize Lanczos":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.LANCZOS)
        elif selected_model == "Resize Bicubic":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.BICUBIC)
        elif selected_model == "Resize Bilinear":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.BILINEAR)
        elif selected_model == "Resize Hamming":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.HAMMING)
        elif selected_model == "Resize Box":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.BOX)
        elif selected_model == "Resize Sharpfin MKS2021":
            from modules.image import sharpfin
            return sharpfin.resize(img, (int(img.width * self.scale), int(img.height * self.scale)), kernel="Sharpfin MKS2021")
        elif selected_model == "Resize Sharpfin Lanczos3":
            from modules.image import sharpfin
            return sharpfin.resize(img, (int(img.width * self.scale), int(img.height * self.scale)), kernel="Sharpfin Lanczos3")
        else:
            return img


    def load_model(self, _):
        pass


class UpscalerLatent(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Latent"
        self.scalers = [
            UpscalerData("Latent Nearest", None, self),
            UpscalerData("Latent Nearest exact", None, self),
            UpscalerData("Latent Area", None, self),
            UpscalerData("Latent Bilinear", None, self),
            UpscalerData("Latent Bicubic", None, self),
            UpscalerData("Latent Bilinear antialias", None, self),
            UpscalerData("Latent Bicubic antialias", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import torch
        import torch.nn.functional as F
        if isinstance(img, torch.Tensor) and (len(img.shape) == 4):
            _batch, _channel, h, w = img.shape
        else:
            log.error(f"Upscale: type=latent image={img.shape if isinstance(img, torch.Tensor) else img} type={type(img)} if not supported")
            return img
        h, w = int((8 * h * self.scale) // 8), int((8 * w * self.scale) // 8)
        mode, antialias = '', ''
        if selected_model == "Latent Nearest":
            mode, antialias = 'nearest', False
        elif selected_model == "Latent Nearest exact":
            mode, antialias = 'nearest-exact', False
        elif selected_model == "Latent Area":
            mode, antialias = 'area', False
        elif selected_model == "Latent Bilinear":
            mode, antialias = 'bilinear', False
        elif selected_model == "Latent Bicubic":
            mode, antialias = 'bicubic', False
        elif selected_model == "Latent Bilinear antialias":
            mode, antialias = 'bilinear', True
        elif selected_model == "Latent Bicubic antialias":
            mode, antialias = 'bicubic', True
        else:
            raise log.error(f"Upscale: type=latent model={selected_model} unknown")
        return F.interpolate(img, size=(h, w), mode=mode, antialias=antialias)
