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
        import torchvision.transforms.functional as F
        import diffusers
        from modules import shared, devices
        if self.vae is None or selected_model != self.selected:
            if 'v1' in selected_model:
                repo_id = 'Heasterian/AsymmetricAutoencoderKLUpscaler'
            else:
                repo_id = 'Heasterian/AsymmetricAutoencoderKLUpscaler_v2'
            self.vae = diffusers.AsymmetricAutoencoderKL.from_pretrained(repo_id, cache_dir=shared.opts.hfcache_dir)
            shared.log.debug(f'Upscaler load: vae="{repo_id}"')
            self.vae.requires_grad_(False)
            self.vae = self.vae.to(device=devices.device, dtype=devices.dtype)
            self.vae.eval()
        img = img.resize((8 * (img.width // 8), 8 * (img.height // 8)), resample=Image.Resampling.LANCZOS).convert('RGB')
        tensor = (F.pil_to_tensor(img).unsqueeze(0) / 255.0).to(device=devices.device, dtype=devices.dtype)
        self.vae = self.vae.to(device=devices.device)
        tensor = self.vae(tensor).sample
        upscaled = F.to_pil_image(tensor.squeeze().clamp(0.0, 1.0).float().cpu())
        self.vae = self.vae.to(device=devices.cpu)
        return upscaled


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
        normalized = np.array(img).astype(np.float32) / 255.0
        scale = math.ceil(self.scale)
        upscaled = DCC(normalized, scale)
        upscaled = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min())
        upscaled = (255.0 * upscaled).astype(np.uint8)
        upscaled = Image.fromarray(upscaled)
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
        vips_image = pyvips.Image.new_from_array(img)
        # import numpy as np
        # np_image = np.array(img)
        # h, w, c = np_image.shape
        # np_linear = np_image.reshape(w * h * c)
        # vips_image = pyvips.Image.new_from_memory(np_linear.data, w, h, c, 'uchar')
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
        return upscaled
