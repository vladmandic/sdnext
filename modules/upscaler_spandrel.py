import os
import time
import torch
import numpy as np
from PIL import Image
from modules.upscaler import Upscaler, UpscalerData
from modules import devices, paths, errors
from modules.logger import log


MODELS = {
    "Spandrel 4x RealPLKSR NomosWebPhoto": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/4xNomosWebPhoto_RealPLKSR.safetensors",
    "Spandrel 2x RealPLKSR AnimeSharpV2": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/2x-AnimeSharpV2_RPLKSR_Sharp.pth",
    "Spandrel 2x RealESRGAN Compact": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RealESRGAN-2x-Compact.pth",
    "Spandrel 2x RealESRGAN UltraCompact": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/RealESRGAN-2x-UltraCompact.pth",
    "Spandrel 2x RealSAFMN++": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/Real-SAFMN++.pth",
    "Spandrel 2x RealSAFMN": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/Real-SAFMN-x2.pth",
    "Spandrel 4x RealSAFMN": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/Real-SAFMN-x4-v2.pth",
    "Spandrel 2x SAFMN PureScale": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/2x_SAFMN_PureScale.pth",
    "Spandrel 2x SAFMN PureScale Sharper": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/2x_SAFMN_PureScale_sharper.pth",
    "Spandrel 4x SAFMN PureScale": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/4x_SAFMN_PureScale.pth",
}

class UpscalerSpandrel(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Spandrel"
        self.model_path = os.path.join(paths.models_path, 'Spandrel')
        self.user_path = os.path.join(paths.models_path, 'Spandrel')
        self.selected = None
        self.model = None
        self.scalers = self.find_scalers()
        found = [os.path.basename(s.data_path) for s in self.scalers]
        for k, v in MODELS.items():
            fn = os.path.basename(v)
            if fn not in found:
                scaler = UpscalerData(name=k, path=v, upscaler=self)
                self.scalers.append(scaler)
            else:
                for s in self.scalers: # update name of existing scaler if it was found
                    if os.path.basename(s.data_path) == fn:
                        s.name = k
                        break

    def process(self, img: Image.Image, output_type='pil', quiet=False) -> Image.Image:
        if isinstance(img, Image.Image):
            from modules.image import convert
            img = img.convert('RGB')
            tensor = convert.to_tensor(img).unsqueeze(0).to(devices.device)
        elif isinstance(img, np.ndarray):
            from modules.image import convert
            tensor = convert.to_tensor(img).unsqueeze(0).to(devices.device)
        elif isinstance(img, torch.Tensor):
            tensor = img.to(devices.device).float()
        else:
            log.error(f'Spandrel: unsupported input type={type(img)}')
            return img
        t0 = time.time()
        with devices.inference_context():
            if tensor.max() > 1.0: # tensor is in [0,255] range, convert to [0,1]
                tensor = tensor.div_(255.0)
            if tensor.min() < 0: # tensor is in [-1,1] range, convert to [0,1]
                tensor = (tensor + 1.0) / 2.0
            tensor = self.model(tensor)
            tensor = tensor.clamp(0, 1).squeeze(0).cpu()
        t1 = time.time()
        if output_type == 'pil':
            upscaled = convert.to_pil(tensor)
            if not quiet:
                log.debug(f'Upscale: name="{self.selected}" input={img.size} type={output_type} output={upscaled.size} time={t1 - t0:.3f}')
        elif output_type == 'nd':
            upscaled = (255.0 * tensor).float().numpy().astype(np.uint8)
            if not quiet:
                log.debug(f'Upscale: name="{self.selected}" input={img.shape} type={output_type} output={upscaled.shape} time={t1 - t0:.3f}')
        elif output_type == 'tensor':
            upscaled = tensor
            if not quiet:
                log.debug(f'Upscale: name="{self.selected}" input={img.shape} type={output_type} output={list(upscaled.shape)} time={t1 - t0:.3f}')
        else:
            upscaled = img
            log.error(f'Upscale: type={output_type} unsupported')
        return upscaled

    def load_model(self, path: str):
        from installer import install
        if path is None:
            return
        install('spandrel')
        import spandrel
        self.selected = path
        model = self.find_model(path)
        self.model = spandrel.ModelLoader().load_from_file(model.local_data_path)
        self.model.to(devices.device).eval()

    def do_upscale(self, img: Image.Image | torch.Tensor | np.ndarray, selected_model: str | None = None, output_type='pil', quiet=False):
        try:
            if (self.model is None) or (self.selected != selected_model):
                self.load_model(selected_model)
            return self.process(img, output_type=output_type, quiet=quiet)
        except Exception as e:
            log.error(f'Spandrel: {e}')
            errors.display(e, "Spandrel")
            return img
