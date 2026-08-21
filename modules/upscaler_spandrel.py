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
    "Spandrel 2x RealSAFMN++": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/Real-SAFMN-x2.pth",
    "Spandrel 4x RealSAFMN++": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/Real-SAFMN-x4-v2.pth",
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

    def process(self, img: Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            from modules.image import convert
            img = img.convert('RGB')
            tensor = convert.to_tensor(img).unsqueeze(0).to(devices.device)
        elif isinstance(img, np.ndarray):
            from modules.image import convert
            tensor = convert.to_tensor(img).unsqueeze(0).to(devices.device)
        elif isinstance(img, torch.Tensor):
            tensor = img.to(devices.device)
        else:
            log.error(f'Spandrel: unsupported input type={type(img)}')
            return img
        t0 = time.time()
        with devices.inference_context():
            tensor = self.model(tensor)
            tensor = tensor.clamp(0, 1).squeeze(0).cpu()
        t1 = time.time()
        upscaled = convert.to_pil(tensor)
        log.debug(f'Upscale: name="{self.selected}" input={img.size} output={upscaled.size} time={t1 - t0:.3f}')
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

    def do_upscale(self, img: Image.Image | torch.Tensor | np.ndarray, selected_model=None):
        try:
            if (self.model is None) or (self.selected != selected_model):
                self.load_model(selected_model)
            return self.process(img)
        except Exception as e:
            log.error(f'Spandrel: {e}')
            errors.display(e, "Spandrel")
            return img
