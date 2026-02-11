import os
import time
from PIL import Image
from modules.upscaler import Upscaler, UpscalerData
from modules import devices, paths
from modules.shared import log


MODELS = {
    "Spandrel 4x RealPLKSR NomosWebPhoto": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/4xNomosWebPhoto_RealPLKSR.safetensors",
    "Spandrel 2x RealPLKSR AnimeSharpV2": "https://huggingface.co/vladmandic/sdnext-upscalers/resolve/main/2x-AnimeSharpV2_RPLKSR_Sharp.pth",
}

class UpscalerSpandrel(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Spandrel"
        self.model_path = os.path.join(paths.models_path, 'Spandrel')
        self.user_path = os.path.join(paths.models_path, 'Spandrel')
        self.selected = None
        self.model = None
        self.scalers = []
        for model_name, model_path in MODELS.items():
            scaler = UpscalerData(name=model_name, path=model_path, upscaler=self)
            self.scalers.append(scaler)

    def process(self, img: Image.Image) -> Image.Image:
        from modules import images_sharpfin
        tensor = images_sharpfin.to_tensor(img).unsqueeze(0).to(devices.device)
        img = img.convert('RGB')
        t0 = time.time()
        with devices.inference_context():
            tensor = self.model(tensor)
            tensor = tensor.clamp(0, 1).squeeze(0).cpu()
        t1 = time.time()
        upscaled = images_sharpfin.to_pil(tensor)
        log.debug(f'Upscale: name="{self.selected}" input={img.size} output={upscaled.size} time={t1 - t0:.2f}')
        return upscaled

    def do_upscale(self, img: Image, selected_model=None):
        from installer import install
        if selected_model is None:
            return img
        install('spandrel')
        try:
            import spandrel
            if (self.model is None) or (self.selected != selected_model):
                self.selected = selected_model
                model = self.find_model(selected_model)
                self.model = spandrel.ModelLoader().load_from_file(model.local_data_path)
                self.model.to(devices.device).eval()
            return self.process(img)
        except Exception as e:
            log.error(f'Spandrel: {e}')
            return img
