from __future__ import annotations
import os
import sys
import threading
from typing import TYPE_CHECKING, final
from modules import shared, errors

if TYPE_CHECKING:
    from diffusers import ModelMixin


@final
class ModelData:
    def __init__(self):
        self._sd_model: ModelMixin | None = None
        self._sd_model_type = "none"
        self._sd_refiner: ModelMixin | None = None
        self._sd_refiner_type = "none"
        self.sd_dict = 'None'
        self.initial = True
        self.locked = True
        self.lock = threading.Lock()

    @final
    @property
    def sd_loaded(self):
        return self._sd_model is not None

    @final
    @property
    def sd_loaded_refiner(self):
        return self._sd_refiner is not None

    @final
    @property
    def sd_model(self):
        if not self.sd_loaded:
            fn = f'{os.path.basename(sys._getframe(2).f_code.co_filename)}:{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            shared.log.debug(f'Model requested: fn={fn}') # pylint: disable=protected-access
        if self.locked:
            if self._sd_model is None:
                fn = f'{os.path.basename(sys._getframe(2).f_code.co_filename)}:{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
                shared.log.warning(f'Model locked: fn={fn}')
            return self._sd_model
        elif (self._sd_model is None) and (shared.opts.sd_model_checkpoint != 'None') and (not self.lock.locked()):
            with self.lock:
                try:
                    from modules.sd_models import reload_model_weights
                    self._sd_model = reload_model_weights(op='model') # note: reload_model_weights directly updates model_data.sd_model and returns it at the end
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self._sd_model = None
        return self._sd_model

    @final
    @sd_model.setter
    def sd_model(self, value):
        if not self.locked:
            self._sd_model = value
            self._sd_model_type = "none" if value is None else ModelData.get_model_type(value)

    @final
    @property
    def sd_refiner(self):
        if not self.sd_loaded_refiner and (shared.opts.sd_model_refiner != 'None') and (not self.lock.locked()):
            with self.lock:
                try:
                    from modules.sd_models import reload_model_weights
                    self._sd_refiner = reload_model_weights(op='refiner')
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self._sd_refiner = None
        return self._sd_refiner

    @final
    @sd_refiner.setter
    def sd_refiner(self, value):
        if not self.locked:
            self._sd_refiner = value
            self._sd_refiner_type = "none" if value is None else ModelData.get_model_type(value)

    @final
    @property
    def sd_model_type(self):
        return self._sd_model_type

    @final
    @property
    def sd_refiner_type(self):
        return self._sd_refiner_type

    @final
    @staticmethod
    def get_model_type(pipe: ModelMixin):
        name = pipe.__class__.__name__
        if not shared.native:
            model_type = 'ldm'
        elif "StableDiffusion3" in name:
            model_type = 'sd3'
        elif "StableDiffusionXL" in name:
            model_type = 'sdxl'
        elif "StableDiffusion" in name:
            model_type = 'sd'
        elif "StableVideoDiffusion" in name:
            model_type = 'svd'
        elif "LatentConsistencyModel" in name:
            model_type = 'sd' # lcm is compatible with sd
        elif "InstaFlowPipeline" in name:
            model_type = 'sd' # instaflow is compatible with sd
        elif "AnimateDiffPipeline" in name:
            model_type = 'sd' # animatediff is compatible with sd
        elif "Kandinsky5" in name:
            model_type = 'kandinsky5'
        elif "Kandinsky3" in name:
            model_type = 'kandinsky3'
        elif "Kandinsky" in name:
            model_type = 'kandinsky'
        elif "HunyuanDiT" in name:
            model_type = 'hunyuandit'
        elif "Cascade" in name:
            model_type = 'sc'
        elif "AuraFlow" in name:
            model_type = 'auraflow'
        elif 'Chroma' in name:
            model_type = 'chroma'
        elif "Flux2" in name:
            model_type = 'f2'
        elif "Flux" in name or "Flex1" in name or "Flex2" in name:
            model_type = 'f1'
        elif "ZImage" in name or "Z-Image" in name:
            model_type = 'z_image'
        elif "Lumina2" in name:
            model_type = 'lumina2'
        elif "Lumina" in name:
            model_type = 'lumina'
        elif "OmniGen2" in name:
            model_type = 'omnigen2'
        elif "OmniGen" in name:
            model_type = 'omnigen'
        elif "CogView3" in name:
            model_type = 'cogview3'
        elif "CogView4" in name:
            model_type = 'cogview4'
        elif "Sana" in name:
            model_type = 'sana'
        elif "HiDream" in name:
            model_type = 'h1'
        elif "Cosmos2TextToImage" in name:
            model_type = 'cosmos'
        elif "FLite" in name:
            model_type = 'flite'
        elif "PixArtSigma" in name:
            model_type = 'pixartsigma'
        elif "PixArtAlpha" in name:
            model_type = 'pixartalpha'
        elif "Bria" in name:
            model_type = 'bria'
        elif 'Qwen' in name:
            model_type = 'qwen'
        elif 'NextStep' in name:
            model_type = 'nextstep'
        elif 'X-Omni' in name:
            model_type = 'x-omni'
        elif 'Photoroom' in name:
            model_type = 'prx'
        # video models
        elif "CogVideo" in name:
            model_type = 'cogvideo'
        elif 'HunyuanVideo15' in name:
            model_type = 'hunyuanvideo15'
        elif 'HunyuanVideoPipeline' in name or 'HunyuanSkyreels' in name:
            model_type = 'hunyuanvideo'
        elif 'LTX' in name:
            model_type = 'ltxvideo'
        elif "Mochi" in name:
            model_type = 'mochivideo'
        elif "Allegro" in name:
            model_type = 'allegrovideo'
        # hybrid models
        elif 'Wan' in name:
            model_type = 'wanai'
        elif 'ChronoEdit' in name:
            model_type = 'chrono'
        elif 'HDM-xut' in name:
            model_type = 'hdm'
        elif 'HunyuanImage3' in name:
            model_type = 'hunyuanimage3'
        elif 'HunyuanImage' in name:
            model_type = 'hunyuanimage'
        # cloud models
        elif 'GoogleVeo' in name:
            model_type = 'veo3'
        elif 'NanoBanana' in name:
            model_type = 'nanobanana'
        else:
            model_type = name
        return model_type
