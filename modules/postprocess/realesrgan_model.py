import os
import numpy as np
from PIL import Image
from modules.postprocess.realesrgan_model_arch import SRVGGNetCompact
from modules.upscaler import Upscaler
from modules.shared import opts, device, log
from modules import devices


class UpscalerRealESRGAN(Upscaler):
    def __init__(self, dirname):
        from installer import install
        install('addict')
        install('yapf')
        install('--no-build-isolation git+https://github.com/Disty0/BasicSR@23c1fb6f5c559ef5ce7ad657f2fa56e41b121754', 'basicsr', ignore=True, quiet=True)
        from basicsr.archs.rrdbnet_arch import RRDBNet
        self.name = "RealESRGAN"
        self.user_path = dirname
        super().__init__()
        self.scalers = self.find_scalers()
        self.models = {}
        for scaler in self.scalers:
            if scaler.name == 'RealESRGAN 2x+':
                scaler.model = lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                scaler.scale = 2
            elif scaler.name == 'RealESRGAN 4x+ Anime6B':
                scaler.model=lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            elif scaler.name == 'RealESRGAN 4x General V3':
                scaler.model = lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            elif scaler.name == 'RealESRGAN 4x General WDN V3':
                scaler.model = lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            elif scaler.name == 'RealESRGAN 4x AnimeVideo V3':
                scaler.model=lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            elif scaler.name == 'RealESRGAN 4x+':
                scaler.model = lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            else:
                log.error(f"Upscaler unrecognized model: type={self.name} model={scaler.name}")

    def load_model(self, path): # pylint: disable=unused-argument
        pass

    def do_upscale(self, img, selected_model): # pylint: disable=arguments-differ
        if not self.enable:
            return img
        try:
            from modules.postprocess.realesrgan_model_arch import RealESRGANer
        except Exception:
            log.error("Error importing Real-ESRGAN:")
            return img
        model = self.find_model(selected_model)
        if model is None or not os.path.exists(model.local_data_path):
            return img
        if self.models.get(model.local_data_path, None) is not None:
            log.debug(f"Upscaler cached: type={self.name} model={model.local_data_path}")
            upsampler=self.models[model.local_data_path]
        else:
            kwargs = {
                'name': model.name,
                'scale': model.scale,
                'model_path': model.local_data_path,
                'model': model.model(),
                'half': not opts.no_half and not opts.upcast_sampling,
                'tile': opts.upscaler_tile_size,
                'tile_pad': opts.upscaler_tile_overlap,
                'device': device,
            }
            upsampler = RealESRGANer(**kwargs)
            self.models[model.local_data_path] = upsampler
        upsampled = upsampler.enhance(np.array(img), outscale=model.scale)[0]
        if opts.upscaler_unload and model.local_data_path in self.models:
            del self.models[model.local_data_path]
            log.debug(f"Upscaler unloaded: type={self.name} model={selected_model}")
            devices.torch_gc(force=True)

        image = Image.fromarray(upsampled)
        return image
