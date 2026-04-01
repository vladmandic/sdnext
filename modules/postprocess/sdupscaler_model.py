import torch
import diffusers
from PIL import Image
from modules import shared, devices
from modules.logger import log
from modules.upscaler import Upscaler, UpscalerData


class UpscalerDiffusion(Upscaler):
    def __init__(self, dirname): # pylint: disable=super-init-not-called
        self.name = "SDUpscale"
        self.user_path = dirname
        self.scalers = [
            UpscalerData(name="Diffusion Latent Upscaler 2x", path="stabilityai/sd-x2-latent-upscaler", upscaler=self, model=None, scale=4),
            UpscalerData(name="Diffusion Latent Upscaler 4x", path="stabilityai/stable-diffusion-x4-upscaler", upscaler=self, model=None, scale=4),
        ]
        self.pipelines = [
            None,
            None,
        ]
        self.models = {}

    def load_model(self, path: str):
        from modules.sd_models import set_diffuser_options
        scaler: UpscalerData = [x for x in self.scalers if x.data_path == path or x.name == path]
        if len(scaler) == 0:
            log.error(f"Upscaler cannot match model: type={self.name} model={path}")
            return None
        scaler = scaler[0]
        if self.models.get(path, None) is not None:
            log.debug(f"Upscaler cached: type={scaler.name} model={path}")
            return self.models[path]
        else:
            model = diffusers.DiffusionPipeline.from_pretrained(scaler.data_path, cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype)
            if hasattr(model, "set_progress_bar_config"):
                model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + 'Upscale', ncols=80, colour='#327fba')
            set_diffuser_options(model, vae=None, op='upscaler')
            self.models[path] = model
        return self.models[path]

    def callback(self, _step: int, _timestep: int, _latents: torch.FloatTensor):
        pass

    def do_upscale(self, img: Image.Image, selected_model):
        devices.torch_gc()
        model = self.load_model(selected_model)
        if model is None:
            return img
        seeds = [torch.randint(0, 2 ** 32, (1,)).item() for _ in range(1)]
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else devices.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in seeds]
        args = {
            'prompt': '',
            'negative_prompt': '',
            'image': img,
            'num_inference_steps': shared.opts.upscaler_latent_steps,
            'guidance_scale': 7.5,
            'generator': generator,
            'latents': None,
            'return_dict': True,
            'callback': self.callback,
            'callback_steps': 1,
            # 'noise_level': 100,
            # 'num_images_per_prompt': 1,
            # 'eta': 0.0,
            # 'cross_attention_kwargs': None,
        }
        model = model.to(devices.device)
        output = model(**args)
        image = output.images[0]
        if shared.opts.upscaler_unload and selected_model in self.models:
            del self.models[selected_model]
            log.debug(f"Upscaler unloaded: type={self.name} model={selected_model}")
            devices.torch_gc(force=True)
        return image
