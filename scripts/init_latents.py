from sdnext_core import MODELDATA
from modules import scripts_manager, processing, shared, devices


class Script(scripts_manager.Script):
    standalone = False

    def title(self):
        return 'Init Latents'

    def show(self, is_img2img):
        return scripts_manager.AlwaysVisible

    @staticmethod
    def get_latents(p):
        import torch
        from diffusers.utils.torch_utils import randn_tensor
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in p.seeds]
        shape = (len(generator), MODELDATA.sd_model.unet.config.in_channels, p.height // MODELDATA.sd_model.vae_scale_factor, p.width // MODELDATA.sd_model.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=MODELDATA.sd_model._execution_device, dtype=MODELDATA.sd_model.unet.dtype) # pylint: disable=protected-access
        var_generator = [torch.Generator(generator_device).manual_seed(ss) for ss in p.subseeds]
        var_latents = randn_tensor(shape, generator=var_generator, device=MODELDATA.sd_model._execution_device, dtype=MODELDATA.sd_model.unet.dtype) # pylint: disable=protected-access
        return latents, var_latents, generator, var_generator

    @staticmethod
    def set_slerp(p, latents, var_latents, generator, var_generator):
        from modules.processing_helpers import slerp
        p.init_latent = slerp(p.subseed_strength, latents, var_latents) if p.subseed_strength < 1 else var_latents
        p.generator = generator if p.subseed_strength <= 0.5 else var_generator

    def process_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=arguments-differ
        if not MODELDATA.sd_loaded or not hasattr(MODELDATA.sd_model, 'unet'):
            return
        from modules.processing_helpers import create_random_tensors
        args = list(args)
        if p.subseed_strength != 0 and getattr(MODELDATA.sd_model, '_execution_device', None) is not None:
            # alt method using slerp
            # latents, var_latents, generator, var_generator = self.get_latents(p)
            # self.set_slerp(p, latents, var_latents, generator, var_generator)
            p.init_latent = create_random_tensors(
                shape=[MODELDATA.sd_model.unet.config.in_channels, p.height // MODELDATA.sd_model.vae_scale_factor, p.width // MODELDATA.sd_model.vae_scale_factor],
                seeds=p.seeds,
                subseeds=p.subseeds,
                subseed_strength=p.subseed_strength,
                p=p
            )
            shared.log.debug(f'Latent: seed={p.seeds} subseed={p.subseeds} strength={p.subseed_strength} tensor={list(p.init_latent.shape)}')
            p.init_latent = p.init_latent.to(device=MODELDATA.sd_model._execution_device, dtype=MODELDATA.sd_model.unet.dtype) # pylint: disable=protected-access
