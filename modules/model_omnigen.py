import os
import diffusers
from modules import errors, shared, devices, sd_models, model_quant

debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_omnigen(checkpoint_info, diffusers_load_config={}): # pylint: disable=unused-argument
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    vae = None

    if shared.opts.sd_vae != 'Default' and shared.opts.sd_vae != 'Automatic':
        try:
            debug(f'Load model: type=OmniGen vae="{shared.opts.sd_vae}"')
            from modules import sd_vae
            # vae = sd_vae.load_vae_diffusers(None, sd_vae.vae_dict[shared.opts.sd_vae], 'override')
            vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
            if os.path.exists(vae_file):
                vae_config = os.path.join('configs', 'sdxl', 'vae', 'config.json')
                vae = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, **diffusers_load_config)
        except Exception as e:
            shared.log.error(f"Load model: type=OmniGen failed to load VAE: {e}")
            shared.opts.sd_vae = 'Default'
            if debug:
                errors.display(e, 'OmniGen VAE:')

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='Model')
    transformer = diffusers.OmniGenTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    if vae is not None:
        load_config['vae'] = vae
    pipe = diffusers.OmniGenPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
    )

    devices.torch_gc(force=True)
    return pipe
