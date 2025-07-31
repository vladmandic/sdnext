import os
import transformers
import diffusers
from modules import errors, shared, sd_models, sd_unet, sd_hijack_te, devices, modelloader, model_quant

debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_lumina(_checkpoint_info, diffusers_load_config={}):
    modelloader.hf_login()
    load_config, _quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    pipe = diffusers.LuminaText2ImgPipeline.from_pretrained(
        'Alpha-VLLM/Lumina-Next-SFT-diffusers',
        cache_dir = shared.opts.diffusers_dir,
        **load_config,
    )
    devices.torch_gc(force=True, reason='load')
    return pipe


def load_lumina2(checkpoint_info, diffusers_load_config={}):
    transformer, text_encoder, vae = None, None, None
    repo_id = sd_models.path_to_repo(checkpoint_info)

    if shared.opts.teacache_enabled:
        from modules import teacache
        shared.log.debug(f'Transformers cache: type=teacache patch=forward cls={diffusers.Lumina2Transformer2DModel.__name__}')
        diffusers.Lumina2Transformer2DModel.forward = teacache.teacache_lumina2_forward # patch must be done before transformer is loaded

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='Model')
    if shared.opts.sd_unet != 'Default':
        try:
            debug(f'Load model: type=Lumina2 unet="{shared.opts.sd_unet}"')
            transformer = diffusers.Lumina2Transformer2DModel.from_single_file(
                sd_unet.unet_dict[shared.opts.sd_unet],
                cache_dir=shared.opts.diffusers_dir,
                **load_config,
                **quant_config
            )
            if transformer is None:
                shared.opts.sd_unet = 'Default'
                sd_unet.failed_unet.append(shared.opts.sd_unet)
        except Exception as e:
            shared.log.error(f"Load model: type=Lumina2 failed to load UNet: {e}")
            shared.opts.sd_unet = 'Default'
            if debug:
                errors.display(e, 'Lumina2 UNet:')

    if shared.opts.sd_vae != 'Default' and shared.opts.sd_vae != 'Automatic':
        try:
            debug(f'Load model: type=Lumina2 vae="{shared.opts.sd_vae}"')
            from modules import sd_vae
            # vae = sd_vae.load_vae_diffusers(None, sd_vae.vae_dict[shared.opts.sd_vae], 'override')
            vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
            if os.path.exists(vae_file):
                vae_config = os.path.join('configs', 'flux', 'vae', 'config.json')
                vae = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, **diffusers_load_config)
        except Exception as e:
            shared.log.error(f"Load model: type=Lumina2 failed to load VAE: {e}")
            shared.opts.sd_vae = 'Default'
            if debug:
                errors.display(e, 'Lumina2 VAE:')

    if transformer is None:
        transformer = diffusers.Lumina2Transformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            cache_dir=shared.opts.diffusers_dir,
            **load_config,
            **quant_config,
        )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, module='TE', device_map=True)
    text_encoder = transformers.AutoModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        cache_dir=shared.opts.diffusers_dir,
        **load_config,
        **quant_config,
    )

    load_config, quant_config = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    if vae is not None:
        load_config['vae'] = vae
    pipe = diffusers.Lumina2Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        text_encoder=text_encoder,
        transformer=transformer,
        **load_config,
    )

    sd_hijack_te.init_hijack(pipe)
    devices.torch_gc(force=True, reason='load')
    return pipe
