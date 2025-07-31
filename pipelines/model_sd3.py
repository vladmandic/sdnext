import os
import diffusers
import transformers
from modules import shared, devices, errors, sd_models, sd_unet, model_quant, model_tools


def load_overrides(kwargs, cache_dir):
    if shared.opts.sd_unet != 'Default':
        try:
            fn = sd_unet.unet_dict[shared.opts.sd_unet]
            if fn.endswith('.safetensors'):
                kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_single_file(fn, cache_dir=cache_dir, torch_dtype=devices.dtype)
                sd_unet.loaded_unet = shared.opts.sd_unet
                shared.log.debug(f'Load model: type=SD3 unet="{shared.opts.sd_unet}" fmt=safetensors')
            elif fn.endswith('.gguf'):
                from modules import ggml
                kwargs['transformer'] = ggml.load_gguf(fn, cls=diffusers.SD3Transformer2DModel, compute_dtype=devices.dtype)
                sd_unet.loaded_unet = shared.opts.sd_unet
                shared.log.debug(f'Load model: type=SD3 unet="{shared.opts.sd_unet}" fmt=gguf')
        except Exception as e:
            shared.log.error(f"Load model: type=SD3 failed to load UNet: {e}")
            errors.display(e, 'UNet')
            shared.opts.sd_unet = 'Default'
            sd_unet.failed_unet.append(shared.opts.sd_unet)

    if shared.opts.sd_text_encoder != 'Default':
        try:
            from modules.model_te import load_t5, load_vit_l, load_vit_g
            if 'vit-l' in shared.opts.sd_text_encoder.lower():
                kwargs['text_encoder'] = load_vit_l()
                shared.log.debug(f'Load model: type=SD3 variant="vit-l" te="{shared.opts.sd_text_encoder}"')
            elif 'vit-g' in shared.opts.sd_text_encoder.lower():
                kwargs['text_encoder_2'] = load_vit_g()
                shared.log.debug(f'Load model: type=SD3 variant="vit-g" te="{shared.opts.sd_text_encoder}"')
            else:
                kwargs['text_encoder_3'] = load_t5(name=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
                shared.log.debug(f'Load model: type=SD3 variant="t5" te="{shared.opts.sd_text_encoder}"')
        except Exception as e:
            shared.log.error(f"Load model: type=SD3 failed to load T5: {e}")
            errors.display(e, 'TE')
            shared.opts.sd_text_encoder = 'Default'

    if shared.opts.sd_vae != 'Default' and shared.opts.sd_vae != 'Automatic':
        try:
            from modules import sd_vae
            vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
            if os.path.exists(vae_file):
                vae_config = os.path.join('configs', 'sd3', 'vae', 'config.json')
                kwargs['vae'] = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, cache_dir=cache_dir, torch_dtype=devices.dtype)
                shared.log.debug(f'Load model: type=SD3 vae="{shared.opts.sd_vae}"')
        except Exception as e:
            shared.log.error(f"Load model: type=SD3 failed to load VAE: {e}")
            errors.display(e, 'VAE')
            shared.opts.sd_vae = 'Default'
    return kwargs


def load_quants(kwargs, repo_id, cache_dir):
    quant_args = model_quant.create_config(module='Model')
    if quant_args and 'quantization_config' in quant_args:
        kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
    quant_args = model_quant.create_config(module='TE')
    if quant_args and 'quantization_config' in quant_args:
        kwargs['text_encoder_3'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_3", variant='fp16', cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
    return kwargs


def load_missing(kwargs, fn, cache_dir):
    keys = model_tools.get_safetensor_keys(fn)
    size = os.stat(fn).st_size // 1024 // 1024
    if size > 15000:
        repo_id = 'stabilityai/stable-diffusion-3.5-large'
    else:
        repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    if 'text_encoder' not in kwargs and 'text_encoder' not in keys:
        kwargs['text_encoder'] = transformers.CLIPTextModelWithProjection.from_pretrained(repo_id, subfolder='text_encoder', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=te1 repo="{repo_id}"')
    if 'text_encoder_2' not in kwargs and 'text_encoder_2' not in keys:
        kwargs['text_encoder_2'] = transformers.CLIPTextModelWithProjection.from_pretrained(repo_id, subfolder='text_encoder_2', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=te2 repo="{repo_id}"')
    if 'text_encoder_3' not in kwargs and 'text_encoder_3' not in keys:
        load_args, quant_args = model_quant.get_dit_args({}, module='TE', device_map=True)
        kwargs['text_encoder_3'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_3", variant='fp16', cache_dir=cache_dir, **load_args, **quant_args)
        shared.log.debug(f'Load model: type=SD3 missing=te3 repo="{repo_id}"')
    if 'vae' not in kwargs and 'vae' not in keys:
        kwargs['vae'] = diffusers.AutoencoderKL.from_pretrained(repo_id, subfolder='vae', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=vae repo="{repo_id}"')
    return kwargs


def load_sd3(checkpoint_info, cache_dir=None, config=None):
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    fn = checkpoint_info.path

    kwargs = {}
    kwargs = load_overrides(kwargs, cache_dir)
    if (fn is None) or (not os.path.exists(fn) or os.path.isdir(fn)):
        kwargs = load_quants(kwargs, repo_id, cache_dir)

    loader = diffusers.StableDiffusion3Pipeline.from_pretrained
    if fn is not None and os.path.exists(fn) and os.path.isfile(fn):
        if fn.endswith('.safetensors'):
            loader = diffusers.StableDiffusion3Pipeline.from_single_file
            repo_id = fn
        elif fn.endswith('.gguf'):
            from modules import ggml
            kwargs['transformer'] = ggml.load_gguf(fn, cls=diffusers.SD3Transformer2DModel, compute_dtype=devices.dtype)
            kwargs = load_missing(kwargs, fn, cache_dir)
            kwargs['variant'] = 'fp16'
    else:
        kwargs['variant'] = 'fp16'

    shared.log.debug(f'Load model: type=SD3 kwargs={list(kwargs)} repo="{repo_id}"')

    if shared.opts.model_sd3_disable_te5:
        shared.log.debug('Load model: type=SD3 option="disable-te5"')
        kwargs['text_encoder_3'] = None

    pipe = loader(
        repo_id,
        torch_dtype=devices.dtype,
        cache_dir=cache_dir,
        config=config,
        **kwargs,
    )
    devices.torch_gc(force=True, reason='load')
    return pipe
