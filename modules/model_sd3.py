import os
import diffusers
import transformers
from modules import shared, devices, sd_models, sd_unet, model_quant, model_tools


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
                # kwargs = load_gguf(kwargs, fn)
                kwargs['transformer'] = ggml.load_gguf(fn, cls=diffusers.SD3Transformer2DModel, compute_dtype=devices.dtype)
                sd_unet.loaded_unet = shared.opts.sd_unet
                shared.log.debug(f'Load model: type=SD3 unet="{shared.opts.sd_unet}" fmt=gguf')
        except Exception as e:
            shared.log.error(f"Load model: type=SD3 failed to load UNet: {e}")
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
            shared.opts.sd_vae = 'Default'
    return kwargs


def load_quants(kwargs, repo_id, cache_dir):
    quant_args = model_quant.create_config()
    if not quant_args:
        return kwargs
    if 'transformer' not in kwargs and (('Model' in shared.opts.bnb_quantization or 'Model' in shared.opts.torchao_quantization or 'Model' in shared.opts.quanto_quantization) or ('Transformer' in shared.opts.bnb_quantization or 'Transformer' in shared.opts.torchao_quantization or 'Transformer' in shared.opts.quanto_quantization)):
        kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, torch_dtype=devices.dtype, **quant_args)
    if 'text_encoder_3' not in kwargs and ('TE' in shared.opts.bnb_quantization or 'TE' in shared.opts.torchao_quantization or 'TE' in shared.opts.quanto_quantization):
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
        kwargs['text_encoder_3'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_3", variant='fp16', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=te3 repo="{repo_id}"')
    if 'vae' not in kwargs and 'vae' not in keys:
        kwargs['vae'] = diffusers.AutoencoderKL.from_pretrained(repo_id, subfolder='vae', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=vae repo="{repo_id}"')
    # if 'transformer' not in kwargs and 'transformer' not in keys:
    #    kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_pretrained(default_repo_id, subfolder="transformer", cache_dir=cache_dir, torch_dtype=devices.dtype)
    return kwargs


"""
def load_gguf(kwargs, fn):
    ggml.install_gguf()
    from accelerate import init_empty_weights
    from diffusers.loaders.single_file_utils import convert_sd3_transformer_checkpoint_to_diffusers
    from modules import ggml, sd_hijack_accelerate
    with init_empty_weights():
        config = diffusers.SD3Transformer2DModel.load_config(os.path.join('configs', 'flux'), subfolder="transformer")
        transformer = diffusers.SD3Transformer2DModel.from_config(config).to(devices.dtype)
        expected_state_dict_keys = list(transformer.state_dict().keys())
    state_dict, stats = ggml.load_gguf_state_dict(fn, devices.dtype)
    state_dict = convert_sd3_transformer_checkpoint_to_diffusers(state_dict)
    applied, skipped = 0, 0
    for param_name, param in state_dict.items():
        if param_name not in expected_state_dict_keys:
            skipped += 1
            continue
        applied += 1
        sd_hijack_accelerate.hijack_set_module_tensor_simple(transformer, tensor_name=param_name, value=param, device=0)
        transformer.gguf = 'gguf'
        state_dict[param_name] = None
    shared.log.debug(f'Load model: type=Unet/Transformer applied={applied} skipped={skipped} stats={stats} compute={devices.dtype}')
    kwargs['transformer'] = transformer
    return kwargs
"""


def load_sd3(checkpoint_info, cache_dir=None, config=None):
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    fn = checkpoint_info.path

    # unload current model
    sd_models.unload_model_weights()
    shared.sd_model = None
    devices.torch_gc(force=True)

    kwargs = {}
    kwargs = load_overrides(kwargs, cache_dir)
    if (fn is None) or (not os.path.exists(fn) or os.path.isdir(fn)):
        kwargs = load_quants(kwargs, repo_id, cache_dir)

    loader = diffusers.StableDiffusion3Pipeline.from_pretrained
    if fn is not None and os.path.exists(fn) and os.path.isfile(fn):
        if fn.endswith('.safetensors'):
            loader = diffusers.StableDiffusion3Pipeline.from_single_file
            # required_modules = model_tools.get_modules(diffusers.StableDiffusion3Pipeline)
            # have_modules = model_tools.get_safetensor_keys(fn)
            # loaded_modules = model_tools.load_modules('stabilityai/stable-diffusion-3.5-medium', required_modules)
            # kwargs = {**kwargs, **loaded_modules}
            # kwargs = load_missing(kwargs, fn, cache_dir)
            repo_id = fn
        elif fn.endswith('.gguf'):
            from modules import ggml
            kwargs['transformer'] = ggml.load_gguf(fn, cls=diffusers.SD3Transformer2DModel, compute_dtype=devices.dtype)
            # kwargs = load_gguf(kwargs, fn)
            kwargs = load_missing(kwargs, fn, cache_dir)
            kwargs['variant'] = 'fp16'
    else:
        kwargs['variant'] = 'fp16'

    shared.log.debug(f'Load model: type=SD3 kwargs={list(kwargs)} repo="{repo_id}"')

    kwargs = model_quant.create_config(kwargs)
    pipe = loader(
        repo_id,
        torch_dtype=devices.dtype,
        cache_dir=cache_dir,
        config=config,
        **kwargs,
    )
    devices.torch_gc(force=True)
    return pipe
