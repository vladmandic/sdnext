# using https://github.com/rootonchair/diffuser_layerdiffuse

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from modules import shared, errors, devices
from modules.logger import log
from .layerdiffuse_model import TransparentVAEDecoder
from .layerdiffuse_loader import load_lora_to_unet, merge_delta_weights_into_unet


def apply_layerdiffuse_sd15(pipeline):
    vae_model_path = hf_hub_download('LayerDiffusion/layerdiffusion-v1', 'layer_sd15_vae_transparent_decoder.safetensors', cache_dir=shared.opts.hfcache_dir)
    transparent_vae = pipeline.vae
    transparent_vae.__class__ = TransparentVAEDecoder
    transparent_vae.set_transparent_decoder(load_file(vae_model_path))
    pipeline.vae = transparent_vae

    lora_model_path = hf_hub_download('LayerDiffusion/layerdiffusion-v1','layer_sd15_transparent_attn.safetensors', cache_dir=shared.opts.hfcache_dir)
    load_lora_to_unet(pipeline.unet, lora_model_path, frames=1, device=devices.device, dtype=devices.dtype)


def apply_layerdiffuse_sdxl_attn(pipeline):
    vae_model_path = hf_hub_download('LayerDiffusion/layerdiffusion-v1', 'vae_transparent_decoder.safetensors', cache_dir=shared.opts.hfcache_dir)
    transparent_vae = pipeline.vae
    transparent_vae.__class__ = TransparentVAEDecoder
    transparent_vae.set_transparent_decoder(load_file(vae_model_path))
    pipeline.vae = transparent_vae

    pipeline.load_lora_weights('rootonchair/diffuser_layerdiffuse', weight_name='diffuser_layer_xl_transparent_attn.safetensors')


def apply_layerdiffuse_sdxl_conv(pipeline):
    model_path = hf_hub_download('LayerDiffusion/layerdiffusion-v1', 'vae_transparent_decoder.safetensors', cache_dir=shared.opts.hfcache_dir)
    transparent_vae = pipeline.vae
    transparent_vae.__class__ = TransparentVAEDecoder
    transparent_vae.set_transparent_decoder(load_file(model_path))
    pipeline.vae = transparent_vae

    lora_model_path = hf_hub_download('rootonchair/diffuser_layerdiffuse', 'diffuser_layer_xl_transparent_conv.safetensors', cache_dir=shared.opts.hfcache_dir)
    lora_state_dict = load_file(lora_model_path)
    merge_delta_weights_into_unet(pipeline, lora_state_dict)


def apply_layerdiffuse():
    try:
        if shared.sd_model_type == 'sd':
            log.info(f'LayerDiffuse: class={shared.sd_model.__class__.__name__}')
            apply_layerdiffuse_sd15(shared.sd_model)
        elif shared.sd_model_type == 'sdxl':
            # log.info(f'LayerDiffuse: class={shared.sd_model.__class__.__name__} type=attn')
            # apply_layerdiffuse_sdxl_attn(shared.sd_model)
            log.info(f'LayerDiffuse: class={shared.sd_model.__class__.__name__} type=conv')
            apply_layerdiffuse_sdxl_conv(shared.sd_model)
        else:
            log.warning(f'LayerDiffuse: class={shared.sd_model.__class__.__name__} not supported')
        shared.sd_model.layerdiffusion = True
    except Exception as e:
        log.error(f'LayerDiffuse: {e}')
        errors.display(e, 'LayerDiffuse')
