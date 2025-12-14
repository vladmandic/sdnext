# using https://github.com/rootonchair/diffuser_layerdiffuse

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sdnext_core import MODELDATA
from modules import shared, errors, devices
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
        if MODELDATA.sd_model_type == 'sd':
            shared.log.info(f'LayerDiffuse: class={MODELDATA.sd_model.__class__.__name__}')
            apply_layerdiffuse_sd15(MODELDATA.sd_model)
        elif MODELDATA.sd_model_type == 'sdxl':
            # shared.log.info(f'LayerDiffuse: class={shared.sd_model.__class__.__name__} type=attn')
            # apply_layerdiffuse_sdxl_attn(shared.sd_model)
            shared.log.info(f'LayerDiffuse: class={MODELDATA.sd_model.__class__.__name__} type=conv')
            apply_layerdiffuse_sdxl_conv(MODELDATA.sd_model)
        else:
            shared.log.warning(f'LayerDiffuse: class={MODELDATA.sd_model.__class__.__name__} not supported')
        MODELDATA.sd_model.layerdiffusion = True
    except Exception as e:
        shared.log.error(f'LayerDiffuse: {e}')
        errors.display(e, 'LayerDiffuse')
