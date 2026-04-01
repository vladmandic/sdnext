from modules.logger import log
from modules import shared, sd_models, devices, attention
from .linfusion import LinFusion
from .attention import GeneralizedLinearAttention


applied: LinFusion = None


def detect(pipeline):
    if pipeline.__class__.__name__ == 'StableDiffusionXLPipeline':
        return "Yuanshi/LinFusion-XL"
    if pipeline.__class__.__name__ == 'StableDiffusionPipeline':
        return "Yuanshi/LinFusion-1-5"
    return None


def apply(pipeline, pretrained: bool = True):
    global applied # pylint: disable=global-statement
    if not shared.opts.enable_linfusion:
        return
    if applied is not None:
        return
    # linfusion = LinFusion.construct_for(pipeline=pipeline)
    if not pretrained:
        model_path = None
        default_config = LinFusion.get_default_config(unet=pipeline.unet)
        applied = LinFusion(**default_config).to(device=pipeline.unet.device, dtype=pipeline.unet.dtype)
        applied.mount_to(unet=pipeline.unet)
    else:
        model_path = detect(pipeline)
        if model_path is None:
            log.error('LinFusion: unsupported model type')
            return
        applied = LinFusion.from_pretrained(model_path, cache_dir=shared.opts.hfcache_dir).to(device=pipeline.unet.device, dtype=pipeline.unet.dtype)
        applied.mount_to(unet=pipeline.unet)
    log.info(f'Applying LinFusion: class={applied.__class__.__name__} model="{model_path}" modules={len(applied.modules_dict)}')


def unapply(pipeline):
    global applied # pylint: disable=global-statement
    if applied is None:
        return
    # log.debug('LinFusion: unapply')
    attention.set_diffusers_attention(pipeline)
    devices.torch_gc()
    applied = None
