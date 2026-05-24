from modules import shared


force_hashes_diffusers = [ # forced always
    # '816d0eed49fd', # flash-sdxl
    # 'c2ec22757b46', # flash-sd15
    # '22c8339e7666', # spo-sdxl-10ep
    # 'aaebf6360f7d', # sd15-lcm
    # '3d18b05e4f56', # sdxl-lcm
    # 'b71dcb732467', # sdxl-tcd
    # '813ea5fb1c67', # sdxl-turbo
    # '5a48ac366664', # hyper-sd15-1step
    # 'ee0ff23dcc42', # hyper-sd15-2step
    # 'e476eb1da5df', # hyper-sd15-4step
    # 'ecb844c3f3b0', # hyper-sd15-8step
    # '1ab289133ebb', # hyper-sd15-8step-cfg
    # '4f494295edb1', # hyper-sdxl-8step
    # 'ca14a8c621f8', # hyper-sdxl-8step-cfg
    # '1c88f7295856', # hyper-sdxl-4step
    # 'fdd5dcd1d88a', # hyper-sdxl-2step
    # '8cca3706050b', # hyper-sdxl-1step
]

allow_native = [
    'sd',
    'sdxl',
    'sd3',
    'f1',
    'f2',
    'chroma',
    'zimage',
    'anima',
    'ernieimage',
    'ltxvideo',
]


force_classes_diffusers = [ # forced always
    'FluxKontextPipeline', 'FluxKontextInpaintPipeline',
]

# Diffusers-path fuse skips fuse_lora/unload_lora_weights after set_adapters so adapters
# stay live for multi-stage composition (LTX 2.x stage 2 distilled LoRA is composed on
# top of user LoRAs; PEFT cannot selectively unfuse, so fuse breaks the composition).
fuse_ignore = [
    'hunyuanvideo',
    'ltxvideo',
]


def get_method(shorthash=''):
    """Return ``(method, reason)`` for the active LoRA loading strategy.

    ``method`` is one of ``'native'``, ``'diffusers'``, ``'nunchaku'``.
    ``reason`` is a short identifier indicating which condition triggered the
    chosen method, useful for distinguishing user-opt-in from automatic
    fallback in logs. Reasons:

    - ``'nunchaku-transformer'`` / ``'nunchaku-unet'``: a Nunchaku-quantized
      component is loaded.
    - ``'opt-in'``: ``shared.opts.lora_force_diffusers`` is on (settings).
    - ``'class-forced'``: pipeline class is in ``force_classes_diffusers``.
    - ``'arch-unsupported'``: ``sd_model_type`` is not in ``allow_native``.
    - ``'hash-forced'``: file hash is in ``force_hashes_diffusers``.
    - ``'default'``: native path is the active and unforced choice.
    """
    nunchaku_dit = hasattr(shared.sd_model, 'transformer') and 'Nunchaku' in shared.sd_model.transformer.__class__.__name__
    nunchaku_unet = hasattr(shared.sd_model, 'unet') and 'Nunchaku' in shared.sd_model.unet.__class__.__name__
    if nunchaku_dit:
        return 'nunchaku', 'nunchaku-transformer'
    if nunchaku_unet:
        return 'nunchaku', 'nunchaku-unet'
    if shared.opts.lora_force_diffusers:
        return 'diffusers', 'opt-in'
    if shared.sd_model.__class__.__name__ in force_classes_diffusers:
        return 'diffusers', 'class-forced'
    if shared.sd_model_type not in allow_native:
        return 'diffusers', 'arch-unsupported'
    if len(shorthash) > 4 and any(x.startswith(shorthash) for x in force_hashes_diffusers):
        return 'diffusers', 'hash-forced'
    return 'native', 'default'


def disable_fuse():
    if hasattr(shared.sd_model, 'quantization_config'):
        return True
    if hasattr(shared.sd_model, 'transformer') and hasattr(shared.sd_model.transformer, 'quantization_config'):
        return True
    if hasattr(shared.sd_model, 'transformer_2') and hasattr(shared.sd_model.transformer_2, 'quantization_config'):
        return True
    if hasattr(shared.sd_model, '_lora_partial'):
        return True
    return shared.sd_model_type in fuse_ignore
