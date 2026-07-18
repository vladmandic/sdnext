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
    'krea2',
]


force_classes_diffusers = [ # forced always
    'FluxKontextPipeline', 'FluxKontextInpaintPipeline',
]

fuse_ignore = [
    'hunyuanvideo',
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


# Roles a LoRA is fused into; a quantized component in any of them makes fusing unsafe.
fuse_roots = ('transformer', 'unet', 'text_encoder', 'llm_adapter')


def fuse_components(sd_model):
    """Component names a network fuses into, matched by role prefix so numbered and reference siblings are covered."""
    names = getattr(sd_model, 'components', None)
    if not isinstance(names, dict):
        names = vars(sd_model)
    return [name for name in names if name.startswith(fuse_roots)]


def is_quantized(module):
    """Return True when ``module`` carries a quantization config.

    ``config.quantization_config`` is read first: SDNQ sets both it and the plain
    attribute when it quantizes in place, but a checkpoint that ships pre-quantized
    only reaches the plain attribute through the diffusers ConfigMixin name proxy,
    which is deprecated for removal.
    """
    if module is None:
        return False
    config = getattr(module, 'config', None)
    if config is not None and getattr(config, 'quantization_config', None) is not None:
        return True
    return getattr(module, 'quantization_config', None) is not None


def disable_fuse():
    """Return True when fusing a network into model weights is unsafe.

    Fusing keeps no pristine copy of the weight, so each apply and restore
    round-trips it through its storage format. On quantized weights that is a
    dequantize-add-requantize cycle per network swap whose error compounds.
    """
    from modules.lora import lora_common as l
    from modules.lora import lora_stack
    if lora_stack.select_possible(len(l.loaded_networks)) or lora_stack.select_engaged():
        return True # select flips per-layer winners against the pristine backup; a dormant select mode leaves fuse alone
    sd_model = getattr(shared.sd_model, 'pipe', shared.sd_model)
    if is_quantized(sd_model):
        return True
    if any(is_quantized(getattr(sd_model, name, None)) for name in fuse_components(sd_model)):
        return True
    if hasattr(sd_model, '_lora_partial'):
        return True
    return shared.sd_model_type in fuse_ignore


def fuse_native():
    """Return True when the native apply path may fuse into model weights.

    The single source of truth for the native fuse decision: it must agree across
    the backup, activate and deactivate passes, since backup mode restores from a
    stored tensor while fuse mode restores by subtracting the delta.
    """
    return shared.opts.lora_fuse_native and not disable_fuse()
