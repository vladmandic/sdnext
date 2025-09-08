from modules import shared


maybe_diffusers = [ # forced if lora_maybe_diffusers is enabled
    # 'aaebf6360f7d', # sd15-lcm
    # '3d18b05e4f56', # sdxl-lcm
    # 'b71dcb732467', # sdxl-tcd
    # '813ea5fb1c67', # sdxl-turbo
    # not really needed, but just in case
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

force_diffusers = [ # forced always
    '816d0eed49fd', # flash-sdxl
    'c2ec22757b46', # flash-sd15
    '22c8339e7666', # spo-sdxl-10ep
]

force_models_diffusers = [ # forced always
    # 'sd3',
    'sc',
    'h1',
    'kandinsky',
    'hunyuandit',
    'auraflow',
    'lumina2',
    'qwen',
    'bria',
    'flite',
    'cosmos',
    # video models
    'hunyuanvideo',
    'cogvideo',
    'wanai',
    'ltxvideo',
    'mochivideo',
    'allegrovideo',
]

force_classes_diffusers = [ # forced always
    'FluxKontextPipeline', 'FluxKontextInpaintPipeline',
]

fuse_ignore = [
    'hunyuanvideo',
]


def get_method(shorthash=''):
    use_diffusers = (shared.sd_model_type in force_models_diffusers) or (shared.sd_model.__class__.__name__ in force_classes_diffusers)
    if shared.opts.lora_maybe_diffusers and len(shorthash) > 4:
        use_diffusers = use_diffusers or any(x.startswith(shorthash) for x in maybe_diffusers)
    if shared.opts.lora_force_diffusers and len(shorthash) > 4:
        use_diffusers = use_diffusers or any(x.startswith(shorthash) for x in force_diffusers)
    use_nunchaku = hasattr(shared.sd_model, 'transformer') and 'Nunchaku' in shared.sd_model.transformer.__class__.__name__
    if use_nunchaku:
        return 'nunchaku'
    elif use_diffusers:
        return 'diffusers'
    else:
        return 'native'


def disable_fuse():
    if hasattr(shared.sd_model, 'quantization_config'):
        return True
    if hasattr(shared.sd_model, 'transformer') and hasattr(shared.sd_model.transformer, 'quantization_config'):
        return True
    return shared.sd_model_type in fuse_ignore
