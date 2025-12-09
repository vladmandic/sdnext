import os
from modules import shared, devices


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
vae_type = None


def set_vae_params(p):
    global vae_type # pylint: disable=global-statement
    vae_type = p.vae_type
    if not hasattr(shared.sd_model, 'vae'):
        return
    if hasattr(shared.sd_model.vae, 'enable_slicing'):
        shared.sd_model.vae.enable_slicing()
    if p.frames > p.vae_tile_frames:
        if hasattr(shared.sd_model.vae, 'tile_sample_min_num_frames'):
            shared.sd_model.vae.tile_sample_min_num_frames = p.vae_tile_frames
        if hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
            shared.sd_model.vae.use_framewise_decoding = True
        if hasattr(shared.sd_model.vae, 'enable_tiling'):
            shared.sd_model.vae.enable_tiling()
    else:
        if hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
            shared.sd_model.vae.use_framewise_decoding = False
        if hasattr(shared.sd_model.vae, 'disable_tiling'):
            shared.sd_model.vae.disable_tiling()


def vae_decode_tiny(latents):
    if 'Hunyuan' in shared.sd_model.__class__.__name__:
        variant = 'TAE HunyuanVideo'
    elif 'Mochi' in shared.sd_model.__class__.__name__:
        variant = 'TAE MochiVideo'
    elif 'WAN' in shared.sd_model.__class__.__name__:
        variant = 'TAE WanVideo'
    elif 'Kandinsky' in shared.sd_model.__class__.__name__:
        variant = 'TAE HunyuanVideo'
    else:
        shared.log.warning(f'Decode: type=Tiny cls={shared.sd_model.__class__.__name__} not supported')
        return None
    from modules import sd_vae_taesd
    vae, variant = sd_vae_taesd.get_model(variant=variant)
    if vae is None:
        return None
    shared.log.debug(f'Decode: type=Tiny cls={vae.__class__.__name__} variant="{variant}" latents={latents.shape}')
    vae = vae.to(device=devices.device, dtype=devices.dtype)
    latents = latents.transpose(1, 2).to(device=devices.device, dtype=devices.dtype)
    images = vae.decode_video(latents, parallel=False).transpose(1, 2).mul_(2).sub_(1)
    images = images.transpose(1, 2).mul_(2).sub_(1)
    return (images, None)
