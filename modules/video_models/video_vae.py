import os
from modules import shared, devices
from modules.logger import log


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
UNIT_RANGE_DECODERS = {'TAEM1'} # taehv shifts its output to [-1,1] on the way out and taem1 leaves it at [0,1]


def set_vae_params(p, slicing:bool=True, tiling:bool=True, framewise:bool=True) -> None:
    if not hasattr(shared.sd_model, 'vae'):
        return
    shared.sd_model.sdnext_vae_type = p.vae_type # the decode hijack reads the choice off the pipe
    if slicing and hasattr(shared.sd_model.vae, 'enable_slicing'):
        shared.sd_model.vae.enable_slicing()
    if (p.frames > p.vae_tile_frames) and (p.vae_tile_frames > 0):
        if hasattr(shared.sd_model.vae, 'tile_sample_min_num_frames'):
            shared.sd_model.vae.tile_sample_min_num_frames = p.vae_tile_frames
        if framewise and hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
            shared.sd_model.vae.use_framewise_decoding = True
        if tiling and hasattr(shared.sd_model.vae, 'enable_tiling'):
            shared.sd_model.vae.enable_tiling()
        debug(f'VAE params: type={p.vae_type} tiling=True frames={p.frames} tile_frames={p.vae_tile_frames} framewise={getattr(shared.sd_model.vae, "use_framewise_decoding", None)}')
    else:
        if hasattr(shared.sd_model.vae, 'use_framewise_decoding'):
            shared.sd_model.vae.use_framewise_decoding = False
        if hasattr(shared.sd_model.vae, 'disable_tiling'):
            shared.sd_model.vae.disable_tiling()
        debug(f'VAE params: type={p.vae_type} tiling=False frames={p.frames} tile_frames={p.vae_tile_frames} framewise={getattr(shared.sd_model.vae, "use_framewise_decoding", None)}')


def vae_decode_tiny(latents):
    """Decode through the tiny counterpart of the model's vae, or None when there is none to use.

    Returning None leaves the caller on the full vae, so every rejection here is a fallback
    rather than a failure.
    """
    cls = shared.sd_model.__class__.__name__
    if 'Hunyuan' in cls:
        variant = 'TAE HunyuanVideo'
    elif 'Mochi' in cls:
        variant = 'TAE MochiVideo'
    elif 'Wan' in cls:
        variant = 'TAE WanVideo'
    elif 'Kandinsky' in cls:
        variant = 'TAE HunyuanVideo'
    else:
        log.warning(f'Decode: type=Tiny cls={cls} not supported')
        return None
    from modules.vae import sd_vae_taesd
    vae, variant = sd_vae_taesd.load_model(variant=variant)
    if vae is None:
        return None
    expected = getattr(vae, 'latent_channels', None) # 16 on taehv and 12 on taem1, so ask the decoder rather than assume
    channels = latents.shape[1] if latents.ndim == 5 else None # the pipes hand the decoder NCTHW
    if expected is not None and channels is not None and channels != expected:
        log.warning(f'Decode: type=Tiny cls={cls} latents={channels}ch expected={expected}ch not supported')
        return None
    log.debug(f'Decode: type=Tiny cls={vae.__class__.__name__} variant="{variant}" latents={latents.shape}')
    vae = vae.to(device=devices.device, dtype=devices.dtype)
    latents = latents.transpose(1, 2).to(device=devices.device, dtype=devices.dtype)
    images = vae.decode_video(latents, parallel=False).transpose(1, 2)
    if type(vae).__name__ in UNIT_RANGE_DECODERS: # the pipelines expect a decode in [-1,1]
        images = images.mul_(2).sub_(1)
    log.debug(f'Decode: type=Tiny decoded={list(images.shape)} range={images.min().item():.3f}..{images.max().item():.3f}')
    return (images, None)
