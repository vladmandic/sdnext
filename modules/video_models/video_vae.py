import os
import time
import torch
from modules import shared, sd_models, devices, timer, errors


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None
vae_type = None


def set_vae_params(p):
    global vae_type # pylint: disable=global-statement
    vae_type = p.vae_type
    if p.vae_tile_frames > p.frames:
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
    else:
        shared.log.warning(f'Video VAE: type=Tiny cls={shared.sd_model.__class__.__name__} not supported')
        return None
    from modules import sd_vae_taesd
    vae = sd_vae_taesd.get_model(variant)
    if vae is None:
        return None
    debug(f'Video VAE: type=Tiny cls={vae.__class__.__name__} variant="{variant}" latents={latents.shape}')
    vae = vae.to(device=devices.device, dtype=devices.dtype)
    latents = latents.transpose(1, 2).to(device=devices.device, dtype=devices.dtype)
    images = vae.decode_video(latents, parallel=False).transpose(1, 2).mul_(2).sub_(1)
    images = images.transpose(1, 2).mul_(2).sub_(1)
    return (images, None)


def hijack_vae_decode(*args, **kwargs):
    t0 = time.time()
    res = None
    if vae_type == 'Tiny':
        res = vae_decode_tiny(args[0])
    if vae_type == 'Remote':
        pass
    if res is None:
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
        try:
            sd_models.move_model(shared.sd_model.vae, devices.device)
            if torch.is_tensor(args[0]):
                latent = args[0]
                latent = latent.to(device=devices.device, dtype=shared.sd_model.vae.dtype) # upcast to vae dtype
                res = shared.sd_model.vae.orig_decode(latent, *args[1:], **kwargs)
            else:
                res = shared.sd_model.vae.orig_decode(*args, **kwargs)
        except Exception as e:
            shared.log.error(f'Video VAE decode: type={vae_type} {e}')
            errors.display(e, 'Video VAE')
            res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    debug(f'Video VAE decode: type={vae_type} vae={shared.sd_model.vae.__class__.__name__} latents={args[0].shape} time={t1-t0:.2f}')
    return res


def hijack_vae_encode(*args, **kwargs):
    t0 = time.time()
    res = None
    if res is None:
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
        try:
            sd_models.move_model(shared.sd_model.vae, devices.device)
            if torch.is_tensor(args[0]):
                latent = args[0]
                latent = latent.to(device=devices.device, dtype=shared.sd_model.vae.dtype) # upcast to vae dtype
                res = shared.sd_model.vae.orig_encode(latent, *args[1:], **kwargs)
            else:
                res = shared.sd_model.vae.orig_encode(*args, **kwargs)
        except Exception as e:
            shared.log.error(f'Video VAE encode: type={vae_type} {e}')
            errors.display(e, 'Video VAE')
            res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    debug(f'Video VAE encode: type={vae_type} vae={shared.sd_model.vae.__class__.__name__} latents={args[0].shape} time={t1-t0:.2f}')
    return res
