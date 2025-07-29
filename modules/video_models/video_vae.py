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


def hijack_vae_decode(*args, **kwargs):
    shared.state.begin('VAE')
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
                latents = args[0].to(device=devices.device, dtype=shared.sd_model.vae.dtype) # upcast to vae dtype
                res = shared.sd_model.vae.orig_decode(latents, *args[1:], **kwargs)
                t1 = time.time()
                shared.log.debug(f'Decode: vae={shared.sd_model.vae.__class__.__name__} slicing={getattr(shared.sd_model.vae, "use_slicing", None)} tiling={getattr(shared.sd_model.vae, "use_tiling", None)} latents={list(latents.shape)}:{latents.device}:{latents.dtype} time={t1-t0:.3f}')
            else:
                res = shared.sd_model.vae.orig_decode(*args, **kwargs)
        except Exception as e:
            shared.log.error(f'Decode: type={vae_type} {e}')
            errors.display(e, 'vae')
            res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.state.end()
    return res


def hijack_vae_encode(*args, **kwargs):
    shared.state.begin('VAE')
    t0 = time.time()
    res = None
    if res is None:
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
        try:
            sd_models.move_model(shared.sd_model.vae, devices.device)
            if torch.is_tensor(args[0]):
                latents = args[0].to(device=devices.device, dtype=shared.sd_model.vae.dtype) # upcast to vae dtype
                res = shared.sd_model.vae.orig_encode(latents, *args[1:], **kwargs)
                t1 = time.time()
                shared.log.debug(f'Encode: vae={shared.sd_model.vae.__class__.__name__} slicing={getattr(shared.sd_model.vae, "use_slicing", None)} tiling={getattr(shared.sd_model.vae, "use_tiling", None)} latents={list(latents.shape)}:{latents.device}:{latents.dtype} time={t1-t0:.3f}')
            else:
                res = shared.sd_model.vae.orig_encode(*args, **kwargs)
        except Exception as e:
            shared.log.error(f'Encode: type={vae_type} {e}')
            errors.display(e, 'vae')
            res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.state.end()
    return res
