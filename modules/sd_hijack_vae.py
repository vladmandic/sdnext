import os
import time
import torch
from modules import shared, sd_models, devices, timer, errors


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def hijack_vae_decode(*args, **kwargs):
    shared.state.begin('VAE')
    t0 = time.time()
    res = None
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
        shared.log.error(f'Decode: vae={shared.sd_model.vae.__class__.__name__} {e}')
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
        shared.log.error(f'Encode: vae={shared.sd_model.vae.__class__.__name__} {e}')
        errors.display(e, 'vae')
        res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.state.end()
    return res


def init_hijack(pipe):
    if pipe is not None and hasattr(pipe, 'vae') and hasattr(pipe.vae, 'decode') and not hasattr(pipe.vae, 'orig_decode'):
        pipe.vae.orig_decode = pipe.vae.decode
        pipe.vae.decode = hijack_vae_decode
    if pipe is not None and hasattr(pipe, 'vae') and hasattr(pipe.vae, 'encode') and not hasattr(pipe.vae, 'orig_encode'):
        pipe.vae.orig_encode = pipe.vae.encode
        pipe.vae.encode = hijack_vae_encode
