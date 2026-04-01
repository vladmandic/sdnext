import os
import time
import torch
from modules import shared, sd_models, devices, timer, errors
from modules.logger import log


debug = log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def hijack_vae_upscale(*args, **kwargs):
    import torch.nn.functional as F
    tensor = shared.sd_model.vae.orig_decode(*args, **kwargs)[0]
    tensor = F.pixel_shuffle(tensor.movedim(2, 1), upscale_factor=2).movedim(1, 2) # vae returns 16-dim latents, we need to pixel shuffle to 4-dim images
    tensor = tensor.unsqueeze(0)  # add batch dimension
    return tensor


def hijack_vae_decode(*args, **kwargs):
    jobid = shared.state.begin('VAE Decode')
    t0 = time.time()
    res = None
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    try:
        sd_models.move_model(shared.sd_model.vae, devices.device)
        if torch.is_tensor(args[0]):
            latents = args[0].to(device=devices.device, dtype=shared.sd_model.vae.dtype) # upcast to vae dtype
            if hasattr(shared.sd_model.vae, '_asymmetric_upscale_vae'):
                res = hijack_vae_upscale(latents, *args[1:], **kwargs)
            else:
                res = shared.sd_model.vae.orig_decode(latents, *args[1:], **kwargs)
            t1 = time.time()
            try:
                log.debug(f'Decode: vae={shared.sd_model.vae.__class__.__name__} dtype={latents.dtype} latents={list(latents.shape)}:{latents.device} decoded={list(res[0].shape)} slicing={getattr(shared.sd_model.vae, "use_slicing", None)} tiling={getattr(shared.sd_model.vae, "use_tiling", None)} time={t1-t0:.3f}')
            except Exception:
                pass
        else:
            res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    except Exception as e:
        log.error(f'Decode: vae={shared.sd_model.vae.__class__.__name__} {e}')
        errors.display(e, 'vae')
        res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.state.end(jobid)
    return res


def hijack_vae_encode(*args, **kwargs):
    jobid = shared.state.begin('VAE Encode')
    t0 = time.time()
    res = None
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    try:
        sd_models.move_model(shared.sd_model.vae, devices.device)
        if torch.is_tensor(args[0]):
            latents = args[0].to(device=devices.device, dtype=shared.sd_model.vae.dtype) # upcast to vae dtype
            res = shared.sd_model.vae.orig_encode(latents, *args[1:], **kwargs)
            t1 = time.time()
            log.debug(f'Encode: vae={shared.sd_model.vae.__class__.__name__} slicing={getattr(shared.sd_model.vae, "use_slicing", None)} tiling={getattr(shared.sd_model.vae, "use_tiling", None)} latents={list(latents.shape)}:{latents.device}:{latents.dtype} time={t1-t0:.3f}')
        else:
            res = shared.sd_model.vae.orig_encode(*args, **kwargs)
    except Exception as e:
        log.error(f'Encode: vae={shared.sd_model.vae.__class__.__name__} {e}')
        errors.display(e, 'vae')
        res = None
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.state.end(jobid)
    return res


def init_hijack(pipe):
    if pipe is not None and hasattr(pipe, 'vae') and hasattr(pipe.vae, 'decode') and not hasattr(pipe.vae, 'orig_decode'):
        pipe.vae.orig_decode = pipe.vae.decode
        pipe.vae.decode = hijack_vae_decode
    if pipe is not None and hasattr(pipe, 'vae') and hasattr(pipe.vae, 'encode') and not hasattr(pipe.vae, 'orig_encode'):
        pipe.vae.orig_encode = pipe.vae.encode
        pipe.vae.encode = hijack_vae_encode
