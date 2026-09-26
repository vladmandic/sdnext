"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
import torch
from modules import devices, paths, shared
from modules.logger import log
from modules.vae.model_map import get_vae_type


debug = os.environ.get('SD_VAE_DEBUG', None) is not None
warned = False
loaded_vae = None
loaded_cls = None
loaded_type = None
dtype = devices.dtype_vae if (devices.dtype_vae != torch.bfloat16) else torch.float16 # taesd does not play nice with bfloat16


taesd_map = {
    # https://github.com/madebyollin/taesd
    'sd': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth'),
    'sdxl': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth'),
    'sd3': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taesd3_decoder.pth'),
    'f1': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taef1_decoder.pth'),
    'f2': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taef2_decoder.pth'),
    'qwen21': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taeqi2_1_decoder.pth'),
    'sana': ('taesd', 'https://github.com/madebyollin/taesd/raw/main/taesana_decoder.pth'),
    # https://github.com/madebyollin/taehv
    'hunyuanvideo': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taehv.pth'),
    'hunyuanvideo15': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taehv1_5.pth'),
    'wan21': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taew2_1.pth'),
    'wan22': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taew2_2.pth'),
    'minimaxh3': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taeh3.pth'),
    'cogvideo': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taecvx.pth'),
    'opensora': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taeos1_3.pth'),
    'ltxvideo': ('taehv', 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taeltx2_3.pth'),
    # https://github.com/madebyollin/taem1
    'mochivideo': ('taem1', 'https://github.com/madebyollin/taem1/raw/refs/heads/main/taem1.pth'),
}


def restore_preview_size(image, vae):
    # TAESD (image) and TAEHV (video) drop spatial upsample blocks when taesd_layers < 3, shrinking output 2x/4x.
    # Rescale spatial dims so preview size stays constant. Other taes (TAEM1, Hybrid) ignore taesd_layers, so skip them.
    from modules.taesd.taesd import TAESD
    from modules.taesd.taehv import TAEHV
    layers = shared.opts.taesd_layers
    if layers >= 3 or not isinstance(vae, (TAESD, TAEHV)) or not isinstance(image, torch.Tensor) or image.ndim < 3 or image.shape[-3] != 3:
        return image
    try:
        frames = image.reshape(-1, *image.shape[-3:]) # flatten any leading dims to a batch of CHW frames
        frames = torch.nn.functional.interpolate(frames, scale_factor=float(2 ** (3 - layers)), mode='bilinear', align_corners=False)
        image = frames.reshape(*image.shape[:-2], frames.shape[-2], frames.shape[-1])
    except Exception:
        pass
    return image


def load_model():
    global loaded_cls, loaded_vae, loaded_type # pylint: disable=global-statement
    if not shared.sd_loaded:
        return None
    vae_type = get_vae_type()
    if vae_type is None: # try direct lookup
        if shared.sd_model_type in list(taesd_map.keys()):
            vae_type = shared.sd_model_type
    if vae_type is None:
        return None
    vae_record = taesd_map.get(vae_type, None)
    if vae_record is None:
        return None
    vae_cls, vae_uri = vae_record
    new_line = '\n'

    if (vae_cls == loaded_cls) and (vae_type == loaded_type) and (loaded_vae is not None):
        if debug:
            log.trace(f'{new_line}Decode: type=Tiny model={vae_type} cls={loaded_vae.__class__.__name__} cached')
            new_line = ''
        return loaded_vae

    vae_folder = os.path.join(paths.models_path, "Preview")
    vae_file = os.path.join(vae_folder, vae_uri.split('/')[-1])
    if not os.path.exists(vae_file):
        log.debug(f'{new_line}Decode: type=Tiny model={vae_type} url="{vae_uri}" download')
        new_line = ''
        os.makedirs(vae_folder, exist_ok=True)
        torch.hub.download_url_to_file(vae_uri, vae_file)
    if not os.path.exists(vae_file):
        new_line = ''
        log.error(f'{new_line}Decode: type=Tiny model={vae_type} file="{vae_file}" not found')
        return None

    if vae_cls == 'taehv':
        from modules.taesd.taehv import TAEHV
        vae_model = TAEHV(checkpoint_path=vae_file)
    elif vae_cls == 'taem1':
        from modules.taesd.taem1 import TAEM1
        vae_model = TAEM1(checkpoint_path=vae_file)
    else:
        from modules.taesd.taesd import TAESD
        vae_model = TAESD(decoder_path=vae_file, encoder_path=None)

    loaded_cls = vae_cls
    loaded_type = vae_type
    loaded_vae = vae_model.to(devices.device, dtype=dtype)
    log.debug(f'{new_line}Decode: type=Tiny model={vae_type} cls={loaded_vae.__class__.__name__} group={loaded_cls} loaded')
    return loaded_vae


def decode(latents):
    if (latents is None) or (not shared.sd_loaded) or (max(latents.shape) > 384):
        return latents
    vae = load_model()
    if vae is None:
        return latents
    with devices.inference_context():
        latents = latents.unsqueeze(0) if len(latents.shape) == 3 else latents
        latents = latents.to(devices.device, dtype=dtype)
        if debug:
            log.trace(f'\nDecode: type=Tiny model={loaded_type} cls={loaded_vae.__class__.__name__} group={loaded_cls} shape={latents.shape}')
        if loaded_type == 'f2' and (len(latents.shape) == 4) and (latents.shape[1] == 128):
            b, _c, h, w = latents.shape
            latents = latents.reshape(b, 32, h * 2, w * 2)
        if loaded_cls == 'taesd':
            image = vae.decoder(latents)[0]
            image = image.clamp(0, 1).detach()
        else:
            image = vae.decode(latents, return_dict=False)[0]
            if image.ndim == 4 and image.shape[0] > 1 and image.shape[1] == 3: # likely a video latent
                image = image[0] # just take the first frame for now
        image = image.clamp(0, 1).detach()
        image = restore_preview_size(image, vae)
        return image


"""
def encode(image):
    with lock:
        vae, variant = load_model(model_type='encoder')
        if vae is None:
            return image
        try:
            with devices.inference_context():
                latents = vae.encoder(image)
            return latents.detach()
        except Exception as e:
            return warn_once(f'encode: {e}', variant=variant)
"""
