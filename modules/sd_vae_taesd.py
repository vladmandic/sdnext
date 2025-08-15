"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
import threading
from PIL import Image
import torch
from modules import devices, paths, shared


TAESD_MODELS = {
    'TAESD 1.3 Mocha Croissant': { 'fn': 'taesd_13_', 'uri': 'https://github.com/madebyollin/taesd/raw/7f572ca629c9b0d3c9f71140e5f501e09f9ea280', 'model': None },
    'TAESD 1.2 Chocolate-Dipped Shortbread': { 'fn': 'taesd_12_', 'uri': 'https://github.com/madebyollin/taesd/raw/8909b44e3befaa0efa79c5791e4fe1c4d4f7884e', 'model': None },
    'TAESD 1.1 Fruit Loops': { 'fn': 'taesd_11_', 'uri': 'https://github.com/madebyollin/taesd/raw/3e8a8a2ab4ad4079db60c1c7dc1379b4cc0c6b31', 'model': None },
    'TAESD 1.0': { 'fn': 'taesd_10_', 'uri': 'https://github.com/madebyollin/taesd/raw/88012e67cf0454e6d90f98911fe9d4aef62add86', 'model': None },
    'TAE HunyuanVideo': { 'fn': 'taehv.pth', 'uri': 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taehv.pth', 'model': None },
    'TAE WanVideo': { 'fn': 'taew1.pth', 'uri': 'https://github.com/madebyollin/taehv/raw/refs/heads/main/taew2_1.pth', 'model': None },
    'TAE MochiVideo': { 'fn': 'taem1.pth', 'uri': 'https://github.com/madebyollin/taem1/raw/refs/heads/main/taem1.pth', 'model': None },
}
CQYAN_MODELS = {
    'Hybrid-Tiny SD': {
        'sd': { 'repo': 'cqyan/hybrid-sd-tinyvae', 'model': None },
        'sdxl': { 'repo': 'cqyan/hybrid-sd-tinyvae-xl', 'model': None },
    },
    'Hybrid-Small SD': {
        'sd': { 'repo': 'cqyan/hybrid-sd-small-vae', 'model': None },
        'sdxl': { 'repo': 'cqyan/hybrid-sd-small-vae-xl', 'model': None },
    },
}

prev_warnings = False
prev_cls = ''
prev_type = ''
prev_model = ''
lock = threading.Lock()
supported = ['sd', 'sdxl', 'sd3', 'f1', 'h1', 'lumina2', 'hunyuanvideo', 'wanai', 'mochivideo', 'pixartsigma', 'pixartalpha', 'hunyuandit', 'omnigen', 'qwen']


def warn_once(msg, variant=None):
    variant = variant or shared.opts.taesd_variant
    global prev_warnings # pylint: disable=global-statement
    if not prev_warnings:
        prev_warnings = True
        shared.log.warning(f'Decode: type="taesd" variant="{variant}": {msg}')
    return Image.new('RGB', (8, 8), color = (0, 0, 0))


def get_model(model_type = 'decoder', variant = None):
    global prev_cls, prev_type, prev_model # pylint: disable=global-statement
    model_cls = shared.sd_model_type
    if model_cls is None or model_cls == 'none':
        return None, variant
    elif model_cls in {'ldm', 'pixartalpha'}:
        model_cls = 'sd'
    elif model_cls in {'pixartsigma', 'hunyuandit', 'omnigen', 'auraflow'}:
        model_cls = 'sdxl'
    elif model_cls in {'h1', 'lumina2', 'chroma'}:
        model_cls = 'f1'
    elif model_cls in {'wanai', 'qwen'}:
        variant = variant or 'TAE WanVideo'
    elif model_cls not in supported:
        warn_once(f'cls={shared.sd_model.__class__.__name__} type={model_cls} unsuppported', variant=variant)
        return None, variant
    variant = variant or shared.opts.taesd_variant
    folder = os.path.join(paths.models_path, "TAESD")
    dtype = devices.dtype_vae if devices.dtype_vae != torch.bfloat16 else torch.float16 # taesd does not support bf16
    os.makedirs(folder, exist_ok=True)
    if variant.startswith('TAE'):
        cfg = TAESD_MODELS[variant]
        if (model_cls == prev_cls) and (model_type == prev_type) and (variant == prev_model) and (cfg['model'] is not None):
            return cfg['model'], variant
        fn = os.path.join(folder, cfg['fn'] + model_type + '_' + model_cls + '.pth')
        if not os.path.exists(fn):
            uri = cfg['uri']
            if not uri.endswith('.pth'):
                uri += '/tae' + model_cls + '_' + model_type + '.pth'
            try:
                torch.hub.download_url_to_file(uri, fn)
                shared.log.info(f'Decode: type="taesd" variant="{variant}": uri="{uri}" fn="{fn}" download')
            except Exception as e:
                warn_once(f'download uri={uri} {e}', variant=variant)
        if os.path.exists(fn):
            prev_cls = model_cls
            prev_type = model_type
            prev_model = variant
            shared.log.debug(f'Decode: type="taesd" variant="{variant}" fn="{fn}" load')
            vae = None
            if 'TAE HunyuanVideo' in variant:
                from modules.taesd.taehv import TAEHV
                vae = TAEHV(checkpoint_path=fn)
            elif 'TAE WanVideo' in variant:
                from modules.taesd.taehv import TAEHV
                vae = TAEHV(checkpoint_path=fn)
            elif 'TAE MochiVideo' in variant:
                from modules.taesd.taem1 import TAEM1
                vae = TAEM1(checkpoint_path=fn)
            else:
                from modules.taesd.taesd import TAESD
                vae = TAESD(decoder_path=fn if model_type=='decoder' else None, encoder_path=fn if model_type=='encoder' else None)
            if vae is not None:
                vae = vae.to(devices.device, dtype=dtype)
                TAESD_MODELS[variant]['model'] = vae
            return vae, variant
    elif variant.startswith('Hybrid'):
        cfg = CQYAN_MODELS[variant].get(model_cls, None)
        if (model_cls == prev_cls) and (model_type == prev_type) and (variant == prev_model) and (cfg['model'] is not None):
            return cfg['model'], variant
        if cfg is None:
            warn_once(f'cls={shared.sd_model.__class__.__name__} type={model_cls} unsuppported', variant=variant)
            return None, variant
        repo = cfg['repo']
        prev_cls = model_cls
        prev_type = model_type
        prev_model = variant
        shared.log.debug(f'Decode: type="taesd" variant="{variant}" id="{repo}" load')
        if 'tiny' in repo:
            from diffusers.models import AutoencoderTiny
            vae = AutoencoderTiny.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir, torch_dtype=dtype)
        else:
            from modules.taesd.hybrid_small import AutoencoderSmall
            vae = AutoencoderSmall.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir, torch_dtype=dtype)
        vae = vae.to(devices.device, dtype=dtype)
        CQYAN_MODELS[variant][model_cls]['model'] = vae
        return vae, variant
    elif variant is None:
        warn_once(f'cls={shared.sd_model.__class__.__name__} type={model_cls} variant is none', variant=variant)
    else:
        warn_once(f'cls={shared.sd_model.__class__.__name__} type={model_cls} unsuppported', variant=variant)
    return None, variant


def decode(latents):
    with lock:
        vae, variant = get_model(model_type='decoder')
        if vae is None or max(latents.shape) > 256: # safetey check of large tensors
            return latents
        try:
            with devices.inference_context():
                dtype = devices.dtype_vae if devices.dtype_vae != torch.bfloat16 else torch.float16 # taesd does not support bf16
                tensor = latents.unsqueeze(0) if len(latents.shape) == 3 else latents
                tensor = tensor.detach().clone().to(devices.device, dtype=dtype)
                if variant.startswith('TAESD'):
                    image = vae.decoder(tensor).clamp(0, 1).detach()
                    return image[0]
                else:
                    image = vae.decode(tensor, return_dict=False)[0]
                    image = (image / 2.0 + 0.5).clamp(0, 1).detach()
                    return image
        except Exception as e:
            # from modules import errors
            # errors.display(e, 'taesd"')
            return warn_once(f'decode: {e}', variant=variant)


def encode(image):
    with lock:
        vae, variant = get_model(model_type='encoder')
        if vae is None:
            return image
        try:
            with devices.inference_context():
                latents = vae.encoder(image)
            return latents.detach()
        except Exception as e:
            return warn_once(f'encode: {e}', variant=variant)
