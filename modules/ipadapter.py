"""
Lightweight IP-Adapter applied to existing pipeline in Diffusers
- Downloads image_encoder or first usage (2.5GB)
- Introduced via: https://github.com/huggingface/diffusers/pull/5713
- IP adapters: https://huggingface.co/h94/IP-Adapter
"""

from __future__ import annotations
import os
import time
import json
from typing import TYPE_CHECKING
from PIL import Image
import transformers
from modules import processing, shared, devices, sd_models, errors, model_quant
from modules.logger import log

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline


clip_loaded = None
adapters_loaded = []
CLIP_ID = "h94/IP-Adapter"
OPEN_ID = "openai/clip-vit-large-patch14"
SIGLIP_ID = 'google/siglip-so400m-patch14-384'
ADAPTERS_NONE = {
    'None': { 'name': 'none', 'repo': 'none', 'subfolder': 'none' },
}
ADAPTERS_SD15 = {
    'None': { 'name': 'none', 'repo': 'none', 'subfolder': 'none' },
    'Base': { 'name': 'ip-adapter_sd15.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'models' },
    'Base ViT-G': { 'name': 'ip-adapter_sd15_vit-G.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'models' },
    'Light': { 'name': 'ip-adapter_sd15_light.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'models' },
    'Plus': { 'name': 'ip-adapter-plus_sd15.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'models' },
    'Plus Face': { 'name': 'ip-adapter-plus-face_sd15.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'models' },
    'Full Face': { 'name': 'ip-adapter-full-face_sd15.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'models' },
    'Ostris Composition ViT-H': { 'name': 'ip_plus_composition_sd15.safetensors', 'repo': 'ostris/ip-composition-adapter', 'subfolder': '' },
}
ADAPTERS_SDXL = {
    'None': { 'name': 'none', 'repo': 'none', 'subfolder': 'none' },
    'Base SDXL': { 'name': 'ip-adapter_sdxl.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'sdxl_models' },
    'Base ViT-H SDXL': { 'name': 'ip-adapter_sdxl_vit-h.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'sdxl_models' },
    'Plus ViT-H SDXL': { 'name': 'ip-adapter-plus_sdxl_vit-h.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'sdxl_models' },
    'Plus Face ViT-H SDXL': { 'name': 'ip-adapter-plus-face_sdxl_vit-h.safetensors', 'repo': 'h94/IP-Adapter', 'subfolder': 'sdxl_models' },
    'Ostris Composition ViT-H SDXL': { 'name': 'ip_plus_composition_sdxl.safetensors', 'repo': 'ostris/ip-composition-adapter', 'subfolder': '' },
}
ADAPTERS_SD3 = {
    'None': { 'name': 'none', 'repo': 'none', 'subfolder': 'none' },
    'InstantX Large': { 'name': 'ip-adapter_diffusers.safetensors', 'repo': 'InstantX/SD3.5-Large-IP-Adapter', 'subfolder': 'none', 'revision': 'refs/pr/10' },
}
ADAPTERS_F1 = {
    'None': { 'name': 'none', 'repo': 'none', 'subfolder': 'none' },
    'XLabs AI v1': { 'name': 'ip_adapter.safetensors', 'repo': 'XLabs-AI/flux-ip-adapter', 'subfolder': 'none' },
    'XLabs AI v2': { 'name': 'ip_adapter.safetensors', 'repo': 'XLabs-AI/flux-ip-adapter-v2', 'subfolder': 'none' },
}
ADAPTERS = { **ADAPTERS_SD15, **ADAPTERS_SDXL, **ADAPTERS_SD3, **ADAPTERS_F1 }
ADAPTERS_ALL = { **ADAPTERS_SD15, **ADAPTERS_SDXL, **ADAPTERS_SD3, **ADAPTERS_F1 }


def get_adapters():
    global ADAPTERS # pylint: disable=global-statement
    if shared.sd_model_type == 'sd':
        ADAPTERS = ADAPTERS_SD15
    elif shared.sd_model_type == 'sdxl':
        ADAPTERS = ADAPTERS_SDXL
    elif shared.sd_model_type == 'sd3':
        ADAPTERS = ADAPTERS_SD3
    elif shared.sd_model_type == 'f1':
        ADAPTERS = ADAPTERS_F1
    else:
        ADAPTERS = ADAPTERS_NONE
    return list(ADAPTERS)


def get_images(input_images):
    output_images = []
    if input_images is None or len(input_images) == 0:
        log.error('IP adapter: no init images')
        return None
    if shared.sd_model_type not in ['sd', 'sdxl', 'sd3', 'f1']:
        log.error('IP adapter: base model not supported')
        return None
    if isinstance(input_images, str):
        from modules.api.api import decode_base64_to_image
        input_images = decode_base64_to_image(input_images).convert("RGB")
    input_images = input_images.copy()
    if not isinstance(input_images, list):
        input_images = [input_images]
    for image in input_images:
        if image is None:
            continue
        if isinstance(image, list):
            output_images.append(get_images(image)) # recursive
        elif isinstance(image, Image.Image):
            output_images.append(image)
        elif isinstance(image, str):
            from modules.api.api import decode_base64_to_image
            decoded_image = decode_base64_to_image(image).convert("RGB")
            output_images.append(decoded_image)
        elif hasattr(image, 'name'): # gradio gallery entry
            pil_image = Image.open(image.name)
            pil_image.load()
            output_images.append(pil_image)
        else:
            log.error(f'IP adapter: unknown input: {image}')
    return output_images


def get_scales(adapter_scales, adapter_images):
    output_scales = [adapter_scales] if not isinstance(adapter_scales, list) else adapter_scales
    while len(output_scales) < len(adapter_images):
        output_scales.append(output_scales[-1])
    return output_scales


def get_crops(adapter_crops, adapter_images):
    output_crops = [adapter_crops] if not isinstance(adapter_crops, list) else adapter_crops
    while len(output_crops) < len(adapter_images):
        output_crops.append(output_crops[-1])
    return output_crops


def crop_images(images, crops):
    try:
        for i in range(len(images)):
            if crops[i]:
                from modules.shared import yolo # pylint: disable=no-name-in-module
                cropped = []
                for image in images[i]:
                    faces = yolo.predict('face-yolo8n', image)
                    if len(faces) > 0:
                        cropped.append(faces[0].item)
                if len(cropped) == len(images[i]):
                    images[i] = cropped
                else:
                    log.error(f'IP adapter: failed to crop image: source={len(images[i])} faces={len(cropped)}')
    except Exception as e:
        log.error(f'IP adapter: failed to crop image: {e}')
    if shared.sd_model_type == 'sd3' and len(images) == 1:
        return images[0]
    return images


def unapply(pipe, unload: bool = False): # pylint: disable=arguments-differ
    if len(adapters_loaded) == 0:
        return
    try:
        if hasattr(pipe, 'set_ip_adapter_scale'):
            pipe.set_ip_adapter_scale(0)
            if unload:
                log.debug('IP adapter unload')
                pipe.unload_ip_adapter()
        if hasattr(pipe, 'unet') and pipe.unet is not None:
            module = pipe.unet
        elif hasattr(pipe, 'transformer'):
            module = pipe.transformer
        else:
            module = None
        if module is not None and hasattr(module, 'config') and module.config.encoder_hid_dim_type == 'ip_image_proj':
            pipe.unet.encoder_hid_proj = None
            pipe.config.encoder_hid_dim_type = None
            pipe.unet.set_default_attn_processor()
    except Exception:
        pass


def load_image_encoder(pipe: DiffusionPipeline, adapter_names: list[str]):
    global clip_loaded # pylint: disable=global-statement
    for adapter_name in adapter_names:
        # which clip to use
        clip_repo = CLIP_ID
        if 'ViT' not in adapter_name: # defaults per model
            clip_subfolder = 'models/image_encoder' if shared.sd_model_type == 'sd' else 'sdxl_models/image_encoder'
        if 'ViT-H' in adapter_name:
            clip_subfolder = 'models/image_encoder' # this is vit-h
        elif 'ViT-G' in adapter_name:
            clip_subfolder = 'sdxl_models/image_encoder' # this is vit-g
        else:
            if shared.sd_model_type == 'sd':
                clip_subfolder = 'models/image_encoder'
            elif shared.sd_model_type == 'sdxl':
                clip_subfolder = 'sdxl_models/image_encoder'
            elif shared.sd_model_type == 'sd3':
                clip_repo = SIGLIP_ID
                clip_subfolder = None
            elif shared.sd_model_type == 'f1':
                clip_repo = OPEN_ID
                clip_subfolder = None
            else:
                log.error(f'IP adapter: unknown model type: {adapter_name}')
                return False

    # load image encoder used by ip adapter
    if pipe.image_encoder is None or clip_loaded != f'{clip_repo}/{clip_subfolder}':
        jobid = shared.state.begin('Load encoder')
        try:
            offline_config = { 'local_files_only': True } if shared.opts.offline_mode else {}
            if shared.sd_model_type == 'sd3':
                image_encoder = transformers.SiglipVisionModel.from_pretrained(clip_repo, torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, **offline_config)
            else:
                if clip_subfolder is None:
                    image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(clip_repo, torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, use_safetensors=True, **offline_config)
                    log.debug(f'IP adapter load: encoder="{clip_repo}" cls={pipe.image_encoder.__class__.__name__}')
                else:
                    image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(clip_repo, subfolder=clip_subfolder, torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, use_safetensors=True, **offline_config)
                    log.debug(f'IP adapter load: encoder="{clip_repo}/{clip_subfolder}" cls={pipe.image_encoder.__class__.__name__}')
            sd_models.clear_caches()
            image_encoder = model_quant.do_post_load_quant(image_encoder, allow=True)
            if hasattr(pipe, 'register_modules'):
                pipe.register_modules(image_encoder=image_encoder)
            else:
                pipe.image_encoder = image_encoder
            clip_loaded = f'{clip_repo}/{clip_subfolder}'
        except Exception as e:
            log.error(f'IP adapter load: encoder="{clip_repo}/{clip_subfolder}" {e}')
            errors.display(e, 'IP adapter: type=encoder')
            return False
        shared.state.end(jobid)
    sd_models.move_model(pipe.image_encoder, devices.device)
    return True


def load_feature_extractor(pipe):
    # load feature extractor used by ip adapter
    if pipe.feature_extractor is None:
        try:
            jobid = shared.state.begin('Load extractor')
            offline_config = { 'local_files_only': True } if shared.opts.offline_mode else {}
            if shared.sd_model_type == 'sd3':
                feature_extractor = transformers.SiglipImageProcessor.from_pretrained(SIGLIP_ID, torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir, **offline_config)
            else:
                feature_extractor = transformers.CLIPImageProcessor()
            if hasattr(pipe, 'register_modules'):
                pipe.register_modules(feature_extractor=feature_extractor)
            else:
                pipe.feature_extractor = feature_extractor
                sd_models.apply_balanced_offload(pipe.feature_extractor)
            log.debug(f'IP adapter load: extractor={pipe.feature_extractor.__class__.__name__}')
        except Exception as e:
            log.error(f'IP adapter load: extractor {e}')
            errors.display(e, 'IP adapter: type=extractor')
            return False
        shared.state.end(jobid)
    return True


def parse_params(p: processing.StableDiffusionProcessing, adapters: list, adapter_scales: list[float], adapter_crops: list[bool], adapter_starts: list[float], adapter_ends: list[float], adapter_images: list):
    if hasattr(p, 'ip_adapter_scales'):
        adapter_scales = p.ip_adapter_scales
    if hasattr(p, 'ip_adapter_crops'):
        adapter_crops = p.ip_adapter_crops
    if hasattr(p, 'ip_adapter_starts'):
        adapter_starts = p.ip_adapter_starts
    if hasattr(p, 'ip_adapter_ends'):
        adapter_ends = p.ip_adapter_ends
    if hasattr(p, 'ip_adapter_images'):
        adapter_images = p.ip_adapter_images
    adapter_images = get_images(adapter_images)
    if hasattr(p, 'ip_adapter_masks') and len(p.ip_adapter_masks) > 0:
        adapter_masks = p.ip_adapter_masks
        adapter_masks = get_images(adapter_masks)
    else:
        adapter_masks = []
    if len(adapter_masks) > 0:
        from diffusers.image_processor import IPAdapterMaskProcessor
        mask_processor = IPAdapterMaskProcessor()
        for i in range(len(adapter_masks)):
            adapter_masks[i] = mask_processor.preprocess(adapter_masks[i], height=p.height, width=p.width)
        adapter_masks = mask_processor.preprocess(adapter_masks, height=p.height, width=p.width)
    if adapter_images is None:
        log.error('IP adapter: no image provided')
        return [], [], [], [], [], []
    if len(adapters) < len(adapter_images):
        adapter_images = adapter_images[:len(adapters)]
    if len(adapters) < len(adapter_masks):
        adapter_masks = adapter_masks[:len(adapters)]
    if len(adapter_masks) > 0 and len(adapter_masks) != len(adapter_images):
        log.error('IP adapter: image and mask count mismatch')
        return [], [], [], [], [], []
    adapter_scales = get_scales(adapter_scales, adapter_images)
    p.ip_adapter_scales = adapter_scales.copy()
    adapter_crops = get_crops(adapter_crops, adapter_images)
    p.ip_adapter_crops = adapter_crops.copy()
    adapter_starts = get_scales(adapter_starts, adapter_images)
    p.ip_adapter_starts = adapter_starts.copy()
    adapter_ends = get_scales(adapter_ends, adapter_images)
    p.ip_adapter_ends = adapter_ends.copy()
    return adapter_images, adapter_masks, adapter_scales, adapter_crops, adapter_starts, adapter_ends


def apply(pipe, p: processing.StableDiffusionProcessing, adapter_names=None, adapter_scales=None, adapter_crops=None, adapter_starts=None, adapter_ends=None, adapter_images=None):
    if adapter_images is None:
        adapter_images = []
    if adapter_ends is None:
        adapter_ends = [1.0]
    if adapter_starts is None:
        adapter_starts = [0.0]
    if adapter_crops is None:
        adapter_crops = [False]
    if adapter_scales is None:
        adapter_scales = [1.0]
    if adapter_names is None:
        adapter_names = []
    global adapters_loaded # pylint: disable=global-statement
    # overrides
    if hasattr(p, 'ip_adapter_names'):
        if isinstance(p.ip_adapter_names, str):
            p.ip_adapter_names = [p.ip_adapter_names]
        adapters = [ADAPTERS_ALL.get(adapter_name, None) for adapter_name in p.ip_adapter_names if adapter_name is not None and adapter_name.lower() != 'none']
        adapter_names = p.ip_adapter_names
    else:
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        adapters = [ADAPTERS.get(adapter_name, None) for adapter_name in adapter_names if adapter_name.lower() != 'none']

    if len(adapters) == 0:
        unapply(pipe, getattr(p, 'ip_adapter_unload', False))
        if hasattr(p, 'ip_adapter_images'):
            del p.ip_adapter_images
        return False
    if shared.sd_model_type not in ['sd', 'sdxl', 'sd3', 'f1']:
        log.error(f'IP adapter: model={shared.sd_model_type} class={pipe.__class__.__name__} not supported')
        return False

    adapter_images, adapter_masks, adapter_scales, adapter_crops, adapter_starts, adapter_ends = parse_params(p, adapters, adapter_scales, adapter_crops, adapter_starts, adapter_ends, adapter_images)

    # init code
    if pipe is None:
        return False
    if len(adapter_images) == 0:
        log.error('IP adapter: no image provided')
        adapters = [] # unload adapter if previously loaded as it will cause runtime errors
    if len(adapters) == 0:
        unapply(pipe, getattr(p, 'ip_adapter_unload', False))
        if hasattr(p, 'ip_adapter_images'):
            del p.ip_adapter_images
        return False
    if not hasattr(pipe, 'load_ip_adapter'):
        log.error(f'IP adapter: pipeline not supported: {pipe.__class__.__name__}')
        return False

    if not load_image_encoder(pipe, adapter_names):
        return False

    if not load_feature_extractor(pipe):
        return False

    # main code
    try:
        t0 = time.time()
        repos = [adapter.get('repo', None) for adapter in adapters if adapter.get('repo', 'none') != 'none']
        subfolders = [adapter.get('subfolder', None) for adapter in adapters if adapter.get('subfolder', 'none') != 'none']
        names = [adapter.get('name', None) for adapter in adapters if adapter.get('name', 'none') != 'none']
        revisions = [adapter.get('revision', None) for adapter in adapters if adapter.get('revision', 'none') != 'none']
        kwargs = {}
        if len(repos) == 1:
            repos = repos[0]
        if len(subfolders) > 0:
            kwargs['subfolder'] = subfolders if len(subfolders) > 1 else subfolders[0]
        if len(names) > 0:
            kwargs['weight_name'] = names if len(names) > 1 else names[0]
        if len(revisions) > 0:
            kwargs['revision'] = revisions[0]
        if shared.opts.offline_mode:
            kwargs["local_files_only"] = True
        pipe.load_ip_adapter(repos, **kwargs)
        adapters_loaded = names
        if hasattr(p, 'ip_adapter_layers'):
            pipe.set_ip_adapter_scale(p.ip_adapter_layers)
            ip_str = ';'.join(adapter_names) + ':' + json.dumps(p.ip_adapter_layers)
        else:
            for i in range(len(adapter_scales)):
                if adapter_starts[i] > 0:
                    adapter_scales[i] = 0.00
            pipe.set_ip_adapter_scale(adapter_scales if len(adapter_scales) > 1 else adapter_scales[0])
            ip_str =  [f'{os.path.splitext(adapter)[0]}:{scale}:{start}:{end}:{crop}' for adapter, scale, start, end, crop in zip(adapter_names, adapter_scales, adapter_starts, adapter_ends, adapter_crops, strict=False)]
        if hasattr(pipe, 'transformer') and 'Nunchaku' in pipe.transformer.__class__.__name__:
            if isinstance(repos, str):
                sd_models.clear_caches(full=True)
                import accelerate
                accelerate.hooks.remove_hook_from_module(pipe.transformer, recurse=True)
                pipe.transformer = pipe.transformer.to(devices.device)
                from nunchaku.models.ip_adapter.diffusers_adapters import apply_IPA_on_pipe
                apply_IPA_on_pipe(pipe, ip_adapter_scale=adapter_scales[0], repo_id=repos)
                pipe = sd_models.apply_balanced_offload(pipe)
                log.debug(f'IP adapter load: engine=nunchaku scale={adapter_scales[0]} repo="{repos}"')
            else:
                log.error('IP adapter: Nunchaku only supports single adapter')
        p.task_args['ip_adapter_image'] = crop_images(adapter_images, adapter_crops)
        if len(adapter_masks) > 0:
            p.cross_attention_kwargs = { 'ip_adapter_masks': adapter_masks }
        p.extra_generation_params["IP Adapter"] = ';'.join(ip_str)
        t1 = time.time()
        log.info(f'IP adapter: {ip_str} image={adapter_images} mask={adapter_masks is not None} time={t1-t0:.2f}')
    except Exception as e:
        log.error(f'IP adapter load: adapters={adapter_names} repo={repos} folders={subfolders} names={names} {e}')
        errors.display(e, 'IP adapter: type=adapter')
    return True
