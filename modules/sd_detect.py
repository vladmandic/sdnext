import os
import time
import torch
import diffusers
from modules import shared, shared_items, devices, errors, model_tools


debug_load = os.environ.get('SD_LOAD_DEBUG', None)


def detect_pipeline(f: str, op: str = 'model', warning=True, quiet=False):
    guess = shared.opts.diffusers_pipeline
    warn = shared.log.warning if warning else lambda *args, **kwargs: None
    size = 0
    pipeline = None
    if guess == 'Autodetect':
        try:
            guess = 'Stable Diffusion XL' if 'XL' in f.upper() else 'Stable Diffusion'
            # guess by size
            if os.path.isfile(f) and f.endswith('.safetensors'):
                size = round(os.path.getsize(f) / 1024 / 1024)
                if (size > 0 and size < 128):
                    warn(f'Model size smaller than expected: {f} size={size} MB')
                elif (size >= 316 and size <= 324) or (size >= 156 and size <= 164): # 320 or 160
                    warn(f'Model detected as VAE model, but attempting to load as model: {op}={f} size={size} MB')
                    guess = 'VAE'
                elif (size >= 2002 and size <= 2038): # 2032
                    guess = 'Stable Diffusion 1.5'
                elif (size >= 3138 and size <= 3142): #3140
                    guess = 'Stable Diffusion XL'
                elif (size >= 3361 and size <= 3369): # 3368
                    guess = 'Stable Diffusion Upscale'
                elif (size >= 4891 and size <= 4899): # 4897
                    guess = 'Stable Diffusion XL Inpaint'
                elif (size >= 4970 and size <= 4976): # 4973
                    guess = 'Stable Diffusion 2' # SD v2 but could be eps or v-prediction
                elif (size >= 5791 and size <= 5799): # 5795
                    if op == 'model':
                        warn(f'Model detected as SD-XL refiner model, but attempting to load a base model: {op}={f} size={size} MB')
                    guess = 'Stable Diffusion XL Refiner'
                elif (size > 5692 and size < 5698) or (size > 4134 and size < 4138) or (size > 10362 and size < 10366) or (size > 15028 and size < 15228):
                    guess = 'Stable Diffusion 3'
                elif (size >= 6420 and size <= 7220): # 6420, IustriousRedux is 6541, monkrenRealisticINT_v10 is 7217
                    guess = 'Stable Diffusion XL'
                elif (size >= 9791 and size <= 9799): # 9794
                    guess = 'Stable Diffusion XL Instruct'
                elif (size >= 18414 and size <= 18420): # sd35-large aio
                    guess = 'Stable Diffusion 3'
                elif (size >= 20000 and size <= 40000):
                    guess = 'FLUX'
            # guess by name
            if 'instaflow' in f.lower():
                guess = 'InstaFlow'
            if 'segmoe' in f.lower():
                guess = 'SegMoE'
            if 'hunyuandit' in f.lower():
                guess = 'HunyuanDiT'
            if 'pixart-xl' in f.lower():
                guess = 'PixArt Alpha'
            if 'stable-diffusion-3' in f.lower():
                guess = 'Stable Diffusion 3'
            if 'stable-cascade' in f.lower() or 'stablecascade' in f.lower() or 'wuerstchen3' in f.lower() or ('sotediffusion' in f.lower() and "v2" in f.lower()):
                if devices.dtype == torch.float16:
                    warn('Stable Cascade does not support Float16')
                guess = 'Stable Cascade'
            if 'pixart-sigma' in f.lower():
                guess = 'PixArt Sigma'
            if 'sana' in f.lower():
                guess = 'Sana'
            if 'lumina-next' in f.lower():
                guess = 'Lumina-Next'
            if 'lumina-image-2' in f.lower():
                guess = 'Lumina2'
            if 'kolors' in f.lower():
                guess = 'Kolors'
            if 'auraflow' in f.lower():
                guess = 'AuraFlow'
            if 'cogview3' in f.lower():
                guess = 'CogView 3'
            if 'cogview4' in f.lower():
                guess = 'CogView 4'
            if 'meissonic' in f.lower():
                guess = 'Meissonic'
                pipeline = 'custom'
            if 'monetico' in f.lower():
                guess = 'Monetico'
                pipeline = 'custom'
            if 'omnigen' in f.lower():
                guess = 'OmniGen'
                pipeline = 'custom'
            if 'sd3' in f.lower():
                guess = 'Stable Diffusion 3'
            if 'hidream' in f.lower():
                guess = 'HiDream'
            if 'chroma' in f.lower():
                guess = 'Chroma'
            if 'flux' in f.lower() or 'flex.1' in f.lower() or 'lodestones' in f.lower():
                guess = 'FLUX'
                if size > 11000 and size < 16000:
                    warn(f'Model detected as FLUX UNET model, but attempting to load a base model: {op}={f} size={size} MB')
            if 'flex.2' in f.lower():
                guess = 'FLEX'
            if 'cosmos-predict2' in f.lower():
                guess = 'Cosmos'
            # guess for diffusers
            index = os.path.join(f, 'model_index.json')
            if os.path.exists(index) and os.path.isfile(index):
                index = shared.readfile(index, silent=True)
                cls = index.get('_class_name', None)
                if cls is not None:
                    pipeline = getattr(diffusers, cls, None)
                    if pipeline is None:
                        pipeline = cls
                if callable(pipeline) and 'Flux' in pipeline.__name__ and guess != 'FLEX':
                    guess = 'FLUX'
                if callable(pipeline) and 'StableDiffusion3' in pipeline.__name__:
                    guess = 'Stable Diffusion 3'
                if callable(pipeline) and 'Lumina2' in pipeline.__name__:
                    guess = 'Lumina 2'

            # switch for specific variant
            if guess == 'Stable Diffusion' and 'inpaint' in f.lower():
                guess = 'Stable Diffusion Inpaint'
            elif guess == 'Stable Diffusion' and 'instruct' in f.lower():
                guess = 'Stable Diffusion Instruct'
            if guess == 'Stable Diffusion XL' and 'inpaint' in f.lower():
                guess = 'Stable Diffusion XL Inpaint'
            elif guess == 'Stable Diffusion XL' and 'instruct' in f.lower():
                guess = 'Stable Diffusion XL Instruct'

            # get actual pipeline
            pipeline = shared_items.get_pipelines().get(guess, None) if pipeline is None else pipeline
            if debug_load is not None:
                shared.log.info(f'Autodetect {op}: detect="{guess}" class={getattr(pipeline, "__name__", None)} file="{f}" size={size}MB')
                t0 = time.time()
                keys = model_tools.get_safetensor_keys(f)
                if keys is not None and len(keys) > 0:
                    modules = model_tools.list_to_dict(keys)
                    modules = model_tools.remove_entries_after_depth(modules, 3)
                    lst = model_tools.list_compact(keys)
                    t1 = time.time()
                    shared.log.debug(f'Autodetect: modules={modules} list={lst} time={t1-t0:.2f}')
        except Exception as e:
            shared.log.error(f'Autodetect {op}: file="{f}" {e}')
            if debug_load:
                errors.display(e, f'Load {op}: {f}')
            return None, None
    else:
        try:
            size = round(os.path.getsize(f) / 1024 / 1024)
            pipeline = shared_items.get_pipelines().get(guess, None) if pipeline is None else pipeline
            if not quiet:
                shared.log.info(f'Load {op}: detect="{guess}" class={getattr(pipeline, "__name__", None)} file="{f}" size={size}MB')
        except Exception as e:
            shared.log.error(f'Load {op}: detect="{guess}" file="{f}" {e}')

    if pipeline is None:
        shared.log.warning(f'Load {op}: detect="{guess}" file="{f}" size={size} not recognized')
        pipeline = diffusers.StableDiffusionPipeline
    return pipeline, guess


def get_load_config(model_file, model_type, config_type='yaml'):
    if config_type == 'yaml':
        yaml = os.path.splitext(model_file)[0] + '.yaml'
        if os.path.exists(yaml):
            return yaml
        if model_type == 'Stable Diffusion':
            return 'configs/v1-inference.yaml'
        if model_type == 'Stable Diffusion XL':
            return 'configs/sd_xl_base.yaml'
        if model_type == 'Stable Diffusion XL Refiner':
            return 'configs/sd_xl_refiner.yaml'
        if model_type == 'Stable Diffusion 2':
            return None # dont know if its eps or v so let diffusers sort it out
            # return 'configs/v2-inference-512-base.yaml'
            # return 'configs/v2-inference-768-v.yaml'
    elif config_type == 'json':
        if not shared.opts.diffuser_cache_config:
            return None
        if model_type == 'Stable Diffusion':
            return 'configs/sd15'
        if model_type == 'Stable Diffusion XL':
            return 'configs/sdxl'
        if model_type == 'Stable Diffusion XL Refiner':
            return 'configs/sdxl-refiner'
        if model_type == 'Stable Diffusion 3':
            return 'configs/sd3'
        if model_type == 'FLUX':
            return 'configs/flux'
    return None
