import os
import time
import torch
import diffusers
from modules import shared, shared_items, devices, errors, model_tools


debug_load = os.environ.get('SD_LOAD_DEBUG', None)


def guess_by_size(fn, current_guess):
    new_guess = None
    if os.path.isfile(fn) and fn.endswith('.safetensors'):
        size = round(os.path.getsize(fn) / 1024 / 1024)
        if (size > 0 and size < 128):
            shared.log.warning(f'Model size smaller than expected: file="{fn}" size={size} MB')
        elif (size >= 316 and size <= 324) or (size >= 156 and size <= 164): # 320 or 160
            shared.log.warning(f'Model detected as VAE model, but attempting to load as model: file="{fn}" size={size} MB')
            new_guess = 'VAE'
        elif (size >= 2002 and size <= 2038): # 2032
            new_guess = 'Stable Diffusion 1.5'
        elif (size >= 3138 and size <= 3142): #3140
            new_guess = 'Stable Diffusion XL'
        elif (size >= 3361 and size <= 3369): # 3368
            new_guess = 'Stable Diffusion Upscale'
        elif (size >= 4891 and size <= 4899): # 4897
            new_guess = 'Stable Diffusion XL Inpaint'
        elif (size >= 4970 and size <= 4976): # 4973
            new_guess = 'Stable Diffusion 2' # SD v2 but could be eps or v-prediction
        elif (size >= 5791 and size <= 5799): # 5795
            new_guess = 'Stable Diffusion XL Refiner'
        elif (size > 5692 and size < 5698) or (size > 4134 and size < 4138) or (size > 10362 and size < 10366) or (size > 15028 and size < 15228):
            new_guess = 'Stable Diffusion 3'
        elif (size >= 6420 and size <= 7220): # 6420, IustriousRedux is 6541, monkrenRealisticINT_v10 is 7217
            new_guess = 'Stable Diffusion XL'
        elif (size >= 9791 and size <= 9799): # 9794
            new_guess = 'Stable Diffusion XL Instruct'
        elif (size >= 18414 and size <= 18420): # sd35-large aio
            new_guess = 'Stable Diffusion 3'
        elif (size >= 20000 and size <= 40000):
            new_guess = 'FLUX'
        if debug_load:
            shared.log.trace(f'Autodetect: method=size file="{fn}" size={size} previous="{current_guess}" current="{new_guess}"')
    return new_guess or current_guess


def guess_by_name(fn, current_guess):
    new_guess = None
    if 'instaflow' in fn.lower():
        new_guess = 'InstaFlow'
    elif 'segmoe' in fn.lower():
        new_guess = 'SegMoE'
    elif 'hunyuandit' in fn.lower():
        new_guess = 'HunyuanDiT'
    elif 'hdm-xut' in fn.lower():
        new_guess = 'hdm'
    elif 'pixart-xl' in fn.lower():
        new_guess = 'PixArt Alpha'
    elif 'stable-diffusion-3' in fn.lower():
        new_guess = 'Stable Diffusion 3'
    elif 'stable-cascade' in fn.lower() or 'stablecascade' in fn.lower() or 'wuerstchen3' in fn.lower() or ('sotediffusion' in fn.lower() and "v2" in fn.lower()):
        if devices.dtype == torch.float16:
            shared.log.warning('Stable Cascade does not support Float16')
        new_guess = 'Stable Cascade'
    elif 'pixart-sigma' in fn.lower():
        new_guess = 'PixArt Sigma'
    elif 'sana' in fn.lower():
        new_guess = 'Sana'
    elif 'lumina-next' in fn.lower():
        new_guess = 'Lumina-Next'
    elif 'lumina-image-2' in fn.lower():
        new_guess = 'Lumina 2'
    elif 'kolors' in fn.lower():
        new_guess = 'Kolors'
    elif 'auraflow' in fn.lower() or 'pony-v7' in fn.lower():
        new_guess = 'AuraFlow'
    elif 'cogview3' in fn.lower():
        new_guess = 'CogView 3'
    elif 'cogview4' in fn.lower():
        new_guess = 'CogView 4'
    elif 'meissonic' in fn.lower():
        new_guess = 'Meissonic'
    elif 'monetico' in fn.lower():
        new_guess = 'Monetico'
    elif 'omnigen2' in fn.lower():
        new_guess = 'OmniGen2'
    elif 'omnigen' in fn.lower():
        new_guess = 'OmniGen'
    elif 'sd3' in fn.lower():
        new_guess = 'Stable Diffusion 3'
    elif 'hidream' in fn.lower():
        new_guess = 'HiDream'
    elif 'chroma' in fn.lower() and 'xl' not in fn.lower():
        new_guess = 'Chroma'
    elif 'flux' in fn.lower() or 'flex.1' in fn.lower():
        size = round(os.path.getsize(fn) / 1024 / 1024) if os.path.isfile(fn) else 0
        if size > 11000 and size < 16000:
            shared.log.warning(f'Model detected as FLUX UNET model, but attempting to load a base model: file="{fn}" size={size} MB')
        new_guess = 'FLUX'
    elif 'flex.2' in fn.lower():
        new_guess = 'FLEX'
    elif 'cosmos-predict2' in fn.lower():
        new_guess = 'Cosmos'
    elif 'f-lite' in fn.lower():
        new_guess = 'FLite'
    elif 'wan' in fn.lower():
        new_guess = 'WanAI'
    if 'chronoedit' in fn.lower():
        new_guess = 'ChronoEdit'
    elif 'bria' in fn.lower():
        new_guess = 'Bria'
    elif 'qwen' in fn.lower():
        new_guess = 'Qwen'
    elif 'nextstep' in fn.lower():
        new_guess = 'NextStep'
    elif 'kandinsky-2-1' in fn.lower():
        new_guess = 'Kandinsky 2.1'
    elif 'kandinsky-2-2' in fn.lower():
        new_guess = 'Kandinsky 2.2'
    elif 'kandinsky-3' in fn.lower():
        new_guess = 'Kandinsky 3.0'
    elif 'hunyuanimage3' in fn.lower() or 'hunyuanimage-3' in fn.lower():
        new_guess = 'HunyuanImage3'
    elif 'hunyuanimage' in fn.lower():
        new_guess = 'HunyuanImage'
    elif 'x-omni' in fn.lower():
        new_guess = 'X-Omni'
    elif 'sdxl-turbo' in fn.lower() or 'stable-diffusion-xl' in fn.lower():
        new_guess = 'Stable Diffusion XL'
    if debug_load:
        shared.log.trace(f'Autodetect: method=name file="{fn}" previous="{current_guess}" current="{new_guess}"')
    return new_guess or current_guess


def guess_by_diffusers(fn, current_guess):
    exclude_by_name = ['ostris/Flex.2-preview'] # pipeline may be misleading
    if not os.path.isdir(fn):
        return current_guess, None
    index = os.path.join(fn, 'model_index.json')
    if os.path.exists(index) and os.path.isfile(index):
        index = shared.readfile(index, silent=True)
        name = index.get('_name_or_path', None)
        if name is not None and name in exclude_by_name:
            return current_guess, None
        cls = index.get('_class_name', None)
        if cls is not None:
            pipeline = getattr(diffusers, cls, None)
            if pipeline is None:
                pipeline = cls
        if callable(pipeline):
            is_quant = False
            for folder in os.listdir(fn):
                folder = os.path.join(fn, folder)
                if is_quant:
                    break
                if folder.endswith('quantization_config.json'):
                    is_quant = True
                    break
                if os.path.isdir(folder):
                    for f in os.listdir(folder):
                        if f.endswith('quantization_config.json'):
                            is_quant = True
                            break
            pipelines = shared_items.get_pipelines()
            for k, v in pipelines.items():
                if v is not None and v.__name__ == pipeline.__name__:
                    if is_quant:
                        k = f'{k} SDNQ'
                    if debug_load:
                        shared.log.trace(f'Autodetect: method=diffusers file="{fn}" previous="{current_guess}" current="{k}"')
                    return k, v
    return current_guess, None


def guess_variant(fn, current_guess):
    new_guess = None
    if 'inpaint' in fn.lower():
        if current_guess == 'Stable Diffusion':
            new_guess = 'Stable Diffusion Inpaint'
        elif current_guess == 'Stable Diffusion XL':
            new_guess = 'Stable Diffusion XL Inpaint'
    elif 'instruct' in fn.lower():
        if current_guess == 'Stable Diffusion':
            new_guess = 'Stable Diffusion Instruct'
        elif current_guess == 'Stable Diffusion XL':
            new_guess = 'Stable Diffusion XL Instruct'
    if debug_load:
        shared.log.trace(f'Autodetect: method=variant file="{fn}" previous="{current_guess}" current="{new_guess}"')
    return new_guess or current_guess


def detect_pipeline(f: str, op: str = 'model'):
    guess = shared.opts.diffusers_pipeline
    pipeline = None
    if guess == 'Autodetect':
        try:
            guess = 'Stable Diffusion XL' if ('XL' in f.upper() or 'SDNQ' in f.upper()) else 'Stable Diffusion' # set default guess
            guess = guess_by_size(f, guess)
            guess = guess_by_name(f, guess)
            guess, pipeline = guess_by_diffusers(f, guess)
            guess = guess_variant(f, guess)
            pipeline = shared_items.get_pipelines().get(guess, None) if pipeline is None else pipeline
            shared.log.info(f'Autodetect {op}: detect="{guess}" class={getattr(pipeline, "__name__", None)} file="{f}"')
            if debug_load is not None:
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
            pipeline = shared_items.get_pipelines().get(guess, None) if pipeline is None else pipeline
            shared.log.info(f'Load {op}: detect="{guess}" class={getattr(pipeline, "__name__", None)} file="{f}"')
        except Exception as e:
            shared.log.error(f'Load {op}: detect="{guess}" file="{f}" {e}')

    if pipeline is None:
        pipeline = diffusers.DiffusionPipeline
    return pipeline, guess


def get_load_config(model_file, model_type, config_type='yaml'):
    model_type = model_type.removesuffix(' SDNQ')
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
