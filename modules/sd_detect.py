import os
import time
import torch
import diffusers
from modules import shared, shared_items, devices, errors, model_tools


debug_load = os.environ.get('SD_LOAD_DEBUG', None)


def guess_by_size(fn, current_guess):
    if os.path.isfile(fn) and fn.endswith('.safetensors'):
        size = round(os.path.getsize(fn) / 1024 / 1024)
        if (size > 0 and size < 128):
            shared.log.warning(f'Model size smaller than expected: file="{fn}" size={size} MB')
        elif (size >= 316 and size <= 324) or (size >= 156 and size <= 164): # 320 or 160
            shared.log.warning(f'Model detected as VAE model, but attempting to load as model: file="{fn}" size={size} MB')
            return 'VAE'
        elif (size >= 2002 and size <= 2038): # 2032
            return 'Stable Diffusion 1.5'
        elif (size >= 3138 and size <= 3142): #3140
            return 'Stable Diffusion XL'
        elif (size >= 3361 and size <= 3369): # 3368
            return 'Stable Diffusion Upscale'
        elif (size >= 4891 and size <= 4899): # 4897
            return 'Stable Diffusion XL Inpaint'
        elif (size >= 4970 and size <= 4976): # 4973
            return 'Stable Diffusion 2' # SD v2 but could be eps or v-prediction
        elif (size >= 5791 and size <= 5799): # 5795
            return 'Stable Diffusion XL Refiner'
        elif (size > 5692 and size < 5698) or (size > 4134 and size < 4138) or (size > 10362 and size < 10366) or (size > 15028 and size < 15228):
            return 'Stable Diffusion 3'
        elif (size >= 6420 and size <= 7220): # 6420, IustriousRedux is 6541, monkrenRealisticINT_v10 is 7217
            return 'Stable Diffusion XL'
        elif (size >= 9791 and size <= 9799): # 9794
            return 'Stable Diffusion XL Instruct'
        elif (size >= 18414 and size <= 18420): # sd35-large aio
            return 'Stable Diffusion 3'
        elif (size >= 20000 and size <= 40000):
            return 'FLUX'
    return current_guess


def guess_by_name(fn, current_guess):
    if 'instaflow' in fn.lower():
        return 'InstaFlow'
    elif 'segmoe' in fn.lower():
        return 'SegMoE'
    elif 'hunyuandit' in fn.lower():
        return 'HunyuanDiT'
    elif 'pixart-xl' in fn.lower():
        return 'PixArt Alpha'
    elif 'stable-diffusion-3' in fn.lower():
        return 'Stable Diffusion 3'
    elif 'stable-cascade' in fn.lower() or 'stablecascade' in fn.lower() or 'wuerstchen3' in fn.lower() or ('sotediffusion' in fn.lower() and "v2" in fn.lower()):
        if devices.dtype == torch.float16:
            shared.log.warning('Stable Cascade does not support Float16')
        return 'Stable Cascade'
    elif 'pixart-sigma' in fn.lower():
        return 'PixArt Sigma'
    elif 'sana' in fn.lower():
        return 'Sana'
    elif 'lumina-next' in fn.lower():
        return 'Lumina-Next'
    elif 'lumina-image-2' in fn.lower():
        return 'Lumina 2'
    elif 'kolors' in fn.lower():
        return 'Kolors'
    elif 'auraflow' in fn.lower():
        return 'AuraFlow'
    elif 'cogview3' in fn.lower():
        return 'CogView 3'
    elif 'cogview4' in fn.lower():
        return 'CogView 4'
    elif 'meissonic' in fn.lower():
        return 'Meissonic'
    elif 'monetico' in fn.lower():
        return 'Monetico'
    elif 'omnigen2' in fn.lower():
        return 'OmniGen2'
    elif 'omnigen' in fn.lower():
        return 'OmniGen'
    elif 'sd3' in fn.lower():
        return 'Stable Diffusion 3'
    elif 'hidream' in fn.lower():
        return 'HiDream'
    elif 'chroma' in fn.lower() and 'xl' not in fn.lower():
        return 'Chroma'
    elif 'flux' in fn.lower() or 'flex.1' in fn.lower():
        size = round(os.path.getsize(fn) / 1024 / 1024) if os.path.isfile(fn) else 0
        if size > 11000 and size < 16000:
            shared.log.warning(f'Model detected as FLUX UNET model, but attempting to load a base model: file="{fn}" size={size} MB')
        return 'FLUX'
    elif 'flex.2' in fn.lower():
        return 'FLEX'
    elif 'cosmos-predict2' in fn.lower():
        return 'Cosmos'
    elif 'f-lite' in fn.lower():
        return 'FLite'
    elif 'wan' in fn.lower():
        return 'WanAI'
    elif 'bria' in fn.lower():
        return 'Bria'
    elif 'qwen' in fn.lower():
        return 'Qwen'
    elif 'nextstep' in fn.lower():
        return 'NextStep'
    elif 'kandinsky-2-1' in fn.lower():
        return 'Kandinsky 2.1'
    elif 'kandinsky-2-2' in fn.lower():
        return 'Kandinsky 2.2'
    elif 'kandinsky-3' in fn.lower():
        return 'Kandinsky 3.0'
    return current_guess


def guess_by_diffusers(fn, current_guess):
    exclude_by_name = ['ostris/Flex.2-preview'] # pipeline may be misleading
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
            pipelines = shared_items.get_pipelines()
            for k, v in pipelines.items():
                if v is not None and v.__name__ == pipeline.__name__:
                    return k, v
    return current_guess, None


def guess_variant(fn, current_guess):
    if 'inpaint' in fn.lower():
        if current_guess == 'Stable Diffusion':
            return 'Stable Diffusion Inpaint'
        elif current_guess == 'Stable Diffusion XL':
            return 'Stable Diffusion XL Inpaint'
    elif 'instruct' in fn.lower():
        if current_guess == 'Stable Diffusion':
            return 'Stable Diffusion Instruct'
        elif current_guess == 'Stable Diffusion XL':
            return 'Stable Diffusion XL Instruct'
    return current_guess


def detect_pipeline(f: str, op: str = 'model'):
    guess = shared.opts.diffusers_pipeline
    pipeline = None
    if guess == 'Autodetect':
        try:
            guess = 'Stable Diffusion XL' if 'XL' in f.upper() else 'Stable Diffusion' # set default guess
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
        shared.log.warning(f'Load {op}: detect="{guess}" file="{f}" not recognized')
        pipeline = diffusers.DiffusionPipeline
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
