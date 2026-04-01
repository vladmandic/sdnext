import inspect
import diffusers
import transformers
import safetensors.torch
from modules import shared, devices, model_quant
from modules.logger import log


def remove_entries_after_depth(d, depth, current_depth=0):
    if current_depth >= depth:
        return None
    if isinstance(d, dict):
        return {k: remove_entries_after_depth(v, depth, current_depth + 1) for k, v in d.items() if remove_entries_after_depth(v, depth, current_depth + 1) is not None}
    return d


def list_compact(flat_list):
    result_list = []
    for item in flat_list:
        keys = item.split('.')
        keys = '.'.join(keys[:2])
        if keys not in result_list:
            result_list.append(keys)
    return result_list


def list_to_dict(flat_list):
    result_dict = {}
    try:
        for item in flat_list:
            keys = item.split('.')
            d = result_dict
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = None
    except Exception:
        pass
    return result_dict


def get_safetensor_keys(filename):
    keys = []
    try:
        with safetensors.torch.safe_open(filename, framework="pt", device="cpu") as f:
            keys = f.keys()
    except Exception:
        pass
    return keys


def get_modules(model: callable):
    signature = inspect.signature(model.__init__, follow_wrapped=True)
    params = {param.name: param.annotation for param in signature.parameters.values() if param.annotation != inspect._empty and hasattr(param.annotation, 'from_pretrained')} # pylint: disable=protected-access
    for name, cls in params.items():
        log.debug(f'Analyze: model={model} module={name} class={cls.__name__} loadable={getattr(cls, "from_pretrained", None)}')
    return params


def load_modules(repo_id: str, params: dict):
    cache_dir = shared.opts.hfcache_dir
    modules = {}
    for name, cls in params.items():
        subfolder = None
        kwargs = {}
        if cls == diffusers.AutoencoderKL:
            subfolder = 'vae'
        if cls == transformers.CLIPTextModel: # clip-vit-l
            subfolder = 'text_encoder'
        if cls == transformers.CLIPTextModelWithProjection: # clip-vit-g
            subfolder = 'text_encoder_2'
        if cls == transformers.T5EncoderModel: # t5-xxl
            subfolder = 'text_encoder_3'
            kwargs = model_quant.create_config(kwargs)
            kwargs['variant'] = 'fp16'
        if cls == diffusers.SD3Transformer2DModel:
            subfolder = 'transformer'
            kwargs = model_quant.create_config(kwargs)
        if subfolder is None:
            continue
        log.debug(f'Load: module={name} class={cls.__name__} repo={repo_id} location={subfolder}')
        modules[name] = cls.from_pretrained(repo_id, subfolder=subfolder, cache_dir=cache_dir, torch_dtype=devices.dtype, **kwargs)
    return modules
