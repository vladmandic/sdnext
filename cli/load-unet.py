# test for manually loading unet state_dict

import torch
import diffusers


class StateDictStats():
    cls: str = None
    device: torch.device = None
    params: int = 0
    weights: dict = {}
    dtypes: dict = {}
    config: dict = None

    def __repr__(self):
        return f'cls={self.cls} params={self.params} weights={self.weights} device={self.device} dtypes={self.dtypes} config={self.config is not None}'


def set_module_tensor(
    module: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    stats: StateDictStats,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    if "." in name:
        splits = name.split(".")
        for split in splits[:-1]:
            module = getattr(module, split)
        name = splits[-1]
    old_value = getattr(module, name)
    with torch.no_grad():
        if value.dtype not in stats.dtypes:
            stats.dtypes[value.dtype] = 0
        stats.dtypes[value.dtype] += 1
        if name in module._buffers: # pylint: disable=protected-access
            module._buffers[name] = value.to(device=device, dtype=dtype) # pylint: disable=protected-access
            if 'buffers' not in stats.weights:
                stats.weights['buffers'] = 0
            stats.weights['buffers'] += 1
        elif value is not None:
            param_cls = type(module._parameters[name]) # pylint: disable=protected-access
            module._parameters[name] = param_cls(value, requires_grad=old_value.requires_grad).to(device, dtype=dtype) # pylint: disable=protected-access
            if 'parameters' not in stats.weights:
                stats.weights['parameters'] = 0
            stats.weights['parameters'] += 1


def load_unet(config_file: str, state_dict: dict, device: torch.device = None, dtype: torch.dtype = None):
    # same can be done for other modules or even for entire model by loading model config and then walking through its modules
    from accelerate import init_empty_weights
    with init_empty_weights():
        stats = StateDictStats()
        stats.device = device
        stats.config = diffusers.UNet2DConditionModel.load_config(config_file)
        unet = diffusers.UNet2DConditionModel.from_config(stats.config)
        stats.cls = unet.__class__.__name__
        expected_state_dict_keys = list(unet.state_dict().keys())
        stats.weights['expected'] = len(expected_state_dict_keys)
    for param_name, param in state_dict.items():
        if param_name not in expected_state_dict_keys:
            if 'unknown' not in stats.weights:
                stats.weights['unknown'] = 0
            stats.weights['unknown'] += 1
            continue
        set_module_tensor(unet, name=param_name, value=param, device=device, dtype=dtype, stats=stats)
        state_dict[param_name] = None # unload as we initialize the model so we dont consume double the memory
    stats.params = sum(p.numel() for p in unet.parameters(recurse=True))
    return unet, stats


def load_safetensors(fn: str):
    import safetensors.torch
    state_dict = safetensors.torch.load_file(fn, device='cpu') # state dict should always be loaded to cpu
    return state_dict


if __name__ == "__main__":
    # need pipe already present to load unet state_dict into or we could load unet first and then manually create pipe with params
    pipe = diffusers.StableDiffusionXLPipeline.from_single_file('/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors', cache_dir='/mnt/models/huggingface')
    # this could be kept in memory so we dont have to reload it
    dct = load_safetensors('/mnt/models/UNET/dpo-sdxl-text2image.safetensors')
    pipe.unet, s = load_unet(
        config_file = 'configs/sdxl/unet/config.json', # can also point to online hf model with subfolder
        state_dict = dct,
        device = torch.device('cpu'), # can leave out to use default device
        dtype = torch.bfloat16, # can leave out to use default dtype, especially for mixed precision modules
    )
    from rich import print as rprint
    rprint(f'Stats: {s}')
