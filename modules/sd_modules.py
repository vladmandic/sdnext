from dataclasses import dataclass
import inspect
import torch


@dataclass
class ModuleStats:
    module: str
    cls: str
    params: float
    size: float
    quant: str
    dtype: str

    def __init__(self, module: str, cls: str, params: float, size: float, quant: str, dtype: str):
        self.module = module
        self.cls = cls
        self.params = params
        self.size = size
        self.quant = quant
        self.dtype = dtype

    def __str__(self):
        return f'module="{self.module}" cls={self.cls} params={self.params:.3f} size={self.size:.3f} quant={self.quant} dtype={self.dtype}'


def get_signature(cls):
    signature = inspect.signature(cls.__init__, follow_wrapped=True)
    return signature.parameters


def get_module_stats(name, module):
    if not isinstance(module, torch.nn.Module):
        return
    try:
        module_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
        param_num = sum(p.numel() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
    except Exception:
        module_size = 0
        param_num = 0
    cls = module.__class__.__name__
    quant = getattr(module, "quantization_method", None)
    module_stats = ModuleStats(name, cls, param_num, module_size, quant, module.dtype)
    return module_stats


def get_model_stats(model, exclude=None):
    # from transformers import Gemma3ForCausalLM
    modules = []

    if isinstance(model, torch.nn.Module):
        module_stats = get_module_stats(model.__class__.__name__, model)
        if module_stats is not None:
            modules.append(module_stats)
        return modules

    if hasattr(model, "_internal_dict"):
        modules_names = model._internal_dict.keys() # pylint: disable=protected-access
    else:
        modules_names = get_signature(model).keys()

    if modules_names is None or not isinstance(modules_names, list) or len(modules_names) == 0:
        return modules

    modules_names = [m for m in modules_names if m is not None and m not in exclude and not m.startswith('_')]
    for module_name in modules_names:
        module = getattr(model, module_name, None)
        if module is not None:
            module_stats = get_module_stats(module_name, module)
            if module_stats is not None:
                modules.append(module_stats)

    return modules
