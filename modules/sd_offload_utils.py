import re
import math
import inspect
import itertools
import torch
import accelerate
from modules import shared
from modules.logger import log
import modules.sd_offload_state as s


def dtype_byte_size(dtype: torch.dtype):
    try:
        if dtype in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]:
            dtype = accelerate.utils.modeling.CustomDtype.FP8
    except Exception: # catch since older torch many not have defined dtypes
        pass
    return s.accelerate_dtype_byte_size(dtype)


def get_signature(cls):
    signature = inspect.signature(cls.__init__, follow_wrapped=True)
    return signature.parameters


def get_module_names(pipe=None, exclude=None):
    def is_valid(module):
        if isinstance(getattr(pipe, module, None), torch.nn.ModuleDict):
            return True
        if isinstance(getattr(pipe, module, None), torch.nn.ModuleList):
            return True
        if isinstance(getattr(pipe, module, None), torch.nn.Module):
            return True
        return False

    if exclude is None:
        exclude = []
    if pipe is None:
        if shared.sd_loaded:
            pipe = shared.sd_model
        else:
            return []
    modules_names = []
    if hasattr(pipe, '_component_specs'): # modular pipelines name their components in specs; the config dict also carries scalars
        modules_names.extend(pipe.components)
    else:
        try:
            dict_keys = pipe._internal_dict.keys() # pylint: disable=protected-access
            modules_names.extend(dict_keys)
        except Exception:
            pass
        try:
            dict_keys = get_signature(pipe).keys()
            modules_names.extend(dict_keys)
        except Exception:
            pass
    modules_names = [m for m in modules_names if m not in exclude and not m.startswith('_')]
    modules_names = [m for m in modules_names if is_valid(m)]
    modules_names = sorted(set(modules_names))
    return modules_names


def get_module_memory(module: torch.nn.Module) -> dict[str, float]:
    tensors = list(itertools.chain(module.parameters(), module.buffers()))
    logical_gib = sum(tensor.numel() * tensor.element_size() for tensor in tensors) / 1024**3
    storages = {}
    for tensor in tensors:
        try:
            storage = tensor.untyped_storage()
        except (AttributeError, RuntimeError):
            continue
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    storage_gib = sum(storages.values()) / 1024**3
    return {
        "logical": round(logical_gib, 3),
        "storage": round(storage_gib, 3),
        "overhead": round(storage_gib - logical_gib, 3),
        "tensors": len(tensors),
        "storages": len(storages),
    }


def get_module_size(module: torch.nn.Module) -> tuple[float, float]:
    module_size = 0
    param_num = 0
    if not isinstance(module, torch.nn.Module):
        return 0, 0
    try:
        # module_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
        tensors = set(itertools.chain(module.parameters(recurse=True), module.buffers(recurse=True)))
        module_size = sum(t.numel() * t.element_size() for t in tensors) / 1024**3
        param_num = sum(p.numel() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
    except Exception as e:
        log.error(f'Offload: type=balanced op=calc module={module.__class__.__name__} {e}')
        module_size = 0
        param_num = 0
    return module_size, param_num


def offload_list(opt: str) -> list:
    return [m.strip() for m in re.split(';|,| ', opt) if len(m.strip()) > 2]


def offload_matches(module, module_name: str | None, names: list) -> bool:
    """Match against an always/never list by class name or by pipeline component name.
    Component entries such as `text_encoder` cover every architecture without listing each encoder class."""
    if module.__class__.__name__ in names:
        return True
    module_name = module_name or getattr(module, 'module_name', None)
    return module_name is not None and module_name in names


def offload_model_types() -> list:
    return [m.lower().strip() for m in re.split(r'[ ,]+', shared.opts.models_not_to_offload) if m.strip()] # type codes like sd and f1 are two characters, so only empty fragments are dropped


def offload_excluded(module_name: str, module) -> bool:
    """Whether the offload exclusion settings keep this component on the accelerator."""
    if shared.sd_model_type.lower() in offload_model_types():
        return True
    return offload_matches(module, module_name, offload_list(shared.opts.diffusers_offload_never))


def get_pipe_variants(pipe=None):
    if pipe is None:
        if shared.sd_loaded:
            pipe = shared.sd_model
        else:
            return [pipe]
    variants = [pipe]
    if hasattr(pipe, "pipe"):
        variants.append(pipe.pipe)
    if hasattr(pipe, "prior_pipe"):
        variants.append(pipe.prior_pipe)
    if hasattr(pipe, "decoder_pipe"):
        variants.append(pipe.decoder_pipe)
    return variants


def set_accelerate(sd_model):
    def set_accelerate_to_module(model):
        if hasattr(model, "pipe"):
            set_accelerate_to_module(model.pipe)
        for module_name in get_module_names(model):
            component = getattr(model, module_name, None)
            if isinstance(component, torch.nn.Module):
                component.has_accelerate = True

    sd_model.has_accelerate = True
    set_accelerate_to_module(sd_model)
    if hasattr(sd_model, "prior_pipe"):
        set_accelerate_to_module(sd_model.prior_pipe)
    if hasattr(sd_model, "decoder_pipe"):
        set_accelerate_to_module(sd_model.decoder_pipe)


def get_logical_param_count(module: torch.nn.Module) -> int:
    if hasattr(module, "sdnq_dequantizer"):
        original_shape = module.sdnq_dequantizer.original_shape
        count = math.prod(original_shape)
        if getattr(module, "bias", None) is not None:
            count += module.bias.numel()
        return int(count)
    count = sum(p.numel() for p in module.parameters(recurse=False))
    for child in module.children():
        count += get_logical_param_count(child)
    return count


def report_model_stats(module_name, module):
    try:
        size = s.offload_hook_instance.offload_map.get(module_name, 0) if s.offload_hook_instance is not None else 0
        if size == 0:
            size, _params = get_module_size(module)
        quant = getattr(module, "quantization_method", None)
        params = sum(p.numel() for p in module.parameters(recurse=True))
        logical = get_logical_param_count(module)
        log.debug(f'Module: name={module_name} cls={module.__class__.__name__} size={size:.3f} params={params} logical={logical} quant={quant}')
    except Exception as e:
        log.error(f'Module stats: name={module_name} {e}')
