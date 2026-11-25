import weakref
import torch
from modules import shared
from modules.logger import log


def module_entries(module):
    entries = [('parameter', name, tensor) for name, tensor in module.named_parameters()]
    entries.extend(('buffer', name, tensor) for name, tensor in module.named_buffers())
    return entries


def module_state(module):
    state = {}
    for kind, name, tensor in module_entries(module):
        if tensor.is_meta:
            return None, f'{kind}={name} is meta'
        try:
            version = tensor._version # pylint: disable=protected-access
        except RuntimeError as e:
            if not tensor.is_inference():
                return None, f'{kind}={name}: {e}'
            version = None
        state[(kind, name)] = (str(tensor.device), tensor.data_ptr(), version)
    if not state:
        return None, 'module has no parameters or buffers'
    return state, None


def set_buffer(module, name, tensor):
    parent_name, _, buffer_name = name.rpartition('.')
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, buffer_name, tensor)


def pin_cpu_tensor(tensor):
    return tensor.detach().pin_memory()


def clear_cpu_master(module):
    if hasattr(module, 'sdnext_balanced_cpu_master'):
        delattr(module, 'sdnext_balanced_cpu_master')


def invalidate_cpu_master(module):
    owner = getattr(module, 'sdnext_balanced_pin_owner', None)
    if owner is not None:
        module = owner() or module
    clear_cpu_master(module)


def disable_cpu_master(module, reason):
    if getattr(module, 'sdnext_balanced_pin_disabled', False):
        return
    module.sdnext_balanced_pin_disabled = True
    clear_cpu_master(module)
    log.warning(f'Offload: type=balanced module={getattr(module, "module_name", module.__class__.__name__)} pin=disabled reason="{reason}"')


def pin_cpu_master(module):
    if not shared.opts.diffusers_offload_pin or getattr(module, 'sdnext_balanced_pin_disabled', False):
        return False
    if shared.opts.layerwise_quantization or getattr(module, 'quantization_method', None) == 'LayerWise':
        disable_cpu_master(module, 'layerwise quantization')
        return False

    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    entries = [('parameter', name, tensor) for name, tensor in parameters.items()]
    entries.extend(('buffer', name, tensor) for name, tensor in buffers.items())
    if not entries:
        return False
    if any(tensor.device.type != 'cpu' for _, _, tensor in entries):
        disable_cpu_master(module, 'module spans multiple devices')
        return False
    if any(tensor.is_meta or tensor.layout != torch.strided for _, _, tensor in entries):
        disable_cpu_master(module, 'unsupported tensor layout')
        return False

    existing = getattr(module, 'sdnext_balanced_cpu_master', None)
    if existing is not None:
        same = parameters.keys() == existing['parameters'].keys() and buffers.keys() == existing['buffers'].keys()
        if same:
            same = all(parameters[name].data_ptr() == master.data_ptr() for name, master in existing['parameters'].items())
            same = same and all(buffers[name].data_ptr() == master.data_ptr() for name, master in existing['buffers'].items())
        if same:
            existing['loaded_state'] = None
            return True

    storage_ptrs = set()
    pinned_parameters = {}
    pinned_buffers = {}
    try:
        for kind, name, tensor in entries:
            storage_ptr = tensor.untyped_storage().data_ptr()
            if storage_ptr and storage_ptr in storage_ptrs:
                disable_cpu_master(module, 'aliased tensor storage')
                return False
            storage_ptrs.add(storage_ptr)
            pinned = tensor.detach() if tensor.is_pinned() else pin_cpu_tensor(tensor)
            if pinned.stride() != tensor.stride():
                disable_cpu_master(module, 'pinning changed tensor strides')
                return False
            if kind == 'parameter':
                pinned_parameters[name] = pinned
            else:
                pinned_buffers[name] = pinned
    except Exception as e:
        disable_cpu_master(module, f'pinned allocation failed: {e}')
        return False

    original_parameters = {name: parameter.data for name, parameter in parameters.items()}
    original_buffers = buffers.copy()
    try:
        for name, parameter in parameters.items():
            parameter.data = pinned_parameters[name]
        for name, tensor in pinned_buffers.items():
            set_buffer(module, name, tensor)
    except Exception as e:
        for name, parameter in parameters.items():
            parameter.data = original_parameters[name]
        for name, tensor in original_buffers.items():
            set_buffer(module, name, tensor)
        disable_cpu_master(module, f'tensor replacement failed: {e}')
        return False

    module.sdnext_balanced_cpu_master = {
        'parameters': pinned_parameters,
        'buffers': pinned_buffers,
        'loaded_state': None,
    }
    owner = weakref.ref(module)
    for submodule in module.modules():
        submodule.sdnext_balanced_pin_owner = owner
    return True


def restore_cpu_master(module):
    master = getattr(module, 'sdnext_balanced_cpu_master', None)
    if master is None:
        return False
    if not getattr(module, 'sdnext_balanced_pin_eligible', False):
        clear_cpu_master(module)
        return False
    state, _ = module_state(module)
    if state is None or state != master['loaded_state']:
        clear_cpu_master(module)
        return False

    parameters = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    if parameters.keys() != master['parameters'].keys() or buffers.keys() != master['buffers'].keys():
        clear_cpu_master(module)
        return False
    for kind, name, tensor in module_entries(module):
        if tensor.is_inference():
            saved = master['parameters'][name] if kind == 'parameter' else master['buffers'][name]
            if not torch.equal(tensor.detach().to(device=saved.device), saved):
                clear_cpu_master(module)
                return False
    for name, parameter in parameters.items():
        parameter.data = master['parameters'][name]
    for name, tensor in master['buffers'].items():
        set_buffer(module, name, tensor)
    master['loaded_state'] = None
    return True


def record_onload(module, eligible):
    module.sdnext_balanced_pin_eligible = eligible
    if not shared.opts.diffusers_offload_pin or not eligible:
        if shared.opts.diffusers_offload_pin and not eligible:
            disable_cpu_master(module, 'accelerate-managed device map')
        else:
            clear_cpu_master(module)
        return
    master = getattr(module, 'sdnext_balanced_cpu_master', None)
    if master is not None:
        master['loaded_state'], reason = module_state(module)
        if master['loaded_state'] is None:
            disable_cpu_master(module, f'unable to track tensor state: {reason}')


def module_is_cpu(module):
    tensors = list(module.parameters()) + list(module.buffers())
    return bool(tensors) and all(tensor.device.type == 'cpu' for tensor in tensors)
