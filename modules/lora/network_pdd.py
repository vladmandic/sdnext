"""Parallel decoding distillation heads carried by a native network.

A PDD file pairs a backbone LoRA with the output projections repeated once per interval of an N-step
training grid; each step fuses the heads of its block into one projection, so N / block_size evaluations
walk the trajectory. Heads ride on ``Network.extras['pdd']``: ``reconcile`` installs and removes them with
the loaded set, ``pin`` holds the step count and schedule the file was distilled for.
"""

import os
import copy
import weakref
import torch
from modules.logger import log


debug_log = log.trace if os.environ.get('SD_LORA_DEBUG', None) is not None else lambda *args, **kwargs: None


METADATA_STEPS = 'pdd_num_steps'
METADATA_BLOCK = 'pdd_block_size'
EXTRAS_KEY = 'pdd'


class ArchSpec:
    """How an architecture hosts parallel heads: the scheduler behind each head and how interval counts map onto its num_inference_steps."""

    def __init__(self, schedulers=None, default_scheduler='scheduler', steps_for=None, shift_keys=None):
        self.schedulers = schedulers or {} # head path -> attribute of the scheduler the head was trained on
        self.default_scheduler = default_scheduler
        self.steps_for = steps_for or (lambda intervals: intervals) # num_inference_steps that yields this many grid intervals
        self.shift_keys = shift_keys or {} # scheduler attribute -> infotext key the pinned shift is recorded under

    def scheduler_name(self, head):
        return self.schedulers.get(head, self.default_scheduler)


class ParallelHeads:
    """The head tensors of one file and the grid they were trained on."""

    def __init__(self, num_steps, block_size, heads):
        self.num_steps = num_steps
        self.block_size = block_size
        self.heads = heads # head path -> (weight [N, out, in], bias [N, out] or None)
        self.nfe = num_steps // block_size


class Installed:
    """Bookkeeping for the heads currently swapped into a pipeline."""

    def __init__(self, name, strength, component, modules, heads, spec):
        self.name = name
        self.strength = strength
        self.component = component
        self.modules = modules # head path -> (parent, attribute, original module)
        self.heads = heads
        self.spec = spec
        self.steps = spec.steps_for(heads.nfe)


def detect(metadata):
    """The (num_steps, block_size) grid a file declares; None without PDD metadata, ValueError for an unusable grid."""
    metadata = metadata or {}
    if METADATA_STEPS not in metadata:
        return None
    num_steps = int(metadata[METADATA_STEPS])
    block_size = int(metadata.get(METADATA_BLOCK, 1))
    if num_steps < 1 or block_size < 1 or num_steps % block_size != 0:
        raise ValueError(f'grid={num_steps} block={block_size}')
    return num_steps, block_size


def load(name, metadata, state_dict):
    """Collect the per-interval head tensors of a file, or None when the file carries no PDD grid."""
    try:
        grid = detect(metadata)
    except ValueError as e:
        log.error(f'Network load: type=PDD name="{name}" {e} block size must divide the grid')
        return None
    if grid is None:
        return None
    num_steps, block_size = grid
    heads = {}
    for key, tensor in state_dict.items():
        if key.endswith('.weight') and tensor.ndim == 3 and tensor.shape[0] == num_steps:
            path = key[:-len('.weight')]
            heads[path] = (tensor, state_dict.get(f'{path}.bias', None))
    if len(heads) == 0:
        log.error(f'Network load: type=PDD name="{name}" grid={num_steps} block={block_size} no head tensors')
        return None
    log.debug(f'Network load: type=PDD name="{name}" grid={num_steps} block={block_size} nfe={num_steps // block_size} heads={list(heads)}')
    return ParallelHeads(num_steps, block_size, heads)


def header_metadata(filename, name):
    """The metadata read from the file itself, for when the cached metadata lacks the grid; warns when head-shaped tensors have no grid."""
    from safetensors import safe_open
    try:
        with safe_open(filename, framework='pt', device='cpu') as f:
            metadata = f.metadata() or {}
            if METADATA_STEPS in metadata:
                log.debug(f'Network load: type=PDD name="{name}" grid read from the file header')
            else:
                heads = sum(1 for key in f.keys() if key.endswith('.weight') and len(f.get_slice(key).get_shape()) == 3)
                if heads > 0:
                    log.warning(f'Network load: type=PDD name="{name}" heads={heads} no {METADATA_STEPS} metadata: heads ignored')
            return metadata
    except Exception as e:
        log.warning(f'Network load: type=PDD name="{name}" header {e}')
        return {}


def try_load(name, network_on_disk, lora_scale): # pylint: disable=unused-argument
    """Family loader for the native chain: a network carrying only the heads."""
    metadata = getattr(network_on_disk, 'metadata', None) or {}
    if METADATA_STEPS not in metadata:
        metadata = header_metadata(network_on_disk.filename, name) # the metadata cache keeps a failed read forever and --no-metadata returns nothing
    if METADATA_STEPS not in metadata:
        return None
    from modules.lora import native_adapter
    state_dict = native_adapter.read_state_dict(network_on_disk.filename, what='network')
    heads = load(name, metadata, state_dict)
    if heads is None:
        return None
    net = native_adapter.new_network(name, network_on_disk)
    net.extras[EXTRAS_KEY] = heads
    return net


def base_tensors(module):
    """Float copies of a projection's weight and bias for the strength blend; None when the weight is not a plain tensor."""
    weight = getattr(module, 'weight', None)
    if weight is None or getattr(module, 'sdnq_dequantizer', None) is not None or not torch.is_floating_point(weight) or weight.ndim != 2:
        return None, None
    bias = getattr(module, 'bias', None)
    return weight.detach().to(dtype=torch.float32), None if bias is None else bias.detach().to(dtype=torch.float32)


class ParallelHead(torch.nn.Module):
    """An output projection replaced by its per-interval heads, fused per step for the block the scheduler is about to take."""

    def __init__(self, base, weight, bias, strength, get_scheduler, block_size, intervals):
        super().__init__()
        object.__setattr__(self, 'base', base) # kept out of the module tree so nothing walks, offloads or serializes it
        object.__setattr__(self, 'get_scheduler', get_scheduler)
        self.weight = torch.nn.Parameter(weight.to(dtype=torch.float32), requires_grad=False) # float32 like the projection it replaces
        self.bias = None if bias is None else torch.nn.Parameter(bias.to(dtype=torch.float32), requires_grad=False)
        self.register_buffer('intervals', intervals.to(dtype=torch.float32))
        self.in_features = weight.shape[2]
        self.out_features = weight.shape[1]
        self.num_steps = weight.shape[0]
        self.block_size = block_size
        self.strength = strength
        self.base_weight, self.base_bias = base_tensors(base) if strength != 1.0 else (None, None)
        self.fused_index = None
        self.fused_weight = None
        self.fused_bias = None
        self.overflow_warned = False

    def step_index(self):
        scheduler = self.get_scheduler()
        index = getattr(scheduler, 'step_index', None) if scheduler is not None else None
        index = 0 if index is None else int(index)
        nfe = self.num_steps // self.block_size
        if index >= nfe:
            if not self.overflow_warned:
                self.overflow_warned = True
                log.warning(f'Network: type=PDD step={index} nfe={nfe} schedule longer than the distilled grid')
            index = nfe - 1
        return index

    def fuse(self, index):
        start = index * self.block_size
        stop = start + self.block_size
        plan = torch.zeros(self.num_steps, dtype=torch.float32, device=self.weight.device)
        span = self.intervals[start:stop]
        plan[start:stop] = (span / span.sum()).to(device=plan.device)
        weight = torch.tensordot(plan, self.weight.detach(), dims=1)
        bias = None if self.bias is None else plan @ self.bias.detach()
        if self.base_weight is not None:
            base_weight = self.base_weight.to(device=weight.device)
            weight = base_weight + self.strength * (weight - base_weight)
            if bias is not None and self.base_bias is not None:
                base_bias = self.base_bias.to(device=bias.device)
                bias = base_bias + self.strength * (bias - base_bias)
        self.fused_index, self.fused_weight, self.fused_bias = index, weight, bias
        debug_log(f'Network: type=PDD fuse block={index} heads={start}:{stop} out={self.out_features} strength={self.strength}')

    def forward(self, hidden_states):
        index = self.step_index()
        if index != self.fused_index or self.fused_weight is None or self.fused_weight.device != self.weight.device:
            self.fuse(index)
        weight = self.fused_weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        bias = None if self.fused_bias is None else self.fused_bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return torch.nn.functional.linear(hidden_states, weight, bias)


def grid_intervals(scheduler, num_steps, spec):
    """Interval lengths of the training grid in ascending time, from a pristine scheduler copy immune to a live shift override."""
    probe = scheduler.__class__.from_config(scheduler.config) if hasattr(scheduler, 'from_config') else copy.deepcopy(scheduler)
    probe.set_timesteps(spec.steps_for(num_steps))
    sigmas = probe.sigmas.detach().to(device='cpu', dtype=torch.float64)
    if sigmas.numel() != num_steps + 1:
        return None
    return (1.0 - sigmas).diff()


def submodule(component, path):
    try:
        return component.get_submodule(path)
    except AttributeError:
        return None


def owner(pipe, heads, components):
    """The component holding every head projection, as (name, module); (None, None) when no component has them all."""
    for name in components:
        component = getattr(pipe, name, None)
        if component is not None and hasattr(component, 'get_submodule') and all(submodule(component, path) is not None for path in heads.heads):
            return name, component
    return None, None


def target_shape(module):
    dequantizer = getattr(module, 'sdnq_dequantizer', None)
    if dequantizer is not None and getattr(dequantizer, 'original_shape', None) is not None:
        return tuple(dequantizer.original_shape)
    weight = getattr(module, 'weight', None)
    return tuple(weight.shape) if weight is not None else None


def install(pipe, net, heads, spec, components):
    """Swap the heads into the component that owns their projections; True when the module tree changed."""
    component_name, component = owner(pipe, heads, components)
    if component is None:
        log.error(f'Network load: type=PDD name="{net.name}" heads={list(heads.heads)} no loaded component holds these projections')
        return False
    strength = float(net.te_multiplier) # transformer-keyed layers scale by the te multiplier, see network.NetworkModule.multiplier
    pipe_ref = weakref.ref(pipe)
    modules = {}
    for path, (weight, bias) in heads.heads.items():
        module = submodule(component, path)
        shape = target_shape(module)
        if shape != tuple(weight.shape[1:]):
            log.error(f'Network load: type=PDD name="{net.name}" head={path} shape={list(weight.shape[1:])} module={list(shape) if shape else None} shape mismatch')
            for parent, attr, original in modules.values():
                setattr(parent, attr, original)
            return False
        scheduler_name = spec.scheduler_name(path)
        scheduler = getattr(pipe, scheduler_name, None)
        intervals = grid_intervals(scheduler, heads.num_steps, spec) if scheduler is not None else None
        if intervals is None:
            log.error(f'Network load: type=PDD name="{net.name}" head={path} scheduler={scheduler.__class__.__name__} cannot build a {heads.num_steps}-interval grid')
            for parent, attr, original in modules.values():
                setattr(parent, attr, original)
            return False
        def get_scheduler(name=scheduler_name):
            owner_pipe = pipe_ref()
            return getattr(owner_pipe, name, None) if owner_pipe is not None else None
        for tensor in list(module.parameters()) + list(module.buffers()):
            tensor.data = tensor.data.clone() # nothing moves the stashed projection, and a shard view would keep the whole shard mapped
        head = ParallelHead(module, weight, bias, strength, get_scheduler, heads.block_size, intervals)
        parent_path, _, attr = path.rpartition('.')
        parent = component.get_submodule(parent_path) if parent_path else component
        setattr(parent, attr, head)
        modules[path] = (parent, attr, module)
    pipe.sdnext_pdd = Installed(net.name, strength, component_name, modules, heads, spec)
    log.info(f'Network load: type=PDD name="{net.name}" component={component_name} heads={list(modules)} grid={heads.num_steps} block={heads.block_size} nfe={heads.nfe} steps={pipe.sdnext_pdd.steps} strength={strength}')
    return True


def restore(pipe):
    """Put the original projections back; True when heads were installed."""
    state = getattr(pipe, 'sdnext_pdd', None)
    if state is None:
        return False
    for parent, attr, original in state.modules.values():
        setattr(parent, attr, original)
    del pipe.sdnext_pdd
    log.info(f'Network unload: type=PDD name="{state.name}" component={state.component} heads={list(state.modules)}')
    return True


def arch_spec():
    """The parallel-head spec of the loaded architecture's native loader module, or None."""
    import importlib
    from modules import shared
    from modules.lora import lora_load
    module_name = lora_load.NATIVE_DISPATCH.get(shared.sd_model_type)
    if module_name is None:
        return None
    return getattr(importlib.import_module(module_name), 'PDD', None)


def reconcile(pipe, loaded, components):
    """Match the installed heads to the loaded networks; True when the module tree changed."""
    carriers = [net for net in loaded if EXTRAS_KEY in getattr(net, 'extras', {})]
    if len(carriers) == 0:
        return restore(pipe)
    if len(carriers) > 1:
        log.warning(f'Network load: type=PDD networks={[net.name for net in carriers]} one grid per model, using first')
    net = carriers[0]
    current = getattr(pipe, 'sdnext_pdd', None)
    if current is not None and current.name == net.name and current.strength == float(net.te_multiplier):
        return False
    spec = arch_spec()
    if spec is None:
        from modules import shared
        log.error(f'Network load: type=PDD name="{net.name}" type={shared.sd_model_type} architecture has no parallel head support')
        return restore(pipe)
    changed = restore(pipe)
    return install(pipe, net, net.extras[EXTRAS_KEY], spec, components) or changed


def pin(p, model):
    """Hold a generation on the distilled evaluation count and shipped schedule while heads are installed; returns the scheduler step argument or None."""
    state = getattr(model, 'sdnext_pdd', None)
    if state is None:
        return None
    shifts = {}
    for path in state.modules:
        name = state.spec.scheduler_name(path)
        scheduler = getattr(model, name, None)
        if scheduler is not None and hasattr(scheduler, 'set_shift') and getattr(scheduler, 'config', None) is not None and 'shift' in scheduler.config:
            scheduler.set_shift(scheduler.config['shift'])
            shifts[name] = scheduler.config['shift']
    requested = p.steps
    p.steps = state.heads.nfe # what the user-facing step count means: transformer evaluations
    if getattr(p, 'task_args', None) is not None:
        p.task_args['num_inference_steps'] = state.steps # the scheduler argument that yields that many grid intervals
    if getattr(model, 'num_timesteps', None) is not None:
        model.num_timesteps = state.heads.nfe # the progress total counts transformer evaluations
    extra = getattr(p, 'extra_generation_params', None)
    if extra is not None:
        extra.update({state.spec.shift_keys[name]: shift for name, shift in shifts.items() if name in state.spec.shift_keys})
    log.info(f'Network: type=PDD name="{state.name}" steps={state.heads.nfe} requested={requested} grid_steps={state.steps} shift={shifts}')
    return state.steps
