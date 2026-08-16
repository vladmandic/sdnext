import time
import itertools
import torch
import accelerate.hooks
import accelerate.utils.modeling
from modules.logger import log
from modules import shared, devices, sd_models
from modules.timer import process as process_timer
from modules.sd_offload_utils import get_pipe_variants, get_module_names, get_module_size, set_accelerate, offload_excluded, report_model_stats
import modules.sd_offload_state as s


def group_offload_config(main: bool) -> dict:
    """Effective group offload settings for one component. Components that run once per
    generation take the leaf no-stream policy regardless of the main settings, so their
    weights are never held in pinned host memory."""
    stream = shared.opts.group_offload_stream if main else False
    blocks = max(1, int(shared.opts.group_offload_blocks))
    if stream and blocks != 1:
        blocks = 1 # streamed prefetch supports one block per group; upstream clamps with a warning otherwise
    return {
        'offload_type': shared.opts.group_offload_type if main else 'leaf_level',
        'num_blocks_per_group': blocks,
        'non_blocking': shared.opts.diffusers_offload_nonblocking,
        'use_stream': stream,
        'record_stream': shared.opts.group_offload_record and stream, # record without streams is rejected upstream
        'low_cpu_mem_usage': stream and not shared.opts.group_offload_pin,
    }


def remove_group_offload_component(module) -> bool:
    if getattr(module, 'sdnext_group_offload_sig', None) is None:
        module = getattr(module, 'model', None) # wrapper components carry the hooks on the inner model
        if module is None or getattr(module, 'sdnext_group_offload_sig', None) is None:
            return False
    from diffusers.hooks.group_offloading import _GROUP_OFFLOADING, _LAYER_EXECUTION_TRACKER, _LAZY_PREFETCH_GROUP_OFFLOADING
    from diffusers.hooks.hooks import HookRegistry
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.remove_hook(_GROUP_OFFLOADING, recurse=True)
    registry.remove_hook(_LAYER_EXECUTION_TRACKER, recurse=True)
    registry.remove_hook(_LAZY_PREFETCH_GROUP_OFFLOADING, recurse=True)
    module.sdnext_group_offload_sig = None
    return True


def remove_group_offload(sd_model):
    removed = []
    for module_name in get_module_names(sd_model):
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module) and remove_group_offload_component(module):
            removed.append(module_name)
    for module_name in getattr(sd_model, 'sdnext_ondemand_modules', None) or []:
        module = getattr(sd_model, module_name, None)
        if module is not None:
            module.sdnext_ondemand = False
            if hasattr(module, '_hf_hook'):
                module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            removed.append(f'{module_name}:ondemand')
    if getattr(sd_model, 'sdnext_ondemand_modules', None):
        sd_model.sdnext_ondemand_modules = []
    if removed:
        log.debug(f'Offload: type=group op=remove modules={removed}')


def apply_group_offload_component(module, module_name: str, main: bool) -> bool:
    """Apply group offload to one component. Re-application with unchanged settings is a no-op:
    the hooks silently keep their original config when re-applied and raise before the first
    forward, so a changed config must remove the old hooks first."""
    from diffusers.hooks import apply_group_offloading
    cfg = group_offload_config(main)
    if cfg['use_stream'] and not cfg['low_cpu_mem_usage']:
        size_gb, _params = get_module_size(module)
        pin_ok = getattr(module, 'sdnext_group_offload_pin', None)
        if pin_ok is None: # decide once per module: a granted pin moves the weights into locked memory, so re-reading available on the next apply would see it lower by the pinned size and revoke its own grant
            from modules import memstats
            avail_gb = memstats.ram_stats().get('avail', 0)
            reserve_gb = max(8.0, 0.25 * shared.cpu_memory) # pinned pages cannot be reclaimed or swapped, so a quarter of the machine, floored at 8 GB, stays pageable for the process and page cache
            limit_gb = (avail_gb - reserve_gb) if avail_gb > 0 else (0.5 * shared.cpu_memory) # budget from memory free right now; total-derived ceiling only when psutil cannot say
            pin_ok = size_gb <= limit_gb
            module.sdnext_group_offload_pin = pin_ok
            module.sdnext_group_offload_pin_limit = limit_gb
        if not pin_ok:
            # unpinned streaming degrades to per-transfer staging and leaf groups make that a per-module cost,
            # so the whole leaf+stream shape goes with the pin: few large synchronous groups instead
            cfg['low_cpu_mem_usage'] = True
            cfg['use_stream'] = False
            cfg['record_stream'] = False
            cfg['offload_type'] = 'block_level'
            cfg['num_blocks_per_group'] = max(4, int(shared.opts.group_offload_blocks))
            log.warning(f'Offload: type=group module={module_name} size={size_gb:.3f} limit={getattr(module, "sdnext_group_offload_pin_limit", 0):.3f} pin=denied type=block_level blocks={cfg["num_blocks_per_group"]} expect ~{size_gb:.0f} GB transferred per step')
    sig = f'{devices.device}:{main}:' + ':'.join(str(v) for v in cfg.values())
    if getattr(module, 'sdnext_group_offload_sig', None) == sig:
        return False
    if hasattr(module, '_hf_hook'): # leftover accelerate hooks from a previous offload mode abort the group apply upstream
        module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
    module.sdnext_ondemand = False # group placement replaces any on-demand hook
    remove_group_offload_component(module)
    module.requires_grad_(False)
    s.debug_move(f'Offload: type=group op=apply type={shared.opts.group_offload_type} module={module_name} pin={cfg["use_stream"] and not cfg["low_cpu_mem_usage"]}') # before the apply: pinning large components takes a while and would otherwise run silently
    module.sdnext_group_offload_sig = 'partial' # a raise below leaves hooks that only a non-empty signature will remove
    apply_group_offloading(module, onload_device=devices.device, offload_device=devices.cpu, **cfg)
    module.sdnext_group_offload_sig = sig
    return True


def set_group_resident(module) -> bool:
    """Keep a component on the accelerator with no hooks of any kind."""
    changed = False
    if hasattr(module, '_hf_hook'):
        module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
        changed = True
    if remove_group_offload_component(module):
        changed = True
    module.sdnext_ondemand = False
    module.requires_grad_(False)
    if any(not devices.same_device(t.device, devices.device) for t in itertools.chain(module.parameters(), module.buffers())): # an interrupted generation can leave a group-hooked module split across devices
        module.to(devices.device)
        changed = True
    return changed


def group_offload_role(module_name: str, module) -> str:
    """Placement role for one component: resident to stay put, ondemand for whole-module onload, main for per-step denoisers, aux for the rest."""
    if offload_excluded(module_name, module):
        return 'resident'
    if has_entry_bridge(module):
        return 'ondemand' # encode and decode bypass the forward that group hooks scope to
    if callable(getattr(module, 'encode', None)) or callable(getattr(module, 'decode', None)):
        s.debug_move(f'Offload: type=group module={module_name} cls={module.__class__.__name__} bridge=missing role=resident') # decorate the entry points with apply_forward_hook to make the component offloadable
        return 'resident' # nothing fires an onload for an undecorated entry point, so any hook placement strands the weights on cpu
    if not getattr(module, '_supports_group_offloading', True):
        return 'ondemand' # upstream marks modules that read submodule weights outside those submodules' forward
    if module_name in s.group_offload_main:
        return 'main'
    return 'aux'


def has_entry_bridge(module) -> bool:
    """Entry points decorated with diffusers' apply_forward_hook fire _hf_hook.pre_forward,
    which is what carries the on-demand onload for encode and decode calls that bypass forward."""
    for name in ('decode', 'encode'):
        fn = getattr(module, name, None)
        if fn is not None and getattr(fn, '__qualname__', '').startswith('apply_forward_hook'):
            return True
    return False


class OnDemandHook(accelerate.hooks.ModelHook):
    """Whole-module onload for components entered through decode or encode rather than forward.
    Tiled calls re-enter inside one entry point, so the module is on device before the first
    tile; the return to cpu happens at the processing seams once outputs are materialized."""
    def pre_forward(self, module, *args, **kwargs):
        param = next(module.parameters(), None)
        if param is not None and not devices.same_device(param.device, devices.device):
            t0 = time.time()
            module.to(devices.device, non_blocking=shared.opts.diffusers_offload_nonblocking)
            t1 = time.time()
            process_timer.add('onload', t1 - t0)
            s.debug_move(f'Offload: type=ondemand op=onload module={module.__class__.__name__} nonblocking={shared.opts.diffusers_offload_nonblocking} time={t1 - t0:.3f}') # working so no need to log
        return args, kwargs


def apply_group_offload_ondemand(module) -> bool:
    """Placement for components that never take group hooks: they onload whole at their entry point."""
    if getattr(module, 'sdnext_ondemand', False) and hasattr(module, '_hf_hook'):
        return False
    if hasattr(module, '_hf_hook'):
        module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
    remove_group_offload_component(module)
    module.requires_grad_(False)
    accelerate.hooks.add_hook_to_module(module, OnDemandHook(), append=False)
    module.sdnext_ondemand = True
    module.to(devices.cpu)
    return True


def offload_ondemand(sd_model, include=[], exclude=[], reason='', force=False):
    """Return on-demand components to cpu once their outputs are materialized."""
    if sd_model is None:
        return
    moved = []
    for pipe in get_pipe_variants(sd_model):
        names = get_module_names(pipe) if force else (getattr(pipe, 'sdnext_ondemand_modules', None) or []) # force enumerates the pipe rather than the list a load pass left on it
        for module_name in names:
            if include and module_name not in include:
                continue
            if exclude and module_name in exclude:
                continue
            module = getattr(pipe, module_name, None)
            if not isinstance(module, torch.nn.Module) or not getattr(module, 'sdnext_ondemand', False):
                continue # nothing else has an onload to bring it back
            param = next(module.parameters(), None)
            if param is None or devices.same_device(param.device, devices.cpu):
                continue
            try:
                t0 = time.time()
                module.to(devices.cpu, non_blocking=shared.opts.diffusers_offload_nonblocking)
                dt = time.time() - t0
                process_timer.add('offload', dt)
                moved.append(module_name)
                s.debug_move(f'Offload: type=ondemand op=offload module={module_name} nonblocking={shared.opts.diffusers_offload_nonblocking} reason="{reason}" time={dt:.3f}')
            except Exception as e:
                log.warning(f'Offload: type=ondemand op=offload module={module_name} {e}')
    if moved:
        devices.torch_gc(reason='ondemand')


def report_group_stats(sd_model, module_names):
    """Per-component stats block once per loaded model; balanced mode prints its own from the hook map."""
    checkpoint_name = sd_model.sd_checkpoint_info.name if getattr(sd_model, "sd_checkpoint_info", None) is not None else sd_model.__class__.__name__
    if checkpoint_name in s.group_stats_reported: # keyed by checkpoint since a task switch rebuilds the pipe object
        return
    s.group_stats_reported.add(checkpoint_name)
    total = 0.0
    counted = []
    for module_name in module_names:
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module):
            total += get_module_size(module)[0]
            counted.append(module_name)
            report_model_stats(module_name, module)
    log.info(f'Model class={sd_model.__class__.__name__} modules={len(counted)} size={total:.3f}')


def apply_group_offload(sd_model):
    """Per-component group offload for classic and modular pipelines."""
    changed = False
    placements = []
    module_names = get_module_names(sd_model)
    for module_name in module_names:
        module = getattr(sd_model, module_name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            role = group_offload_role(module_name, module)
            placements.append(f'{module_name}:{role}')
            if role == 'resident':
                applied = set_group_resident(module)
            elif role == 'ondemand':
                applied = apply_group_offload_ondemand(module)
            else:
                applied = apply_group_offload_component(module, module_name, main=role == 'main')
            changed = changed or applied
        except Exception as e:
            log.error(f'Offload: type=group module={module_name} {e}')
    sd_model.sdnext_ondemand_modules = [name for name in module_names if getattr(getattr(sd_model, name, None), 'sdnext_ondemand', False)]
    if sd_models.get_diffusers_task(sd_model) != sd_models.DiffusersTaskType.MODULAR: # group hooks are not accelerate hooks, so modular pipelines stay unstamped
        set_accelerate(sd_model)
    if changed:
        log.info(f'Offload: type=group modules={placements}')
    else:
        log.debug(f'Offload: type=group modules={placements}')
    report_group_stats(sd_model, module_names)
    return sd_model
