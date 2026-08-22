import time
import torch
import accelerate.hooks
import accelerate.utils.modeling
from modules.logger import log
from modules import shared, devices, sd_models
from modules.timer import process as process_timer
import modules.sd_offload_state as s
from modules.sd_offload_utils import get_module_names, get_module_size, get_module_memory, offload_list, offload_matches, offload_model_types, dtype_byte_size, get_pipe_variants, set_accelerate # pylint: disable=unused-import
from modules.sd_offload_balanced import apply_balanced_offload
from modules.sd_offload_group import apply_group_offload, remove_group_offload, offload_ondemand # pylint: disable=unused-import


def disable_offload(sd_model):
    remove_group_offload(sd_model) # group hooks block the meta move at unload, keeping component weights alive for as long as any reference to the pipe survives
    if not getattr(sd_model, 'has_accelerate', False):
        return
    for module_name in get_module_names(sd_model):
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module):
            network_layer_name = getattr(module, "network_layer_name", None)
            try:
                module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            except Exception as e:
                log.warning(f'Offload: remove hook module={module_name} {e}')
            if network_layer_name:
                module.network_layer_name = network_layer_name
    sd_model.has_accelerate = False


def reapply_offload():
    """Re-place loaded components after an offload setting changed."""
    if not shared.sd_loaded:
        return
    modular = sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.MODULAR
    if shared.opts.diffusers_offload_mode == 'group' or (modular and shared.opts.diffusers_offload_mode in {'model', 'sequential'}):
        apply_group_offload(shared.sd_model)
    elif shared.opts.diffusers_offload_mode == 'balanced':
        s.offload_hook_instance = None # the hook snapshots the exclusion lists when constructed
        apply_balanced_offload(shared.sd_model)


def apply_model_offload(sd_model, quiet:bool=False):
    try:
        remove_group_offload(sd_model)
        log.quiet(quiet, f'Offload: type={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
        if not hasattr(sd_model, "_all_hooks") or len(sd_model._all_hooks) == 0: # pylint: disable=protected-access
            sd_model.enable_model_cpu_offload(device=devices.device)
        else:
            sd_model.maybe_free_model_hooks()
        set_accelerate(sd_model)
    except Exception as e:
        log.error(f'Offload: type={shared.opts.diffusers_offload_mode} {e}')


def apply_sequential_offload(sd_model, op:str='model', quiet:bool=False):
    try:
        remove_group_offload(sd_model)
        log.quiet(quiet, f'Offload: type={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
        if sd_model.has_accelerate:
            if op == "vae": # reapply sequential offload to vae
                from accelerate import cpu_offload
                sd_model.vae.to(devices.cpu)
                cpu_offload(sd_model.vae, devices.device, offload_buffers=len(sd_model.vae._parameters) > 0) # pylint: disable=protected-access
            else:
                pass # do nothing if offload is already applied
        else:
            sd_model.enable_sequential_cpu_offload(device=devices.device)
        set_accelerate(sd_model)
    except Exception as e:
        log.error(f'Offload: type={shared.opts.diffusers_offload_mode} {e}')


def apply_none_offload(sd_model, quiet:bool=False):
    if shared.sd_model_type not in s.offload_allow_none:
        log.warning(f'Offload: type={shared.opts.diffusers_offload_mode} cls={shared.sd_model.__class__.__name__} large model')
    else:
        log.quiet(quiet, f'Offload: type={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
    try:
        sd_model.has_accelerate = False
        remove_group_offload(sd_model)
        if hasattr(sd_model, 'maybe_free_model_hooks'):
            sd_model.maybe_free_model_hooks()
        sd_model = accelerate.hooks.remove_hook_from_module(sd_model, recurse=True)
    except Exception:
        pass
    sd_models.move_model(sd_model, devices.device, force=True)


def set_diffuser_offload(sd_model, op:str='model', quiet:bool=False, force:bool=False):
    t0 = time.time()
    if sd_model is None:
        log.warning(f'{op} is not loaded')
        return
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False
    if s.accelerate_dtype_byte_size is None:
        s.accelerate_dtype_byte_size = accelerate.utils.modeling.dtype_byte_size
        accelerate.utils.modeling.dtype_byte_size = dtype_byte_size

    if sd_models.get_diffusers_task(sd_model) == sd_models.DiffusersTaskType.MODULAR and shared.opts.diffusers_offload_mode in {'model', 'sequential', 'group'}:
        if shared.opts.diffusers_offload_mode != 'group' and not getattr(sd_model, 'sdnext_modular_offload_warned', False):
            sd_model.sdnext_modular_offload_warned = True
            log.warning(f'Offload: desired={shared.opts.diffusers_offload_mode} override=group reason="modular pipeline"')
        apply_group_offload(sd_model)
        process_timer.add('offload', time.time() - t0)
        return

    if shared.opts.diffusers_offload_mode == "none":
        log.warning('Offload: type=none "use balanced offload with model type set not to offload"')
        apply_none_offload(sd_model, quiet=quiet)

    if shared.opts.diffusers_offload_mode == "model" and hasattr(sd_model, "enable_model_cpu_offload"):
        log.warning('Offload: type=model "use balanced offload instead"')
        apply_model_offload(sd_model, quiet=quiet)

    if shared.opts.diffusers_offload_mode == "sequential" and hasattr(sd_model, "enable_sequential_cpu_offload"):
        apply_sequential_offload(sd_model, op=op, quiet=quiet)

    if shared.opts.diffusers_offload_mode == "group":
        sd_model = apply_group_offload(sd_model)

    if shared.opts.diffusers_offload_mode == "balanced":
        sd_model = apply_balanced_offload(sd_model, force=force)

    process_timer.add('offload', time.time() - t0)
