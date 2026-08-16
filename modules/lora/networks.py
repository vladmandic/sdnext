from contextlib import nullcontext
import time
import rich.progress as rp
from modules.errorlimiter import limit_errors
from modules.lora import lora_common as l
from modules.lora.lora_apply import network_apply_weights, network_apply_direct, network_backup_weights, network_calc_weights
from modules import shared, devices, sd_models
from modules.logger import log, console


applied_layers: list[str] = []
native_active: bool = False
default_components = ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoder_4', 'unet', 'transformer', 'transformer_2', 'llm_adapter']


def group_will_mutate(module, network_layer_name: str, loaded) -> bool:
    """True when the pass will write to this module: a loaded network covers its layer, a
    tensor backup awaits restore, or an svd factor stash awaits removal."""
    if any(net.modules.get(network_layer_name, None) is not None for net in loaded):
        return True
    weights_backup = getattr(module, 'network_weights_backup', None)
    if weights_backup is not None and not isinstance(weights_backup, bool):
        return True
    bias_backup = getattr(module, 'network_bias_backup', None)
    if bias_backup is not None and not isinstance(bias_backup, bool):
        return True
    return getattr(module, 'sdnq_lora_svd_stash', None) is not None


def group_offload_strip(sd_model, component_name: str, stripped: dict):
    """Group offload hooks come off before the first weight write in a component: a write
    under live hooks either replaces a parameter out of the hook's group bookkeeping or is
    lost on the next onload. With hooks removed the weights rest on cpu and the component
    reports its truthful device, so writes land in place; the offload reapply at the end
    of the pass snapshots the result into fresh groups."""
    from modules.sd_offload_group import remove_group_offload_component
    component = getattr(sd_model, component_name, None)
    remove_group_offload_component(component)
    stripped[component_name] = component.device
    return stripped[component_name]


def network_activate(include=None, exclude=None):
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    for net in l.loaded_networks: # promote staged multipliers only now: the deactivate pass ran against the previous values, which fuse-mode removal recomputes with
        pending = getattr(net, 'pending_config', None)
        if pending is not None:
            net.te_multiplier = pending['te']
            net.unet_multiplier = pending['unet']
            net.dyn_dim = pending['dyn']
    t0 = time.time()
    with limit_errors("network_activate") as elimit:
        sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
        if shared.opts.diffusers_offload_mode == "sequential":
            sd_models.disable_offload(sd_model)
            sd_models.move_model(sd_model, device=devices.cpu)
        elif shared.opts.diffusers_offload_mode == "balanced":
            sd_model = sd_models.apply_balanced_offload(sd_model, force=True) # dispatched modules hold meta tensors backed by the offload map; rebuild them real on cpu with hooks intact before touching weights
        group_offload = shared.opts.diffusers_offload_mode == "group"
        group_stripped = {}
        device = None
        modules = {}
        components = include if len(include) > 0 else default_components
        components = [x for x in components if x not in exclude]
        filtered_components = [x for x in default_components if x not in components] # filtered components restore to backup so a filter means detached, not frozen with stale weights
        active_components = []
        for name in components + filtered_components:
            component = getattr(sd_model, name, None)
            if component is not None and hasattr(component, 'named_modules'):
                if name in components:
                    active_components.append(name)
                modules[name] = list(component.named_modules())
        total = sum(len(x) for x in modules.values())
        if len(l.loaded_networks) > 0:
            pbar = rp.Progress(rp.TextColumn('[cyan]Network: type=LoRA action=activate'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=console)
            task = pbar.add_task(description='' , total=total)
        else:
            task = None
            pbar = nullcontext()
        applied_weight = 0
        applied_bias = 0
        with devices.inference_context(), pbar:
            wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in l.loaded_networks) if len(l.loaded_networks) > 0 else ()
            applied_layers.clear()
            backup_size = 0
            for component in modules.keys():
                component_wanted = wanted_names if component in components else ()
                device = getattr(sd_model, component, None).device
                for _, module in modules[component]:
                    network_layer_name = getattr(module, 'network_layer_name', None)
                    current_names = getattr(module, "network_current_names", ())
                    if getattr(module, 'weight', None) is None or shared.state.interrupted or (network_layer_name is None) or (current_names == component_wanted):
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    if group_offload and component not in group_stripped and group_will_mutate(module, network_layer_name, l.loaded_networks):
                        device = group_offload_strip(sd_model, component, group_stripped)
                    backup_size += network_backup_weights(module, network_layer_name, component_wanted)
                    if not component_wanted:
                        weights_backup = getattr(module, "network_weights_backup", None)
                        if weights_backup is None or isinstance(weights_backup, bool): # fuse mode has no tensor backup, restore stays with network_deactivate
                            if task is not None:
                                pbar.update(task, advance=1)
                            continue
                        batch_updown, batch_ex_bias = None, None # restore-only pass, apply with no weights reverts to backup
                    else:
                        batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, elimit=elimit)
                    if shared.opts.lora_fuse_native:
                        network_apply_direct(module, batch_updown, batch_ex_bias, device=device)
                    else:
                        network_apply_weights(module, batch_updown, batch_ex_bias, device=device)
                    if batch_updown is not None or batch_ex_bias is not None:
                        applied_layers.append(network_layer_name)
                        applied_weight += 1 if batch_updown is not None else 0
                        applied_bias += 1 if batch_ex_bias is not None else 0
                    batch_updown, batch_ex_bias = None, None
                    del batch_updown, batch_ex_bias
                    module.network_current_names = component_wanted
                    if task is not None:
                        bs = round(backup_size/1024/1024/1024, 2) if backup_size > 0 else None
                        pbar.update(task, advance=1, description=f'networks={len(l.loaded_networks)} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={bs} device={device}')

            if task is not None and len(applied_layers) == 0:
                pbar.remove_task(task) # hide progress bar for no action
    global native_active # pylint: disable=global-statement
    native_active = len(l.loaded_networks) > 0
    l.timer.activate += time.time() - t0
    if l.debug and len(l.loaded_networks) > 0:
        log.debug(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={round(backup_size/1024/1024/1024, 2)} fuse={shared.opts.lora_fuse_native}:{shared.opts.lora_fuse_diffusers} device={device} time={l.timer.summary}')
    modules.clear()
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential" or len(group_stripped) > 0:
        sd_models.set_diffuser_offload(sd_model, op="model")


def network_deactivate(include=None, exclude=None):
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    if not shared.opts.lora_fuse_native or shared.opts.lora_force_diffusers:
        return
    if len(l.previously_loaded_networks) == 0:
        return
    t0 = time.time()
    with limit_errors("network_deactivate") as elimit:
        sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
        if shared.opts.diffusers_offload_mode == "sequential":
            sd_models.disable_offload(sd_model)
            sd_models.move_model(sd_model, device=devices.cpu)
        elif shared.opts.diffusers_offload_mode == "balanced":
            sd_model = sd_models.apply_balanced_offload(sd_model, force=True) # dispatched modules hold meta tensors backed by the offload map; rebuild them real on cpu with hooks intact before touching weights
        group_offload = shared.opts.diffusers_offload_mode == "group"
        group_stripped = {}
        modules = {}

        components = include if len(include) > 0 else ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'unet', 'transformer', 'llm_adapter']
        components = [x for x in components if x not in exclude]
        active_components = []
        for name in components:
            component = getattr(sd_model, name, None)
            if component is not None and hasattr(component, 'named_modules'):
                modules[name] = list(component.named_modules())
                active_components.append(name)
        total = sum(len(x) for x in modules.values())
        if len(l.previously_loaded_networks) > 0 and l.debug:
            pbar = rp.Progress(rp.TextColumn('[cyan]Network: type=LoRA action=deactivate'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=console)
            task = pbar.add_task(description='', total=total)
        else:
            task = None
            pbar = nullcontext()
        with devices.inference_context(), pbar:
            applied_layers.clear()
            for component in modules.keys():
                device = getattr(sd_model, component, None).device
                for _, module in modules[component]:
                    network_layer_name = getattr(module, 'network_layer_name', None)
                    if shared.state.interrupted or network_layer_name is None:
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    if group_offload and component not in group_stripped and group_will_mutate(module, network_layer_name, l.previously_loaded_networks):
                        device = group_offload_strip(sd_model, component, group_stripped)
                    batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, use_previous=True, elimit=elimit)
                    if shared.opts.lora_fuse_native:
                        network_apply_direct(module, batch_updown, batch_ex_bias, device=device, deactivate=True)
                    else:
                        network_apply_weights(module, batch_updown, batch_ex_bias, device=device, deactivate=True)
                    if batch_updown is not None or batch_ex_bias is not None:
                        applied_layers.append(network_layer_name)
                    del batch_updown, batch_ex_bias
                    module.network_current_names = ()
                    if task is not None:
                        pbar.update(task, advance=1, description=f'networks={len(l.previously_loaded_networks)} modules={active_components} layers={total} unapply={len(applied_layers)}')
    l.timer.deactivate = time.time() - t0
    if l.debug and len(l.previously_loaded_networks) > 0:
        log.debug(f'Network deactivate: type=LoRA networks={[n.name for n in l.previously_loaded_networks]} modules={active_components} layers={total} apply={len(applied_layers)} fuse={shared.opts.lora_fuse_native}:{shared.opts.lora_fuse_diffusers} time={l.timer.summary}')
    modules.clear()
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential" or len(group_stripped) > 0:
        sd_models.set_diffuser_offload(sd_model, op="model")
