from contextlib import nullcontext
import time
import rich.progress as rp
from modules.errorlimiter import limit_errors
from modules.lora import lora_blocks
from modules.lora import lora_common as l
from modules.lora import lora_overrides
from modules.lora import lora_sdnq
from modules.lora import lora_stack
from modules.lora.lora_apply import network_apply_weights, network_apply_direct, network_backup_weights, network_calc_weights
from modules import shared, devices, sd_models
from modules.logger import log, console


applied_layers: list[str] = []
refused_writes: int = 0 # deltas the modules would not take on the last activate pass; infotext reports the network as partial
native_active: bool = False
default_components = ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoder_4', 'unet', 'transformer', 'transformer_2', 'llm_adapter']
deactivate_components = ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'unet', 'transformer', 'llm_adapter']


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


def promote_pending():
    """Promote staged multipliers onto the loaded networks; the deactivate pass ran against the previous values, which fuse-mode removal recomputes with."""
    for net in l.loaded_networks:
        pending = getattr(net, 'pending_config', None)
        if pending is not None:
            net.te_multiplier = pending['te']
            net.unet_multiplier = pending['unet']
            net.dyn_dim = pending['dyn']
            net.block_spec = pending.get('blocks', None)
            net.pending_config = None # promotion is one-shot


def prepare_model_for_write(sd_model):
    """Bring the model into a state where weight writes land; balanced offload returns a rebuilt model."""
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.disable_offload(sd_model)
        sd_models.move_model(sd_model, device=devices.cpu)
    elif shared.opts.diffusers_offload_mode == "balanced":
        sd_model = sd_models.apply_balanced_offload(sd_model, force=True, silent=True) # dispatched modules hold meta tensors backed by the offload map; rebuild them real on cpu with hooks intact before touching weights
    return sd_model


def collect_components(sd_model, include, exclude, defaults, restore_filtered):
    """Modules to walk, as (modules, wanted components, walked component names, module count).

    With restore_filtered the walk also covers the components a filter left
    out, so they restore to backup instead of freezing with stale weights;
    those names stay out of the reported list because nothing applies to them.
    """
    components = include if len(include) > 0 else defaults
    components = [x for x in components if x not in exclude]
    filtered = [x for x in defaults if x not in components] if restore_filtered else []
    modules = {}
    active_components = []
    for name in components + filtered:
        component = getattr(sd_model, name, None)
        if component is not None and hasattr(component, 'named_modules'):
            if name in components:
                active_components.append(name)
            modules[name] = list(component.named_modules())
    return modules, components, active_components, sum(len(x) for x in modules.values())


def pass_progress(action, total, show):
    """Progress bar for one pass, or a nullcontext with no task when there is nothing to show."""
    if not show:
        return nullcontext(), None
    pbar = rp.Progress(rp.TextColumn(f'[cyan]Network: type=LoRA action={action}'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=console)
    return pbar, pbar.add_task(description='', total=total)


def tensor_backup(module):
    """The module's weight backup when it holds real tensors; None in fuse mode, where the backup is a marker."""
    weights_backup = getattr(module, 'network_weights_backup', None)
    return None if isinstance(weights_backup, bool) else weights_backup


def restore_pristine(module, device):
    """Put a backed-up layer back on its checkpoint weights, so a mechanism sees the pristine base."""
    if tensor_backup(module) is not None:
        network_apply_weights(module, None, None, device=device)


def should_skip(module, network_layer_name, wanted, stack_sig):
    """True when the pass has nothing to do here: no weight, interrupted, unnamed, or already carrying this set under these settings."""
    if getattr(module, 'weight', None) is None or shared.state.interrupted or network_layer_name is None:
        return True
    return getattr(module, 'network_current_names', ()) == wanted and getattr(module, 'network_current_stack', 'sum') == stack_sig


def network_activate(include=None, exclude=None):
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    promote_pending()
    t0 = time.time()
    fuse = lora_overrides.fuse_native() # resolve once: backup and apply passes must agree
    with limit_errors("network_activate") as elimit:
        sd_model = prepare_model_for_write(getattr(shared.sd_model, "pipe", shared.sd_model))
        group_offload = shared.opts.diffusers_offload_mode == "group"
        group_stripped = {}
        device = None
        modules, components, active_components, total = collect_components(sd_model, include, exclude, default_components, restore_filtered=True)
        pbar, task = pass_progress('activate', total, len(l.loaded_networks) > 0)
        applied_weight = 0
        applied_bias = 0
        refused = 0
        with devices.inference_context(), pbar:
            wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in l.loaded_networks) if len(l.loaded_networks) > 0 else ()
            stack_sig = lora_stack.signature() + lora_blocks.signature() + lora_sdnq.signature() # tracked beside network_current_names so stack-setting, block-weight and mechanism changes re-apply
            select_active = len(l.loaded_networks) > 0 and lora_stack.active_select(len(l.loaded_networks)) # restore-only walks have nothing to stack; the count warning would fire on every network-free generation
            applied_layers.clear()
            lora_sdnq.fallback_layers.clear() # a raise mid-pass leaves stale entries behind
            lora_sdnq.hosted_layers.clear()
            lora_sdnq.factor_layers.clear()
            lora_sdnq.select_layers.clear()
            backup_size = 0
            for component in modules.keys():
                component_wanted = wanted_names if component in components else ()
                device = getattr(sd_model, component, None).device
                for _, module in modules[component]:
                    network_layer_name = getattr(module, 'network_layer_name', None)
                    if should_skip(module, network_layer_name, component_wanted, stack_sig):
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    lora_stack.drop(network_layer_name) # re-application invalidates any live selection schedule; the select branch re-registers
                    if group_offload and component not in group_stripped and group_will_mutate(module, network_layer_name, l.loaded_networks):
                        device = group_offload_strip(sd_model, component, group_stripped)
                    calced = False # tracks whether this iteration assembled the delta, so the fallthrough reuses it instead of recomputing
                    if select_active and component_wanted and not network_layer_name.startswith('lora_te'):
                        if lora_sdnq.select_candidate(module, network_layer_name, component_wanted): # SDNQ pairs ride the channel as separate segments at any bit width; weight rewrites cannot flip a quantized layer
                            restore_pristine(module, device)
                            applied = lora_sdnq.apply_select_cached(module, network_layer_name, component_wanted) # a stored score record and factor pair serve before the deltas are assembled
                            if applied is None:
                                per_net, sel_bias = network_calc_weights(module, network_layer_name, elimit=elimit, per_net=True)
                                if sel_bias is None:
                                    applied = lora_sdnq.apply_select(module, network_layer_name, per_net, component_wanted)
                            if applied is not None:
                                if applied and component_wanted:
                                    applied_layers.append(network_layer_name)
                                    applied_weight += 1
                                module.network_current_names = component_wanted
                                module.network_current_stack = stack_sig
                                if task is not None:
                                    pbar.update(task, advance=1)
                                continue
                            lora_stack.warn_once('select-unridable', f'Network stack: mode={lora_stack.mode()} layer="{network_layer_name}" fallback=sum') # a pair the channel cannot carry (bias delta or malformed member) sums like any unsupported set
                        elif getattr(module, 'sdnq_dequantizer', None) is not None: # hosting disabled: quantized layers have no side-channel to carry segments and packed backups cannot flip, so the sum paths below take the layer
                            if any(net.modules.get(network_layer_name, None) is not None for net in l.loaded_networks):
                                lora_stack.warn_once('select-host-disabled', f'Network stack: mode={lora_stack.mode()} quant=sdnq host=disabled fallback=sum')
                        else: # other layers select by recomputing the winner from the pristine backup at schedule time
                            sel_backup = network_backup_weights(module, network_layer_name, component_wanted, fuse)
                            if tensor_backup(module) is not None: # a flip recomputes the winner from the pristine tensor, which fuse mode does not keep
                                if lora_stack.register_weight_pair_cached(network_layer_name, module, component_wanted): # a stored score record registers without assembling the pair
                                    backup_size += sel_backup
                                    network_apply_weights(module, None, None, device=device) # pristine until the schedule applies the winner
                                    applied_layers.append(network_layer_name)
                                    applied_weight += 1
                                    module.network_current_names = component_wanted
                                    module.network_current_stack = stack_sig
                                    if task is not None:
                                        pbar.update(task, advance=1)
                                    continue
                                per_net, sel_bias = network_calc_weights(module, network_layer_name, elimit=elimit, per_net=True)
                                if sel_bias is None and lora_stack.register_weight_pair(network_layer_name, module, per_net, component_wanted):
                                    backup_size += sel_backup # counted only when this branch keeps the layer; the fallthrough re-enters the shared backup call below, which counts it then
                                    network_apply_weights(module, None, None, device=device) # pristine until the schedule applies the winner
                                    applied_layers.append(network_layer_name)
                                    applied_weight += 1
                                    module.network_current_names = component_wanted
                                    module.network_current_stack = stack_sig
                                    if task is not None:
                                        pbar.update(task, advance=1)
                                    continue
                    if lora_sdnq.factor_candidate(module, network_layer_name, component_wanted):
                        restore_pristine(module, device) # an earlier non-factorable set may have requantized this layer
                        applied = lora_sdnq.apply_factors(module, network_layer_name, component_wanted)
                        if applied is not None: # exact path took the layer; None falls through to hosting or requantize
                            if applied and component_wanted:
                                applied_layers.append(network_layer_name)
                                applied_weight += 1
                            module.network_current_names = component_wanted
                            module.network_current_stack = stack_sig
                            if task is not None:
                                pbar.update(task, advance=1)
                            continue
                    if lora_sdnq.host_candidate(module, network_layer_name, component_wanted):
                        restore_pristine(module, device) # the hosted delta is measured against the pristine base
                        hosted = lora_sdnq.apply_cached(module, network_layer_name, component_wanted) # a stored entry serves the layer before the delta is assembled
                        if hosted is None:
                            batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, elimit=elimit)
                            calced = True
                            if batch_ex_bias is None: # bias deltas need the plain path; weight-only sets ride the side-channel without a weight backup
                                hosted = lora_sdnq.apply_hosted(module, network_layer_name, batch_updown, component_wanted)
                            if hosted is not None:
                                batch_updown, batch_ex_bias = None, None
                                del batch_updown, batch_ex_bias
                        if hosted is not None:
                            if hosted and component_wanted:
                                applied_layers.append(network_layer_name)
                                applied_weight += 1
                            module.network_current_names = component_wanted
                            module.network_current_stack = stack_sig
                            if task is not None:
                                pbar.update(task, advance=1)
                            continue
                    stripped = lora_sdnq.remove_factors(module) # the mechanism gate can decline a layer still carrying attached factors; the weight path must start from the pristine channel
                    if stripped and not component_wanted: # factor-mode layers have no tensor backup, dropping the factors is the whole restore
                        module.network_current_names = ()
                        module.network_current_stack = stack_sig
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    backup_size += network_backup_weights(module, network_layer_name, component_wanted, fuse)
                    if not component_wanted:
                        lora_stack.drop(network_layer_name) # a restored layer must leave the selection schedule
                        if tensor_backup(module) is None: # fuse mode has no tensor backup, restore stays with network_deactivate
                            if task is not None:
                                pbar.update(task, advance=1)
                            continue
                        batch_updown, batch_ex_bias = None, None # restore-only pass, apply with no weights reverts to backup
                    else:
                        if not calced: # the host branch may have assembled the delta already; a declined layer reuses it
                            batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, elimit=elimit)
                        if batch_updown is not None:
                            lora_sdnq.note_fallback(module, network_layer_name) # only layers whose quantized weight actually takes a delta
                    if fuse:
                        weight_written, bias_written = network_apply_direct(module, batch_updown, batch_ex_bias, device=device)
                    else:
                        weight_written, bias_written = network_apply_weights(module, batch_updown, batch_ex_bias, device=device)
                    if batch_updown is not None or batch_ex_bias is not None:
                        applied_layers.append(network_layer_name)
                        applied_weight += 1 if weight_written else 0
                        applied_bias += 1 if bias_written else 0
                        refused += (batch_updown is not None and not weight_written) + (batch_ex_bias is not None and not bias_written) # a delta the module would not take leaves that layer on its base value
                    batch_updown, batch_ex_bias = None, None
                    del batch_updown, batch_ex_bias
                    module.network_current_names = component_wanted
                    module.network_current_stack = stack_sig
                    if task is not None:
                        bs = round(backup_size/1024/1024/1024, 2) if backup_size > 0 else None
                        pbar.update(task, advance=1, description=f'networks={len(l.loaded_networks)} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={bs} device={device}')

            if task is not None and len(applied_layers) == 0:
                pbar.remove_task(task) # hide progress bar for no action
    global native_active, refused_writes # pylint: disable=global-statement
    lora_sdnq.report_fallbacks()
    native_active = len(l.loaded_networks) > 0
    refused_writes = refused
    l.last_backup_size = backup_size
    l.timer.activate += time.time() - t0
    if refused > 0:
        log.error(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} weights={applied_weight} bias={applied_bias} refused={refused} network partially applied')
    if l.debug and len(l.loaded_networks) > 0:
        log.debug(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} refused={refused} backup={round(backup_size/1024/1024/1024, 2)} fuse={fuse}:{shared.opts.lora_fuse_diffusers} device={device} time={l.timer.summary}')
    modules.clear()
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential" or len(group_stripped) > 0:
        sd_models.set_diffuser_offload(sd_model, op="model")


def effective_mode():
    """Weight-state label for load logs: backup and fuse say how touched weights restore, factor means the whole load rode the svd channel and unload just drops factors."""
    if getattr(l, 'last_backup_size', 0) > 0:
        return 'backup'
    if lora_overrides.fuse_native():
        return 'fuse'
    return 'factor'


def network_deactivate(include=None, exclude=None):
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    fuse = lora_overrides.fuse_native() # must match network_activate: backup mode restores in its restore-only pass instead
    if not fuse or shared.opts.lora_force_diffusers:
        return
    if len(l.previously_loaded_networks) == 0:
        return
    t0 = time.time()
    with limit_errors("network_deactivate") as elimit:
        sd_model = prepare_model_for_write(getattr(shared.sd_model, "pipe", shared.sd_model))
        group_offload = shared.opts.diffusers_offload_mode == "group"
        group_stripped = {}
        modules, _components, active_components, total = collect_components(sd_model, include, exclude, deactivate_components, restore_filtered=False)
        pbar, task = pass_progress('deactivate', total, len(l.previously_loaded_networks) > 0 and l.debug)
        refused = 0
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
                    if lora_sdnq.remove_factors(module): # exact inverse for factor-mode layers, weights were never touched
                        applied_layers.append(network_layer_name)
                        module.network_current_names = ()
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, use_previous=True, elimit=elimit)
                    if fuse:
                        weight_written, bias_written = network_apply_direct(module, batch_updown, batch_ex_bias, device=device, deactivate=True)
                    else:
                        weight_written, bias_written = network_apply_weights(module, batch_updown, batch_ex_bias, device=device, deactivate=True)
                    if batch_updown is not None or batch_ex_bias is not None:
                        applied_layers.append(network_layer_name)
                        refused += (batch_updown is not None and not weight_written) + (batch_ex_bias is not None and not bias_written) # a delta the module would not take stays applied on that layer
                    del batch_updown, batch_ex_bias
                    module.network_current_names = ()
                    if task is not None:
                        pbar.update(task, advance=1, description=f'networks={len(l.previously_loaded_networks)} modules={active_components} layers={total} unapply={len(applied_layers)}')
    l.timer.deactivate += time.time() - t0
    if refused > 0:
        log.error(f'Network unload: type=LoRA networks={[n.name for n in l.previously_loaded_networks]} unapply={len(applied_layers)} refused={refused} network partially removed')
    if l.debug and len(l.previously_loaded_networks) > 0:
        log.debug(f'Network deactivate: type=LoRA networks={[n.name for n in l.previously_loaded_networks]} modules={active_components} layers={total} apply={len(applied_layers)} refused={refused} fuse={fuse}:{shared.opts.lora_fuse_diffusers} time={l.timer.summary}')
    modules.clear()
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential" or len(group_stripped) > 0:
        sd_models.set_diffuser_offload(sd_model, op="model")
