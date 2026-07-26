from contextlib import nullcontext
import time
import rich.progress as rp
from modules.errorlimiter import limit_errors
from modules.lora import lora_common as l
from modules.lora import lora_overrides
from modules.lora import lora_sdnq
from modules.lora import lora_stack
from modules.lora.lora_apply import network_apply_weights, network_apply_direct, network_backup_weights, network_calc_weights
from modules import shared, devices, sd_models
from modules.logger import log, console


applied_layers: list[str] = []
native_active: bool = False
default_components = ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'text_encoder_4', 'unet', 'transformer', 'transformer_2', 'llm_adapter']


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
    fuse = lora_overrides.fuse_native() # resolve once: backup and apply passes must agree
    with limit_errors("network_activate") as elimit:
        sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
        if shared.opts.diffusers_offload_mode == "sequential":
            sd_models.disable_offload(sd_model)
            sd_models.move_model(sd_model, device=devices.cpu)
        elif shared.opts.diffusers_offload_mode == "balanced":
            sd_model = sd_models.apply_balanced_offload(sd_model, force=True, silent=True) # dispatched modules hold meta tensors backed by the offload map; rebuild them real on cpu with hooks intact before touching weights
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
            stack_sig = lora_stack.signature() # tracked beside network_current_names so stack-setting changes re-apply
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
                    current_names = getattr(module, "network_current_names", ())
                    if getattr(module, 'weight', None) is None or shared.state.interrupted or (network_layer_name is None) or (current_names == component_wanted and getattr(module, 'network_current_stack', 'sum') == stack_sig):
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    lora_stack.drop(network_layer_name) # re-application invalidates any live selection schedule; the select branch re-registers
                    calced = False # tracks whether this iteration assembled the delta, so the fallthrough reuses it instead of recomputing
                    if select_active and component_wanted and not network_layer_name.startswith('lora_te'):
                        if lora_sdnq.select_candidate(module, network_layer_name, component_wanted): # SDNQ pairs ride the channel as separate segments at any bit width; weight rewrites cannot flip a quantized layer
                            weights_backup = getattr(module, "network_weights_backup", None)
                            if weights_backup is not None and not isinstance(weights_backup, bool):
                                network_apply_weights(module, None, None, device=device)
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
                            weights_backup = getattr(module, "network_weights_backup", None)
                            if weights_backup is not None and not isinstance(weights_backup, bool):
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
                        weights_backup = getattr(module, "network_weights_backup", None)
                        if weights_backup is not None and not isinstance(weights_backup, bool):
                            network_apply_weights(module, None, None, device=device) # an earlier non-factorable set requantized this layer, restore the pristine base before attaching factors
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
                        weights_backup = getattr(module, "network_weights_backup", None)
                        if weights_backup is not None and not isinstance(weights_backup, bool):
                            network_apply_weights(module, None, None, device=device) # the hosted delta is measured against the pristine base
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
                    backup_size += network_backup_weights(module, network_layer_name, component_wanted, fuse)
                    if not component_wanted:
                        lora_stack.drop(network_layer_name) # a restored layer must leave the selection schedule
                        weights_backup = getattr(module, "network_weights_backup", None)
                        if weights_backup is None or isinstance(weights_backup, bool): # fuse mode has no tensor backup, restore stays with network_deactivate
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
                    module.network_current_stack = stack_sig
                    if task is not None:
                        bs = round(backup_size/1024/1024/1024, 2) if backup_size > 0 else None
                        pbar.update(task, advance=1, description=f'networks={len(l.loaded_networks)} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={bs} device={device}')

            if task is not None and len(applied_layers) == 0:
                pbar.remove_task(task) # hide progress bar for no action
    lora_sdnq.report_fallbacks()
    global native_active # pylint: disable=global-statement
    native_active = len(l.loaded_networks) > 0
    l.last_backup_size = backup_size
    l.timer.activate += time.time() - t0
    if l.debug and len(l.loaded_networks) > 0:
        log.debug(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={round(backup_size/1024/1024/1024, 2)} fuse={fuse}:{shared.opts.lora_fuse_diffusers} device={device} time={l.timer.summary}')
    modules.clear()
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential":
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
        sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
        if shared.opts.diffusers_offload_mode == "sequential":
            sd_models.disable_offload(sd_model)
            sd_models.move_model(sd_model, device=devices.cpu)
        elif shared.opts.diffusers_offload_mode == "balanced":
            sd_model = sd_models.apply_balanced_offload(sd_model, force=True, silent=True) # dispatched modules hold meta tensors backed by the offload map; rebuild them real on cpu with hooks intact before touching weights
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
                    if lora_sdnq.remove_factors(module): # exact inverse for factor-mode layers, weights were never touched
                        applied_layers.append(network_layer_name)
                        module.network_current_names = ()
                        if task is not None:
                            pbar.update(task, advance=1)
                        continue
                    batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, use_previous=True, elimit=elimit)
                    if fuse:
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
        log.debug(f'Network deactivate: type=LoRA networks={[n.name for n in l.previously_loaded_networks]} modules={active_components} layers={total} apply={len(applied_layers)} fuse={fuse}:{shared.opts.lora_fuse_diffusers} time={l.timer.summary}')
    modules.clear()
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential":
        sd_models.set_diffuser_offload(sd_model, op="model")
