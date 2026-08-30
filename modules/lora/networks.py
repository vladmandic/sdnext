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


class ActivationPass:
    """State of one activation walk, built before the walk so a pass that raises still has it.

    `wanted_names` is built once here and reaches every layer as
    `component_wanted`, either this tuple or the empty one. The factor cache
    keys its pass entry on that object's identity, so an equal tuple rebuilt
    per component would send every lookup back to disk.
    """

    def __init__(self, fuse):
        self.sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
        self.fuse = fuse
        self.elimit = None # the error limiter, bound for the duration of the walk
        self.wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in l.loaded_networks) if len(l.loaded_networks) > 0 else ()
        self.stack_sig = lora_stack.signature() + lora_blocks.signature() + lora_sdnq.signature() # tracked beside network_current_names so stack-setting, block-weight and mechanism changes re-apply
        self.select_active = len(l.loaded_networks) > 0 and lora_stack.active_select(len(l.loaded_networks)) # restore-only walks have nothing to stack; the count warning would fire on every network-free generation
        self.component_wanted = ()
        self.device = None
        self.group_offload = shared.opts.diffusers_offload_mode == "group"
        self.group_stripped = {}
        self.pbar = nullcontext()
        self.task = None
        self.total = 0
        self.active_components = []
        self.applied_weight = 0
        self.applied_bias = 0
        self.refused = 0
        self.backup_size = 0

    def stamp(self, module):
        """Mark the layer as carrying this set under these settings; the pair is the skip key."""
        module.network_current_names = self.component_wanted
        module.network_current_stack = self.stack_sig

    def tick(self, description=None):
        if self.task is None:
            return
        if description is None:
            self.pbar.update(self.task, advance=1)
        else:
            self.pbar.update(self.task, advance=1, description=description)

    def claim(self, module, network_layer_name, changed):
        """Accept a layer one of the mechanisms took; only a layer whose weights changed counts as applied."""
        if changed and self.component_wanted:
            applied_layers.append(network_layer_name)
            self.applied_weight += 1
        self.stamp(module)
        self.tick()

    def keep_selected(self, module, network_layer_name, sel_backup):
        """Hold a scheduled weight-kind layer on its pristine tensor until the schedule applies the winner."""
        self.backup_size += sel_backup # counted only where this branch keeps the layer; the weight path below re-enters the shared backup call, which counts it then
        network_apply_weights(module, None, None, device=self.device)
        self.claim(module, network_layer_name, True)
        return True


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


def try_select(ctx, module, network_layer_name):
    """Put the layer under a selection schedule; True when it took the layer.

    The three arms are mutually exclusive and their warnings are keyed, so a
    layer that cannot be scheduled reports one reason and falls through.
    """
    if not ctx.select_active or not ctx.component_wanted or network_layer_name.startswith('lora_te'):
        return False
    if lora_sdnq.select_candidate(module, network_layer_name, ctx.component_wanted): # SDNQ pairs ride the channel as separate segments at any bit width; weight rewrites cannot flip a quantized layer
        restore_pristine(module, ctx.device)
        applied = lora_sdnq.apply_select_cached(module, network_layer_name, ctx.component_wanted) # a stored score record and factor pair serve before the deltas are assembled
        if applied is None:
            per_net, sel_bias = network_calc_weights(module, network_layer_name, elimit=ctx.elimit, per_net=True)
            if sel_bias is None:
                applied = lora_sdnq.apply_select(module, network_layer_name, per_net, ctx.component_wanted)
        if applied is not None:
            ctx.claim(module, network_layer_name, applied)
            return True
        lora_stack.warn_once('select-unridable', f'Network stack: mode={lora_stack.mode()} layer="{network_layer_name}" fallback=sum') # a pair the channel cannot carry (bias delta or malformed member) sums like any unsupported set
    elif getattr(module, 'sdnq_dequantizer', None) is not None: # hosting disabled: quantized layers have no side-channel to carry segments and packed backups cannot flip, so the sum paths below take the layer
        if any(net.modules.get(network_layer_name, None) is not None for net in l.loaded_networks):
            lora_stack.warn_once('select-host-disabled', f'Network stack: mode={lora_stack.mode()} quant=sdnq host=disabled fallback=sum')
    else: # other layers select by recomputing the winner from the pristine backup at schedule time
        sel_backup = network_backup_weights(module, network_layer_name, ctx.component_wanted, ctx.fuse)
        if tensor_backup(module) is not None: # a flip recomputes the winner from the pristine tensor, which fuse mode does not keep
            if lora_stack.register_weight_pair_cached(network_layer_name, module, ctx.component_wanted): # a stored score record registers without assembling the pair
                return ctx.keep_selected(module, network_layer_name, sel_backup)
            per_net, sel_bias = network_calc_weights(module, network_layer_name, elimit=ctx.elimit, per_net=True)
            if sel_bias is None and lora_stack.register_weight_pair(network_layer_name, module, per_net, ctx.component_wanted):
                return ctx.keep_selected(module, network_layer_name, sel_backup)
    return False


def try_factors(ctx, module, network_layer_name):
    """Attach the set to the quantized side channel as exact factors; True when it took the layer."""
    if not lora_sdnq.factor_candidate(module, network_layer_name, ctx.component_wanted):
        return False
    restore_pristine(module, ctx.device) # an earlier non-factorable set may have requantized this layer
    applied = lora_sdnq.apply_factors(module, network_layer_name, ctx.component_wanted)
    if applied is None: # the exact path declined; hosting or the weight path takes the layer
        return False
    ctx.claim(module, network_layer_name, applied)
    return True


def try_hosted(ctx, module, network_layer_name):
    """Host the combined delta on the side channel as truncated factors.

    Returns whether it took the layer and, when it declined after assembling
    the delta, that delta, so the weight path applies it without a second
    calc. A returned pair of Nones still counts as assembled.
    """
    if not lora_sdnq.host_candidate(module, network_layer_name, ctx.component_wanted):
        return False, None
    restore_pristine(module, ctx.device) # the hosted delta is measured against the pristine base
    batch = None
    hosted = lora_sdnq.apply_cached(module, network_layer_name, ctx.component_wanted) # a stored entry serves the layer before the delta is assembled
    if hosted is None:
        batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, elimit=ctx.elimit)
        batch = (batch_updown, batch_ex_bias)
        if batch_ex_bias is None: # bias deltas need the plain path; weight-only sets ride the side-channel without a weight backup
            hosted = lora_sdnq.apply_hosted(module, network_layer_name, batch_updown, ctx.component_wanted)
        if hosted is not None:
            batch = None # hosting took the delta
    if hosted is None:
        return False, batch
    ctx.claim(module, network_layer_name, hosted)
    return True, None


def apply_generic(ctx, module, network_layer_name, batch):
    """The weight path, which takes any layer the mechanisms above declined."""
    stripped = lora_sdnq.remove_factors(module) # the mechanism gate can decline a layer still carrying attached factors; the weight path must start from the pristine channel
    if stripped and not ctx.component_wanted: # factor-mode layers have no tensor backup, dropping the factors is the whole restore
        ctx.stamp(module)
        ctx.tick()
        return
    ctx.backup_size += network_backup_weights(module, network_layer_name, ctx.component_wanted, ctx.fuse)
    if not ctx.component_wanted:
        lora_stack.drop(network_layer_name) # a restored layer must leave the selection schedule
        if tensor_backup(module) is None: # fuse mode has no tensor backup, restore stays with network_deactivate
            ctx.tick()
            return
        batch_updown, batch_ex_bias = None, None # restore-only pass, apply with no weights reverts to backup
    else:
        batch_updown, batch_ex_bias = batch if batch is not None else network_calc_weights(module, network_layer_name, elimit=ctx.elimit)
        if batch_updown is not None:
            lora_sdnq.note_fallback(module, network_layer_name) # only layers whose quantized weight actually takes a delta
    if ctx.fuse:
        weight_written, bias_written = network_apply_direct(module, batch_updown, batch_ex_bias, device=ctx.device)
    else:
        weight_written, bias_written = network_apply_weights(module, batch_updown, batch_ex_bias, device=ctx.device)
    if batch_updown is not None or batch_ex_bias is not None:
        applied_layers.append(network_layer_name)
        ctx.applied_weight += 1 if weight_written else 0
        ctx.applied_bias += 1 if bias_written else 0
        ctx.refused += (batch_updown is not None and not weight_written) + (batch_ex_bias is not None and not bias_written) # a delta the module would not take leaves that layer on its base value
    ctx.stamp(module)
    bs = round(ctx.backup_size/1024/1024/1024, 2) if ctx.backup_size > 0 else None
    ctx.tick(f'networks={len(l.loaded_networks)} modules={ctx.active_components} layers={ctx.total} weights={ctx.applied_weight} bias={ctx.applied_bias} backup={bs} device={ctx.device}')


def finish_pass(ctx, t0):
    """Publish what the pass did and put the model back under its offload mode.

    Runs even when the error limiter aborts the walk: the hooks it stripped
    and the offload it disabled have to come back, and the counters other
    modules read have to describe this pass.
    """
    global native_active, refused_writes # pylint: disable=global-statement
    lora_sdnq.report_fallbacks()
    native_active = len(l.loaded_networks) > 0
    refused_writes = ctx.refused
    l.last_backup_size = ctx.backup_size
    l.last_mode = 'backup' if ctx.backup_size > 0 else ('fuse' if ctx.fuse else 'factor')
    l.timer.activate += time.time() - t0
    if ctx.refused > 0:
        log.error(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} weights={ctx.applied_weight} bias={ctx.applied_bias} refused={ctx.refused} network partially applied')
    if l.debug and len(l.loaded_networks) > 0:
        log.debug(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} modules={ctx.active_components} layers={ctx.total} weights={ctx.applied_weight} bias={ctx.applied_bias} refused={ctx.refused} backup={round(ctx.backup_size/1024/1024/1024, 2)} fuse={ctx.fuse}:{shared.opts.lora_fuse_diffusers} device={ctx.device} time={l.timer.summary}')
    if len(applied_layers) > 0 or shared.opts.diffusers_offload_mode == "sequential" or len(ctx.group_stripped) > 0:
        sd_models.set_diffuser_offload(ctx.sd_model, op="model")


def network_activate(include=None, exclude=None):
    if exclude is None:
        exclude = []
    if include is None:
        include = []
    promote_pending()
    t0 = time.time()
    ctx = ActivationPass(lora_overrides.fuse_native()) # fuse resolved once: the backup, apply and restore paths must agree
    applied_layers.clear()
    lora_sdnq.reset_pass()
    modules = {}
    try:
        with limit_errors("network_activate") as elimit:
            ctx.elimit = elimit
            ctx.sd_model = prepare_model_for_write(ctx.sd_model)
            modules, components, ctx.active_components, ctx.total = collect_components(ctx.sd_model, include, exclude, default_components, restore_filtered=True)
            ctx.pbar, ctx.task = pass_progress('activate', ctx.total, len(l.loaded_networks) > 0)
            with devices.inference_context(), ctx.pbar:
                for component in modules.keys():
                    ctx.component_wanted = ctx.wanted_names if component in components else () # the pass tuple itself, never a copy
                    ctx.device = getattr(ctx.sd_model, component, None).device
                    for _, module in modules[component]:
                        network_layer_name = getattr(module, 'network_layer_name', None)
                        if should_skip(module, network_layer_name, ctx.component_wanted, ctx.stack_sig):
                            ctx.tick()
                            continue
                        lora_stack.drop(network_layer_name) # re-application invalidates any live selection schedule; the select branch re-registers
                        if ctx.group_offload and component not in ctx.group_stripped and group_will_mutate(module, network_layer_name, l.loaded_networks):
                            ctx.device = group_offload_strip(ctx.sd_model, component, ctx.group_stripped)
                        if try_select(ctx, module, network_layer_name):
                            continue
                        if try_factors(ctx, module, network_layer_name):
                            continue
                        hosted, batch = try_hosted(ctx, module, network_layer_name)
                        if hosted:
                            continue
                        apply_generic(ctx, module, network_layer_name, batch)
                if ctx.task is not None and len(applied_layers) == 0:
                    ctx.pbar.remove_task(ctx.task) # hide progress bar for no action
    finally:
        finish_pass(ctx, t0)
        modules.clear()


def effective_mode():
    """Weight-state label for load logs: backup and fuse say how touched weights restore, factor means the whole load rode the svd channel and unload just drops factors.

    Recorded by the pass rather than derived here, so the unload line
    describes the pass being unloaded even when the settings it ran under
    have since changed.
    """
    if l.last_mode:
        return l.last_mode
    return 'fuse' if lora_overrides.fuse_native() else 'factor'


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
        modules, _components, active_components, total = collect_components(sd_model, include, exclude, default_components, restore_filtered=False)
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
