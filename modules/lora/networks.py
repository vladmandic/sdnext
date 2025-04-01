from contextlib import nullcontext
import time
import rich.progress as rp
from modules.lora.lora_common import timer, debug, loaded_networks, previously_loaded_networks
from modules.lora.lora_apply import network_apply_weights, network_apply_direct, network_backup_weights, network_calc_weights
from modules import shared, devices, sd_models


applied_layers: list[str] = []


def network_activate(include=[], exclude=[]):
    t0 = time.time()
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatiblility
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.disable_offload(sd_model)
        sd_models.move_model(sd_model, device=devices.cpu)
    modules = {}
    components = include if len(include) > 0 else ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'unet', 'transformer']
    components = [x for x in components if x not in exclude]
    active_components = []
    for name in components:
        component = getattr(sd_model, name, None)
        if component is not None and hasattr(component, 'named_modules'):
            active_components.append(name)
            modules[name] = list(component.named_modules())
    total = sum(len(x) for x in modules.values())
    if len(loaded_networks) > 0:
        pbar = rp.Progress(rp.TextColumn('[cyan]Network: type=LoRA action=activate'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
        task = pbar.add_task(description='' , total=total)
    else:
        task = None
        pbar = nullcontext()
    applied_weight = 0
    applied_bias = 0
    device = devices.device if shared.opts.lora_apply_gpu else devices.cpu
    with devices.inference_context(), pbar:
        wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks) if len(loaded_networks) > 0 else ()
        applied_layers.clear()
        backup_size = 0
        for component in modules.keys():
            orig_device = getattr(sd_model, component, None).device
            for _, module in modules[component]:
                network_layer_name = getattr(module, 'network_layer_name', None)
                current_names = getattr(module, "network_current_names", ())
                if getattr(module, 'weight', None) is None or shared.state.interrupted or network_layer_name is None or current_names == wanted_names:
                    if task is not None:
                        pbar.update(task, advance=1)
                    continue
                backup_size += network_backup_weights(module, network_layer_name, wanted_names)
                batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name)
                if shared.opts.lora_fuse_diffusers:
                    network_apply_direct(module, batch_updown, batch_ex_bias, device=device)
                else:
                    network_apply_weights(module, batch_updown, batch_ex_bias, device=orig_device)
                if batch_updown is not None or batch_ex_bias is not None:
                    applied_layers.append(network_layer_name)
                    # module.to(device) # TODO maybe
                    if batch_updown is not None:
                        applied_weight += 1
                    if batch_ex_bias is not None:
                        applied_bias += 1
                del batch_updown, batch_ex_bias
                module.network_current_names = wanted_names
                if task is not None:
                    pbar.update(task, advance=1, description=f'networks={len(loaded_networks)} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={backup_size}')

        if task is not None and len(applied_layers) == 0:
            pbar.remove_task(task) # hide progress bar for no action
    timer.activate += time.time() - t0
    if debug and len(loaded_networks) > 0:
        shared.log.debug(f'Network load: type=LoRA networks={[n.name for n in loaded_networks]} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={backup_size} fuse={shared.opts.lora_fuse_diffusers} device={device} time={timer.summary}')
    modules.clear()
    if len(loaded_networks) > 0 and (applied_weight > 0 or applied_bias > 0):
        if shared.opts.diffusers_offload_mode == "sequential":
            sd_models.set_diffuser_offload(sd_model, op="model")


def network_deactivate(include=[], exclude=[]):
    if not shared.opts.lora_fuse_diffusers or shared.opts.lora_force_diffusers:
        return
    t0 = time.time()
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatiblility
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.disable_offload(sd_model)
        sd_models.move_model(sd_model, device=devices.cpu)
    modules = {}

    components = include if len(include) > 0 else ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'unet', 'transformer']
    components = [x for x in components if x not in exclude]
    active_components = []
    for name in components:
        component = getattr(sd_model, name, None)
        if component is not None and hasattr(component, 'named_modules'):
            modules[name] = list(component.named_modules())
            active_components.append(name)
    total = sum(len(x) for x in modules.values())
    device = devices.device if shared.opts.lora_apply_gpu else devices.cpu
    if len(previously_loaded_networks) > 0 and debug:
        pbar = rp.Progress(rp.TextColumn('[cyan]Network: type=LoRA action=deactivate'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
        task = pbar.add_task(description='', total=total)
    else:
        task = None
        pbar = nullcontext()
    with devices.inference_context(), pbar:
        applied_layers.clear()
        for component in modules.keys():
            orig_device = getattr(sd_model, component, None).device
            for _, module in modules[component]:
                network_layer_name = getattr(module, 'network_layer_name', None)
                if shared.state.interrupted or network_layer_name is None:
                    if task is not None:
                        pbar.update(task, advance=1)
                    continue
                batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, use_previous=True)
                if shared.opts.lora_fuse_diffusers:
                    network_apply_direct(module, batch_updown, batch_ex_bias, device=device, deactivate=True)
                else:
                    network_apply_weights(module, batch_updown, batch_ex_bias, device=orig_device, deactivate=True)
                if batch_updown is not None or batch_ex_bias is not None:
                    # module.to(device) # TODO maybe
                    applied_layers.append(network_layer_name)
                del batch_updown, batch_ex_bias
                module.network_current_names = ()
                if task is not None:
                    pbar.update(task, advance=1, description=f'networks={len(previously_loaded_networks)} modules={active_components} layers={total} unapply={len(applied_layers)}')

    timer.deactivate = time.time() - t0
    if debug and len(previously_loaded_networks) > 0:
        shared.log.debug(f'Network deactivate: type=LoRA networks={[n.name for n in previously_loaded_networks]} modules={active_components} layers={total} apply={len(applied_layers)} fuse={shared.opts.lora_fuse_diffusers} time={timer.summary}')
    modules.clear()
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.set_diffuser_offload(sd_model, op="model")
