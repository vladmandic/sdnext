import os
import time
import diffusers
from modules import shared, errors
from modules.logger import log
from modules.lora import network
from modules.lora import lora_common as l


diffuser_loaded = []
diffuser_scales = []


def load_per_module(sd_model: diffusers.DiffusionPipeline, filename: str, adapter_name: str, lora_modules: list[str]):
    log.debug(f'LoRA load: modules={lora_modules}')
    try:
        state_dict = sd_model.lora_state_dict(filename)
        if isinstance(state_dict, tuple) and len(state_dict) == 2:
            state_dict, network_alphas = state_dict
        else:
            network_alphas = {}
    except Exception as e:
        log.error(f'LoRA load: {e}')
        if l.debug:
            errors.display(e, "LoRA")
        return None
    for lora_module in lora_modules:
        if lora_module == 'transformer':
            if hasattr(sd_model, 'transformer') and sd_model.transformer is not None:
                sd_model.load_lora_into_transformer(state_dict, transformer=sd_model.transformer, adapter_name=adapter_name)
            else:
                log.warning(f'LoRA load: requested={lora_module} missing')
        elif lora_module == 'transformer_2':
            if hasattr(sd_model, 'transformer_2') and sd_model.transformer_2 is not None:
                sd_model.load_lora_into_transformer(state_dict, transformer=sd_model.transformer_2, adapter_name=adapter_name)
            else:
                log.warning(f'LoRA load: requested={lora_module} missing')
        elif lora_module == 'unet':
            if hasattr(sd_model, 'unet') and sd_model.unet is not None:
                sd_model.load_lora_into_unet(state_dict, network_alphas, unet=sd_model.unet, adapter_name=adapter_name)
            else:
                log.warning(f'LoRA load: requested={lora_module} missing')
        elif lora_module == 'text_encoder' or lora_module == 'te':
            if hasattr(sd_model, 'text_encoder') and sd_model.text_encoder is not None:
                sd_model.load_lora_into_text_encoder(state_dict, network_alphas, text_encoder=sd_model.text_encoder, adapter_name=adapter_name)
            else:
                log.warning(f'LoRA load: requested={lora_module} missing')
        else:
            log.warning(f'LoRA load: requested={lora_module} unknown')
    return adapter_name


def load_diffusers(name: str, network_on_disk: network.NetworkOnDisk, lora_scale:float=shared.opts.extra_networks_default_multiplier, lora_module=None) -> network.Network | None:
    t0 = time.time()
    name = name.replace(".", "_")
    sd_model: diffusers.DiffusionPipeline = getattr(shared.sd_model, "pipe", shared.sd_model)
    log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" detected={network_on_disk.sd_version} method=diffusers scale={lora_scale} fuse={shared.opts.lora_fuse_native}:{shared.opts.lora_fuse_diffusers}')
    if not hasattr(sd_model, 'load_lora_weights'):
        log.error(f'Network load: type=LoRA class={sd_model.__class__} does not implement load lora')
        return None
    try:
        if lora_module is not None and isinstance(lora_module, list) and len(lora_module) > 0:
            name = load_per_module(sd_model, network_on_disk.filename, adapter_name=name, lora_modules=lora_module)
            sd_model._lora_partial = True # pylint: disable=protected-access
        else:
            sd_model.load_lora_weights(network_on_disk.filename, adapter_name=name)
    except Exception as e:
        if 'already in use' in str(e):
            pass
        else:
            if 'following keys have not been correctly renamed' in str(e):
                log.error(f'Network load: type=LoRA name="{name}" diffusers unsupported format')
            elif 'object has no attribute' in str(e):
                log.error(f'Network load: type=LoRA name="{name}" diffusers empty module')
            else:
                log.error(f'Network load: type=LoRA name="{name}" {e}')
            if l.debug:
                errors.display(e, "LoRA")
            return None
    if name is None:
        return None
    if name not in diffuser_loaded:
        list_adapters = sd_model.get_list_adapters()
        list_adapters = [adapter for adapters in list_adapters.values() for adapter in adapters]
        if name not in list_adapters:
            log.error(f'Network load: type=LoRA name="{name}" adapters={list_adapters} not loaded')
        else:
            diffuser_loaded.append(name)
            diffuser_scales.append(lora_scale)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    l.timer.activate += time.time() - t0
    return net
