import time
from modules import shared, errors
from modules.logger import log
from modules.lora import lora_load, lora_common


previously_loaded = [] # we maintain private state here


def load_nunchaku(names, strengths):
    global previously_loaded # pylint: disable=global-statement
    strengths = [s[0] if isinstance(s, list) else s for s in strengths]
    networks = lora_load.gather_networks(names)
    networks = [(network, strength) for network, strength in zip(networks, strengths, strict=False) if network is not None and strength > 0]
    loras = [(network.filename, strength) for network, strength in networks]
    is_changed = loras != previously_loaded
    if not is_changed:
        return False
    if not hasattr(shared.sd_model, 'transformer') or not hasattr(shared.sd_model.transformer, 'update_lora_params'):
        log.error(f'Network load: type=LoRA method=nunchaku model={shared.sd_model.__class__.__name__} unsupported')
        return False

    previously_loaded = loras
    try:
        t0 = time.time()
        from nunchaku.lora.flux.compose import compose_lora
        composed_lora = compose_lora(loras)
        shared.sd_model.transformer.update_lora_params(composed_lora)
        lora_common.loaded_networks = [n[0] for n in networks] # used by infotext
        t1 = time.time()
        lora_common.timer.load = t1 - t0
        log.debug(f"Network load: type=LoRA method=nunchaku loras={names} strength={strengths} time={t1-t0:.3f}")
    except Exception as e:
        log.error(f'Network load: type=LoRA method=nunchaku {e}')
        if lora_common.debug:
            errors.display(e, 'LoRA')
    return is_changed
