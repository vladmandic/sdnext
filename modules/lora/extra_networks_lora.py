from typing import List
import os
import re
import numpy as np
from modules.lora import networks, network_overrides
from modules import extra_networks, shared, sd_models


debug = os.environ.get('SD_SCRIPT_DEBUG', None) is not None
debug_log = shared.log.trace if debug else lambda *args, **kwargs: None


def get_stepwise(param, step, steps): # from https://github.com/cheald/sd-webui-loractl/blob/master/loractl/lib/utils.py
    def sorted_positions(raw_steps):
        steps = [[float(s.strip()) for s in re.split("[@~]", x)]
                 for x in re.split("[,;]", str(raw_steps))]
        if len(steps[0]) == 1: # If we just got a single number, just return it
            return steps[0][0]
        steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps] # Add implicit 1s to any steps which don't have a weight
        steps.sort(key=lambda k: k[1]) # Sort by index
        steps = [list(v) for v in zip(*steps)]
        return steps

    def calculate_weight(m, step, max_steps, step_offset=2):
        if isinstance(m, list):
            if m[1][-1] <= 1.0:
                step = step / (max_steps - step_offset) if max_steps > 0 else 1.0
            v = np.interp(step, m[1], m[0])
            return v
        else:
            return m

    stepwise = calculate_weight(sorted_positions(param), step, steps)
    return stepwise


def prompt(p):
    if shared.opts.lora_apply_tags == 0:
        return
    all_tags = []
    for loaded in networks.loaded_networks:
        page = [en for en in shared.extra_networks if en.name == 'lora'][0]
        item = page.create_item(loaded.name)
        tags = (item or {}).get("tags", {})
        loaded.tags = list(tags)
        if len(loaded.tags) == 0:
            loaded.tags.append(loaded.name)
        if shared.opts.lora_apply_tags > 0:
            loaded.tags = loaded.tags[:shared.opts.lora_apply_tags]
        all_tags.extend(loaded.tags)
    if len(all_tags) > 0:
        all_tags = list(set(all_tags))
        all_tags = [t for t in all_tags if t not in p.prompt]
        if len(all_tags) > 0:
            shared.log.debug(f"Network load: type=LoRA tags={all_tags} max={shared.opts.lora_apply_tags} apply")
        all_tags = ', '.join(all_tags)
        p.extra_generation_params["LoRA tags"] = all_tags
        if '_tags_' in p.prompt:
            p.prompt = p.prompt.replace('_tags_', all_tags)
        else:
            p.prompt = f"{p.prompt}, {all_tags}"
        if p.all_prompts is not None:
            for i in range(len(p.all_prompts)):
                if '_tags_' in p.all_prompts[i]:
                    p.all_prompts[i] = p.all_prompts[i].replace('_tags_', all_tags)
                else:
                    p.all_prompts[i] = f"{p.all_prompts[i]}, {all_tags}"


def infotext(p):
    names = [i.name for i in networks.loaded_networks]
    if len(names) > 0:
        p.extra_generation_params["LoRA networks"] = ", ".join(names)
    if shared.opts.lora_add_hashes_to_infotext:
        network_hashes = []
        for item in networks.loaded_networks:
            if not item.network_on_disk.shorthash:
                continue
            network_hashes.append(item.network_on_disk.shorthash)
        if len(network_hashes) > 0:
            p.extra_generation_params["LoRA hashes"] = ", ".join(network_hashes)


def parse(p, params_list, step=0):
    names = []
    te_multipliers = []
    unet_multipliers = []
    dyn_dims = []
    for params in params_list:
        assert params.items
        names.append(params.positional[0])
        te_multiplier = params.named.get("te", params.positional[1] if len(params.positional) > 1 else shared.opts.extra_networks_default_multiplier)
        if isinstance(te_multiplier, str) and "@" in te_multiplier:
            te_multiplier = get_stepwise(te_multiplier, step, p.steps)
        else:
            te_multiplier = float(te_multiplier)
        unet_multiplier = [params.positional[2] if len(params.positional) > 2 else te_multiplier] * 3
        unet_multiplier = [params.named.get("unet", unet_multiplier[0])] * 3
        unet_multiplier[0] = params.named.get("in", unet_multiplier[0])
        unet_multiplier[1] = params.named.get("mid", unet_multiplier[1])
        unet_multiplier[2] = params.named.get("out", unet_multiplier[2])
        for i in range(len(unet_multiplier)):
            if isinstance(unet_multiplier[i], str) and "@" in unet_multiplier[i]:
                unet_multiplier[i] = get_stepwise(unet_multiplier[i], step, p.steps)
            else:
                unet_multiplier[i] = float(unet_multiplier[i])
        dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
        dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
        te_multipliers.append(te_multiplier)
        unet_multipliers.append(unet_multiplier)
        dyn_dims.append(dyn_dim)
    return names, te_multipliers, unet_multipliers, dyn_dims


class ExtraNetworkLora(extra_networks.ExtraNetwork):

    def __init__(self):
        super().__init__('lora')
        self.active = False
        self.model = None
        self.errors = {}

    def signature(self, names: List[str], te_multipliers: List, unet_multipliers: List):
        return [f'{name}:{te}:{unet}' for name, te, unet in zip(names, te_multipliers, unet_multipliers)]

    def changed(self, requested: List[str], include: List[str], exclude: List[str]):
        sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)
        if not hasattr(sd_model, 'loaded_loras'):
            sd_model.loaded_loras = {}
        key = f'{",".join(include)}:{",".join(exclude)}'
        loaded = sd_model.loaded_loras.get(key, [])
        # shared.log.trace(f'Network load: type=LoRA key="{key}" requested={requested} loaded={loaded}')
        if (len(requested) == 0) or (len(requested) != len(loaded)):
            sd_model.loaded_loras[key] = requested
            return True
        for r, l in zip(requested, loaded):
            if r != l:
                sd_model.loaded_loras[key] = requested
                return True
        return False

    def activate(self, p, params_list, step=0, include=[], exclude=[]):
        self.errors.clear()
        if self.active:
            if self.model != shared.opts.sd_model_checkpoint: # reset if model changed
                self.active = False
        if len(params_list) > 0 and not self.active: # activate patches once
            self.active = True
            self.model = shared.opts.sd_model_checkpoint
        names, te_multipliers, unet_multipliers, dyn_dims = parse(p, params_list, step)
        requested = self.signature(names, te_multipliers, unet_multipliers)

        if debug:
            import sys
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            debug_log(f'Network load: type=LoRA include={include} exclude={exclude} requested={requested} fn={fn}')

        force_diffusers = network_overrides.check_override()
        if force_diffusers:
            has_changed = False # diffusers handle their own loading
            if len(exclude) == 0:
                networks.network_load(names, te_multipliers, unet_multipliers, dyn_dims) # load only on first call
        else:
            networks.network_load(names, te_multipliers, unet_multipliers, dyn_dims) # load
            has_changed = self.changed(requested, include, exclude)
            if has_changed:
                networks.network_deactivate(include, exclude)
                networks.network_activate(include, exclude)
                shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model) # TODO lora: required for flux to reapply offload after lora has been applied, but fails with oom
                debug_log(f'Network load: type=LoRA previous={[n.name for n in networks.previously_loaded_networks]} current={[n.name for n in networks.loaded_networks]} changed')

        if len(networks.loaded_networks) > 0 and (len(networks.applied_layers) > 0 or force_diffusers) and step == 0:
            infotext(p)
            prompt(p)
            if (has_changed or force_diffusers) and len(include) == 0: # print only once
                shared.log.info(f'Network load: type=LoRA apply={[n.name for n in networks.loaded_networks]} mode={"fuse" if shared.opts.lora_fuse_diffusers else "backup"} te={te_multipliers} unet={unet_multipliers} time={networks.timer.summary}')

    def deactivate(self, p):
        if shared.native:
            networks.previously_loaded_networks = networks.loaded_networks.copy()
            debug_log(f'Network load: type=LoRA active={[n.name for n in networks.previously_loaded_networks]} deactivate')
        if shared.native and len(networks.diffuser_loaded) > 0:
            if not (shared.compiled_model_state is not None and shared.compiled_model_state.is_compiled is True):
                if hasattr(shared.sd_model, "unfuse_lora"):
                    try:
                        shared.sd_model.unfuse_lora()
                    except Exception:
                        pass
                if hasattr(shared.sd_model, "unload_lora_weights"):
                    try:
                        shared.sd_model.unload_lora_weights() # fails for non-CLIP models
                    except Exception:
                        pass
        if self.active and networks.debug:
            shared.log.debug(f"Network end: type=LoRA time={networks.timer.summary}")
        if self.errors:
            for k, v in self.errors.items():
                shared.log.error(f'LoRA: name="{k}" errors={v}')
            self.errors.clear()
