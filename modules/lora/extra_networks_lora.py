import os
import re
import numpy as np
from modules.lora import networks, lora_overrides, lora_load, lora_diffusers
from modules.lora import lora_common as l
from modules import extra_networks, shared, sd_models
from modules.logger import log


debug = os.environ.get('SD_LORA_DEBUG', None) is not None
debug_log = log.trace if debug else lambda *args, **kwargs: None


def get_stepwise(param, step, steps): # from https://github.com/cheald/sd-webui-loractl/blob/master/loractl/lib/utils.py
    def sorted_positions(raw_steps):
        steps = [[float(s.strip()) for s in re.split("[@~]", x)]
                 for x in re.split("[,;]", str(raw_steps))]
        if len(steps[0]) == 1: # If we just got a single number, just return it
            return steps[0][0]
        steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps] # Add implicit 1s to any steps which don't have a weight
        steps.sort(key=lambda k: k[1]) # Sort by index
        steps = [list(v) for v in zip(*steps, strict=False)]
        return steps

    def calculate_weight(m, step, max_steps, step_offset=2):
        if isinstance(m, list):
            if m[1][-1] <= 1.0:
                step = step / (max_steps - step_offset) if max_steps > 0 else 1.0
            v = np.interp(step, m[1], m[0])
            debug_log(f"Network load: type=LoRA step={step} steps={max_steps} v={v}")
            return v
        else:
            return m

    stepwise = calculate_weight(sorted_positions(param), step, steps)
    return stepwise


def prompt(p):
    if shared.opts.lora_apply_tags == 0:
        return
    all_tags = []
    for loaded in l.loaded_networks:
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
            log.debug(f"Network load: type=LoRA tags={all_tags} max={shared.opts.lora_apply_tags} apply")
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
    names = [i.name for i in l.loaded_networks]
    if len(names) > 0:
        p.extra_generation_params["LoRA networks"] = ", ".join(names)
    if shared.opts.lora_add_hashes_to_infotext:
        network_hashes = []
        for item in l.loaded_networks:
            if not item.network_on_disk.shorthash:
                continue
            network_hashes.append(item.network_on_disk.shorthash)
        if len(network_hashes) > 0:
            p.extra_generation_params["LoRA hashes"] = ", ".join(network_hashes)


def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def parse(p, params_list, step=0):
    names = []
    te_multipliers = []
    unet_multipliers = []
    dyn_dims = []
    lora_modules = []
    for params in params_list:
        name = params.positional[0]

        default_multiplier = params.positional[1] if len(params.positional) > 1 else shared.opts.extra_networks_default_multiplier
        default_multiplier = to_float(default_multiplier)
        if isinstance(default_multiplier, str) and "@" not in default_multiplier:
            default_multiplier = shared.opts.extra_networks_default_multiplier

        te_multiplier = params.named.get("te", default_multiplier)
        if isinstance(te_multiplier, str) and "@" in te_multiplier:
            te_multiplier = get_stepwise(te_multiplier, step, p.steps)
        else:
            te_multiplier = to_float(te_multiplier)

        unet_multiplier = 3 * [params.named.get("unet", te_multiplier)] # fill all 3 with same value
        unet_multiplier[0] = params.named.get("in", unet_multiplier[0])
        unet_multiplier[1] = params.named.get("mid", unet_multiplier[1])
        unet_multiplier[2] = params.named.get("out", unet_multiplier[2])
        for i in range(len(unet_multiplier)):
            if isinstance(unet_multiplier[i], str) and "@" in unet_multiplier[i]:
                unet_multiplier[i] = get_stepwise(unet_multiplier[i], step, p.steps)
            else:
                unet_multiplier[i] = to_float(unet_multiplier[i])

        dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else None

        if (te_multiplier == 0) and all(u == 0 for u in unet_multiplier): # skip lora with strength zero
            continue

        names.append(name)
        te_multipliers.append(te_multiplier)
        unet_multipliers.append(unet_multiplier)
        dyn_dims.append(dyn_dim)

        lora_module = []
        if 'high' in params.positional or 'HIGH 14B' in params.positional[0]:
            lora_module.append('transformer')
        if 'low' in params.positional or 'LOW 14B' in params.positional[0]:
            lora_module.append('transformer_2')
        if params.named.get('module', None) is not None:
            lora_module.append(params.named['module'].lower())

        if len(lora_module) == 0 and shared.sd_loaded:
            if hasattr(shared.sd_model, 'transformer') and (shared.sd_model.transformer is not None) and hasattr(shared.sd_model, 'transformer_2') and (shared.sd_model.transformer_2 is None):
                lora_module.append('transformer')
            if hasattr(shared.sd_model, 'transformer') and (shared.sd_model.transformer is None) and hasattr(shared.sd_model, 'transformer_2') and (shared.sd_model.transformer_2 is not None):
                lora_module.append('transformer_2')

        lora_modules.append(lora_module)

    return names, te_multipliers, unet_multipliers, dyn_dims, lora_modules


def unload_diffusers():
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


class ExtraNetworkLora(extra_networks.ExtraNetwork):

    def __init__(self):
        super().__init__('lora')
        self.active = False
        self.model = None
        self.errors = {}

    def signature(self, names: list[str], te_multipliers: list, unet_multipliers: list):
        return [f'{name}:{te}:{unet}' for name, te, unet in zip(names, te_multipliers, unet_multipliers, strict=False)]

    def changed(self, requested: list[str], include: list[str] = None, exclude: list[str] = None) -> bool:
        if shared.opts.lora_force_reload:
            debug_log(f'Network check: type=LoRA requested={requested} status=forced')
            return True
        sd_model = shared.sd_model.pipe if hasattr(shared.sd_model, 'pipe') else shared.sd_model
        if sd_model is None:
            return False
        if not hasattr(sd_model, 'loaded_loras'):
            sd_model.loaded_loras = {}
        if include is None or len(include) == 0:
            include = ['all']
        if exclude is None or len(exclude) == 0:
            exclude = ['none']
        key = f'include={",".join(include)}:exclude={",".join(exclude)}'
        loaded = sd_model.loaded_loras.get(key, [])
        if len(requested) != len(loaded):
            sd_model.loaded_loras[key] = requested
            debug_log(f'Network check: type=LoRA key="{key}" requested={requested} loaded={loaded} status=changed')
            return True
        for req, load in zip(requested, loaded, strict=False):
            if req != load:
                sd_model.loaded_loras[key] = requested
                debug_log(f'Network check: type=LoRA key="{key}" requested={requested} loaded={loaded} status=changed')
                return True
        debug_log(f'Network check: type=LoRA key="{key}" requested={requested} loaded={loaded} status=same')
        return False

    def activate(self, p, params_list, step=0, include=None, exclude=None):
        if exclude is None:
            exclude = []
        if include is None:
            include = []
        self.errors.clear()
        if self.active:
            if self.model != shared.opts.sd_model_checkpoint: # reset if model changed
                self.active = False
        if len(params_list) > 0 and not self.active: # activate patches once
            self.active = True
            self.model = shared.opts.sd_model_checkpoint
        names, te_multipliers, unet_multipliers, dyn_dims, lora_modules = parse(p, params_list, step)
        requested = self.signature(names, te_multipliers, unet_multipliers)

        load_method = lora_overrides.get_method()
        if debug:
            import sys
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            debug_log(f'Network load: type=LoRA include={include} exclude={exclude} method={load_method} requested={requested} fn={fn}')

        if load_method == 'diffusers':
            has_changed = self.changed(requested)
            if has_changed:
                jobid = shared.state.begin('LoRA')
                lora_load.network_load(names, te_multipliers, unet_multipliers, dyn_dims, lora_modules) # load only on first call
                sd_models.set_diffuser_offload(shared.sd_model, op="model")
                shared.state.end(jobid)

        elif load_method == 'nunchaku':
            from modules.lora import lora_nunchaku
            has_changed = lora_nunchaku.load_nunchaku(names, unet_multipliers)

        else: # native
            lora_load.network_load(names, te_multipliers, unet_multipliers, dyn_dims) # load
            has_changed = self.changed(requested, include, exclude)
            if has_changed:
                jobid = shared.state.begin('LoRA')
                if len(l.previously_loaded_networks) > 0:
                    log.info(f'Network unload: type=LoRA networks={[n.name for n in l.previously_loaded_networks]} mode={"fuse" if shared.opts.lora_fuse_native else "backup"}')
                    networks.network_deactivate(include, exclude)
                networks.network_activate(include, exclude)
                debug_log(f'Network change: type=LoRA previous={[n.name for n in l.previously_loaded_networks]} current={[n.name for n in l.loaded_networks]}')
                if len(include) == 0:
                    l.previously_loaded_networks = l.loaded_networks.copy()
                shared.state.end(jobid)

        if len(l.loaded_networks) > 0 and (len(networks.applied_layers) > 0 or load_method=='diffusers' or load_method=='nunchaku') and step == 0:
            infotext(p)
            prompt(p)
            if has_changed and len(include) == 0: # print only once
                log.info(f'Network load: type=LoRA networks={[n.name for n in l.loaded_networks]} method={load_method} mode={"fuse" if shared.opts.lora_fuse_native else "backup"} te={te_multipliers} unet={unet_multipliers} time={l.timer.summary}')

    def deactivate(self, p, force=False):
        if len(lora_diffusers.diffuser_loaded) > 0 and (shared.opts.lora_force_reload or force):
            unload_diffusers()
        if force:
            networks.network_deactivate()
        if self.active and l.debug:
            log.debug(f"Network end: type=LoRA time={l.timer.summary}")
        if self.errors:
            for k, v in self.errors.items():
                log.error(f'Network: type=LoRA name="{k}" errors={v}')
            self.errors.clear()
