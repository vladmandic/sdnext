import re
import inspect
from collections import defaultdict
from modules import errors, shared


extra_network_registry = {}


def initialize():
    extra_network_registry.clear()


def register_extra_network(extra_network):
    extra_network_registry[extra_network.name] = extra_network


def register_default_extra_networks():
    from modules.ui_extra_networks_styles import ExtraNetworkStyles
    register_extra_network(ExtraNetworkStyles())

    from modules.lora import lora_common, extra_networks_lora
    lora_common.extra_network_lora = extra_networks_lora.ExtraNetworkLora()
    register_extra_network(lora_common.extra_network_lora)


class ExtraNetworkParams:
    def __init__(self, items=None):
        self.items = items or []
        self.positional = []
        self.named = {}
        for item in self.items:
            parts = item.split('=', 2) if isinstance(item, str) else [item]
            if len(parts) == 2:
                self.named[parts[0]] = parts[1]
            else:
                self.positional.append(item)


class ExtraNetwork:
    def __init__(self, name):
        self.name = name

    def activate(self, p, params_list):
        """
        Called by processing on every run. Whatever the extra network is meant to do should be activated here. Passes arguments related to this extra network in params_list. User passes arguments by specifying this in his prompt:
        <name:arg1:arg2:arg3>
        Where name matches the name of this ExtraNetwork object, and arg1:arg2:arg3 are any natural number of text arguments separated by colon.
        Even if the user does not mention this ExtraNetwork in his prompt, the call will stil be made, with empty params_list - in this case, all effects of this extra networks should be disabled.
        Can be called multiple times before deactivate() - each new call should override the previous call completely.
        For example, if this ExtraNetwork's name is 'hypernet' and user's prompt is:
        > "1girl, <hypernet:agm:1.1> <extrasupernet:master:12:13:14> <hypernet:ray>"
        params_list will be:
        [
            ExtraNetworkParams(items=["agm", "1.1"]),
            ExtraNetworkParams(items=["ray"])
        ]
        """
        raise NotImplementedError

    def deactivate(self, p):
        """
        Called at the end of processing for housekeeping. No need to do anything here.
        """
        raise NotImplementedError


def is_stepwise(en_obj):
    all_args = []
    for en in en_obj:
        all_args.extend(en.positional[1:])
        all_args.extend(en.named.values())
    return any([len(str(x).split("@")) > 1 for x in all_args]) # noqa C419 # pylint: disable=use-a-generator


def activate(p, extra_network_data=None, step=0, include=[], exclude=[]):
    """call activate for extra networks in extra_network_data in specified order, then call activate for all remaining registered networks with an empty argument list"""
    if p.disable_extra_networks:
        return
    extra_network_data = extra_network_data or p.network_data
    # if extra_network_data is None or len(extra_network_data) == 0:
        # return
    stepwise = False
    for extra_network_args in extra_network_data.values():
        stepwise = stepwise or is_stepwise(extra_network_args)
    functional = shared.opts.lora_functional
    if shared.opts.lora_force_diffusers and stepwise:
        shared.log.warning("Network load: type=LoRA method=composable loader=diffusers not compatible")
        stepwise = False
    shared.opts.data['lora_functional'] = stepwise or functional

    for extra_network_name, extra_network_args in extra_network_data.items():
        extra_network = extra_network_registry.get(extra_network_name, None)
        if extra_network is None:
            errors.log.warning(f"Skipping unknown extra network: {extra_network_name}")
            continue
        try:
            signature = list(inspect.signature(extra_network.activate).parameters)
            if 'include' in signature and 'exclude' in signature:
                extra_network.activate(p, extra_network_args, step=step, include=include, exclude=exclude)
            else:
                extra_network.activate(p, extra_network_args, step=step)
        except Exception as e:
            errors.display(e, f"Activating network: type={extra_network_name} args:{extra_network_args}")

    for extra_network_name, extra_network in extra_network_registry.items():
        args = extra_network_data.get(extra_network_name, None)
        if args is not None:
            continue
        try:
            # extra_network.activate(p, [])
            signature = list(inspect.signature(extra_network.activate).parameters)
            if 'include' in signature and 'exclude' in signature:
                extra_network.activate(p, [], include=include, exclude=exclude)
            else:
                extra_network.activate(p, [])
        except Exception as e:
            errors.display(e, f"Activating network: type={extra_network_name}")

    p.network_data = extra_network_data
    if stepwise:
        p.stepwise_lora = True
        shared.opts.data['lora_functional'] = functional


def deactivate(p, extra_network_data=None):
    """call deactivate for extra networks in extra_network_data in specified order, then call deactivate for all remaining registered networks"""
    if p.disable_extra_networks:
        return
    extra_network_data = extra_network_data or p.network_data
    # if extra_network_data is None or len(extra_network_data) == 0:
    #    return
    for extra_network_name in extra_network_data:
        extra_network = extra_network_registry.get(extra_network_name, None)
        if extra_network is None:
            continue
        try:
            extra_network.deactivate(p)
        except Exception as e:
            errors.display(e, f"deactivating extra network {extra_network_name}")

    for extra_network_name, extra_network in extra_network_registry.items():
        args = extra_network_data.get(extra_network_name, None)
        if args is not None:
            continue
        try:
            extra_network.deactivate(p)
        except Exception as e:
            errors.display(e, f"deactivating unmentioned extra network {extra_network_name}")


re_extra_net = re.compile(r"<(\w+):([^>]+)>")


def parse_prompt(prompt):
    res = defaultdict(list)

    def found(m):
        name = m.group(1)
        args = m.group(2)
        res[name].append(ExtraNetworkParams(items=args.split(":")))
        return ""
    if isinstance(prompt, list):
        prompt = [re.sub(re_extra_net, found, p) for p in prompt]
    else:
        prompt = re.sub(re_extra_net, found, prompt)
    return prompt, res


def parse_prompts(prompts):
    res = []
    extra_data = None

    for prompt in prompts:
        updated_prompt, parsed_extra_data = parse_prompt(prompt)
        if extra_data is None:
            extra_data = parsed_extra_data
        res.append(updated_prompt)

    return res, extra_data
