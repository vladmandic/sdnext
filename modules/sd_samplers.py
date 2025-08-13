import os
import copy
from modules import shared
from modules.sd_samplers_common import samples_to_image_grid, sample_to_image # pylint: disable=unused-import


debug = shared.log.trace if os.environ.get('SD_SAMPLER_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: SAMPLER')
all_samplers = []
all_samplers_map = {}
samplers = all_samplers
samplers_for_img2img = all_samplers
samplers_map = {}
loaded_config = None


def find_sampler(name:str):
    if name is None or name == 'None':
        return all_samplers_map.get("UniPC", None)
    for sampler in all_samplers:
        if sampler.name.lower() == name.lower() or name in sampler.aliases:
            debug(f'Find sampler: name="{name}" found={sampler.name}')
            return sampler
    debug(f'Find sampler: name="{name}" found=None')
    return None


def list_samplers():
    global all_samplers # pylint: disable=global-statement
    global all_samplers_map # pylint: disable=global-statement
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    global samplers_map # pylint: disable=global-statement
    from modules import sd_samplers_diffusers
    all_samplers = [*sd_samplers_diffusers.samplers_data_diffusers]
    all_samplers_map = {x.name: x for x in all_samplers}
    samplers = all_samplers
    samplers_for_img2img = all_samplers
    samplers_map = {}
    # shared.log.debug(f'Available samplers: {[x.name for x in all_samplers]}')


def find_sampler_config(name):
    if name is not None and name != 'None':
        config = all_samplers_map.get(name, None)
    else:
        config = all_samplers[0]
    return config


def visible_sampler_names():
    visible_samplers = [x for x in all_samplers if x.name in shared.opts.show_samplers] if len(shared.opts.show_samplers) > 0 else all_samplers
    return visible_samplers


def restore_default(model):
    if model is None:
        return None
    if getattr(model, "default_scheduler", None) is not None:
        model.scheduler = copy.deepcopy(model.default_scheduler)
        if hasattr(model, "prior_pipe") and hasattr(model.prior_pipe, "scheduler"):
            model.prior_pipe.scheduler = copy.deepcopy(model.default_scheduler)
            model.prior_pipe.scheduler.config.clip_sample = False
    config = {k: v for k, v in model.scheduler.config.items() if not k.startswith('_')}
    if "flow" in model.scheduler.__class__.__name__.lower():
        shared.state.prediction_type = "flow_prediction"
    elif hasattr(model.scheduler, "config") and hasattr(model.scheduler.config, "prediction_type"):
        shared.state.prediction_type = model.scheduler.config.prediction_type
    shared.log.debug(f'Sampler: "Default" cls={model.scheduler.__class__.__name__} config={config}')
    return model.scheduler


def create_sampler(name, model):
    if name is None or name == 'None':
        return model.scheduler if model is not None else None

    # create default scheduler if it doesnt exist
    if model is not None:
        if getattr(model, "default_scheduler", None) is None:
            model.default_scheduler = copy.deepcopy(model.scheduler)
        requires_flow = ('FlowMatch' in model.default_scheduler.__class__.__name__) or (getattr(model.default_scheduler.config, 'prediction_type', None) == 'flow_prediction')
    else:
        requires_flow = False

    # restore default scheduler
    if name == 'Default' and hasattr(model, 'scheduler'):
        return restore_default(model)

    # create sampler
    config = find_sampler_config(name)
    if config is None or config.constructor is None:
        return restore_default(model)
    sampler = config.constructor(model)
    if sampler.sampler is None:
        return restore_default(model)
    is_flow = ('FlowMatch' in sampler.sampler.__class__.__name__) or (getattr(sampler.sampler.config, 'prediction_type', None) == 'flow_prediction')

    # validate sampler prediction type
    if (model is not None) and (is_flow and not requires_flow):
        shared.log.error(f'Sampler: "{sampler.name}" cls={sampler.sampler.__class__.__name__} pipe={model.__class__.__name__} model requires sampler with discrete prediction')
        return restore_default(model)
    if (model is not None) and (not is_flow and requires_flow):
        shared.log.error(f'Sampler: "{sampler.name}" cls={sampler.sampler.__class__.__name__} pipe={model.__class__.__name__} model requires sampler with flow prediction')
        return restore_default(model)

    # assign sampler
    if model is not None:
        if sampler is None or sampler.sampler is None:
            model.scheduler = copy.deepcopy(model.default_scheduler)
        else:
            model.scheduler = sampler.sampler
        if not hasattr(model, 'scheduler_config'):
            model.scheduler_config = sampler.sampler.config.copy() if hasattr(sampler, 'sampler') and hasattr(sampler.sampler, 'config') else {}
        if hasattr(model, "prior_pipe") and hasattr(model.prior_pipe, "scheduler"):
            model.prior_pipe.scheduler = sampler.sampler
            model.prior_pipe.scheduler.config.clip_sample = False
        if "flow" in model.scheduler.__class__.__name__.lower():
            shared.state.prediction_type = "flow_prediction"
        elif hasattr(model.scheduler, "config") and hasattr(model.scheduler.config, "prediction_type"):
            shared.state.prediction_type = model.scheduler.config.prediction_type
        clean_config = {k: v for k, v in model.scheduler.config.items() if not k.startswith('_') and v is not None and v is not False}
        cls = model.scheduler.__class__.__name__
    else:
        clean_config = {k: v for k, v in sampler.sampler.config.items() if not k.startswith('_') and v is not None and v is not False}
        cls = sampler.sampler.__class__.__name__
    name = sampler.name if sampler is not None and sampler.sampler is not None else 'Default'
    shared.log.debug(f'Sampler: "{name}" class={cls} config={clean_config}')
    return sampler.sampler


def set_samplers():
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    samplers = visible_sampler_names()
    # samplers_for_img2img = [x for x in samplers if x.name != "PLMS"]
    samplers_for_img2img = samplers
    samplers_map.clear()
    for sampler in all_samplers:
        samplers_map[sampler.name.lower()] = sampler.name
        for alias in sampler.aliases:
            samplers_map[alias.lower()] = sampler.name
