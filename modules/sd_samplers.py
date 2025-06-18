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
flow_models = ['Flux', 'StableDiffusion3', 'Lumina', 'AuraFlow', 'Sana', 'CogView4', 'HiDream', 'Chroma']
flow_models += ['Hunyuan', 'LTX', 'Mochi']


def list_samplers():
    global all_samplers # pylint: disable=global-statement
    global all_samplers_map # pylint: disable=global-statement
    global samplers # pylint: disable=global-statement
    global samplers_for_img2img # pylint: disable=global-statement
    global samplers_map # pylint: disable=global-statement
    if not shared.native:
        from modules import sd_samplers_compvis, sd_samplers_kdiffusion
        all_samplers = [*sd_samplers_compvis.samplers_data_compvis, *sd_samplers_kdiffusion.samplers_data_k_diffusion]
    else:
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


def create_sampler(name, model):
    if name is None or name == 'None':
        return model.scheduler if model is not None else None
    try:
        current = model.scheduler.__class__.__name__
    except Exception:
        current = None
    if name == 'Default' and hasattr(model, 'scheduler'):
        if getattr(model, "default_scheduler", None) is not None:
            model.scheduler = copy.deepcopy(model.default_scheduler)
            if hasattr(model, "prior_pipe") and hasattr(model.prior_pipe, "scheduler"):
                model.prior_pipe.scheduler = copy.deepcopy(model.default_scheduler)
                model.prior_pipe.scheduler.config.clip_sample = False
        config = {k: v for k, v in model.scheduler.config.items() if not k.startswith('_')}
        shared.log.debug(f'Sampler: "default" class={current}: {config}')
        if "flow" in model.scheduler.__class__.__name__.lower():
            shared.state.prediction_type = "flow_prediction"
        elif hasattr(model.scheduler, "config") and hasattr(model.scheduler.config, "prediction_type"):
            shared.state.prediction_type = model.scheduler.config.prediction_type
        return model.scheduler
    config = find_sampler_config(name)
    if config is None or config.constructor is None:
        # shared.log.warning(f'Sampler: sampler="{name}" not found')
        return None
    sampler = None
    if not shared.native:
        sampler = config.constructor(model)
        sampler.config = config
        sampler.name = name
        sampler.initialize(p=None)
        shared.log.debug(f'Sampler: "{name}" config={config.options}')
        return sampler
    elif shared.native:
        if 'KDiffusion' in model.__class__.__name__:
            return None
        if not any(x in model.__class__.__name__ for x in flow_models) and 'FlowMatch' in name:
            shared.log.warning(f'Sampler: default={current} target="{name}" class={model.__class__.__name__} flow-match scheduler unsupported')
            return None
        sampler = config.constructor(model)
        if sampler is None:
            sampler = config.constructor(model)
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
        if model is not None:
            clean_config = {k: v for k, v in model.scheduler.config.items() if not k.startswith('_') and v is not None and v is not False}
            cls = model.scheduler.__class__.__name__
        else:
            clean_config = {k: v for k, v in sampler.sampler.config.items() if not k.startswith('_') and v is not None and v is not False}
            cls = sampler.sampler.__class__.__name__
        name = sampler.name if sampler is not None and sampler.sampler is not None else 'Default'
        shared.log.debug(f'Sampler: "{name}" class={cls} config={clean_config}')
        return sampler.sampler
    else:
        return None


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
