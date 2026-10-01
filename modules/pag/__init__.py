from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionXLPipeline # pylint: disable=unused-import
from modules import shared, processing, sd_models
from modules.logger import log
from modules.pag.pipe_sd import StableDiffusionPAGPipeline
from modules.pag.pipe_sdxl import StableDiffusionXLPAGPipeline
from modules.control.units import detect


orig_pipeline = None


def apply(p: processing.StableDiffusionProcessing): # pylint: disable=arguments-differ
    global orig_pipeline # pylint: disable=global-statement
    if shared.sd_loaded and shared.sd_model.__class__ in (StableDiffusionPAGPipeline, StableDiffusionXLPAGPipeline):
        unapply()
    if p.cfg_true is None or p.cfg_true <= 0: # -1 is the ui default and leaves the pipeline alone
        return
    model = shared.sd_model if shared.sd_loaded else None
    if model is None:
        return
    cls = model.__class__
    if 'PAG' in cls.__name__:
        pass
    elif detect.is_sdxl(model): # before is_sd15, whose prefix also matches the sdxl and sd3 class names
        if sd_models.get_diffusers_task(model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE:
            log.warning(f'PAG: pipeline={cls.__name__} not implemented')
            return
        orig_pipeline = model
        shared.sd_model = sd_models.switch_pipe(StableDiffusionXLPAGPipeline, model)
    elif detect.is_sd15(model) and not detect.is_compatible(model, pattern='StableDiffusion3'):
        if sd_models.get_diffusers_task(model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE:
            log.warning(f'PAG: pipeline={cls.__name__} not implemented')
            return
        orig_pipeline = model
        shared.sd_model = sd_models.switch_pipe(StableDiffusionPAGPipeline, model)
    elif detect.is_f1(model):
        p.task_args['true_cfg_scale'] = p.cfg_true
    else:
        return

    p.task_args['cfg_true'] = p.cfg_true
    p.task_args['cfg_adaptive_scaling'] = p.cfg_adaptive
    p.task_args['cfg_adaptive_scale'] = p.cfg_adaptive
    pag_applied_layers = shared.opts.pag_apply_layers
    pag_applied_layers_index = pag_applied_layers.split() if len(pag_applied_layers) > 0 else []
    pag_applied_layers_index = [p.strip() for p in pag_applied_layers_index]
    p.task_args['pag_applied_layers_index'] = pag_applied_layers_index if len(pag_applied_layers_index) > 0 else ['m0'] # Available layers: d[0-5], m[0], u[0-8]
    p.extra_generation_params["CFG true"] = p.cfg_true
    p.extra_generation_params["CFG adaptive"] = p.cfg_adaptive
    # log.debug(f'{c}: args={p.task_args}')


def unapply():
    global orig_pipeline # pylint: disable=global-statement
    if orig_pipeline is not None:
        shared.sd_model = orig_pipeline
        orig_pipeline = None
    return shared.sd_model.__class__
