from diffusers.pipelines import FluxPipeline
 # pylint: disable=unused-import
from modules import shared, processing, sd_models
from modules.cfgzero.flux_pipeline import FluxCFGZeroPipeline


orig_pipeline = None


def apply(p: processing.StableDiffusionProcessing): # pylint: disable=arguments-differ
    if not shared.native:
        return None
    cls = shared.sd_model.__class__ if shared.sd_loaded else None
    if cls == FluxPipeline and shared.opts.cfgzero_enabled:
        global orig_pipeline # pylint: disable=global-statement
        orig_pipeline = shared.sd_model
        shared.sd_model = sd_models.switch_pipe(FluxCFGZeroPipeline, shared.sd_model)
        p.task_args['use_zero_init'] = True
        p.extra_generation_params["CFGZero"] = True


def unapply():
    global orig_pipeline # pylint: disable=global-statement
    if orig_pipeline is not None:
        shared.sd_model = orig_pipeline
        orig_pipeline = None
    return shared.sd_model.__class__
