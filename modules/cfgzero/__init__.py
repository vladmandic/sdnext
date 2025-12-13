# reference: <https://github.com/WeichenFan/CFG-Zero-star>

from core import MODELDATA
from modules import shared, processing, sd_models


orig_pipeline = None
supported = [
    'FluxPipeline',
    'CogView4Pipeline',
    'StableDiffusion3Pipeline',
    'HiDreamImagePipeline',
    'WanPipeline',
    'HunyuanVideoPipeline',
]


def apply(p: processing.StableDiffusionProcessing):
    if not shared.opts.cfgzero_enabled:
        return None
    cls = MODELDATA.sd_model.__class__.__name__ if MODELDATA.sd_loaded else 'None'
    if 'CFGZero' in cls:
        unapply()
    if cls not in supported:
        return None
    global orig_pipeline # pylint: disable=global-statement
    orig_pipeline = MODELDATA.sd_model

    if cls == 'FluxPipeline':
        from diffusers import pipelines
        from modules.cfgzero.flux_pipeline import FluxCFGZeroPipeline
        MODELDATA.sd_model = sd_models.switch_pipe(FluxCFGZeroPipeline, MODELDATA.sd_model)
        pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["fluxcfgzero"] = FluxCFGZeroPipeline
        pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["fluxcfgzero"] = pipelines.FluxImg2ImgPipeline
        pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["fluxcfgzero"] = pipelines.FluxInpaintPipeline
    if cls == 'CogView4Pipeline':
        from modules.cfgzero.cogview4_pipeline import CogView4CFGZeroPipeline
        MODELDATA.sd_model = sd_models.switch_pipe(CogView4CFGZeroPipeline, MODELDATA.sd_model)
    if cls == 'StableDiffusion3Pipeline':
        from modules.cfgzero.sd3_pipeline import StableDiffusion3CFGZeroPipeline
        MODELDATA.sd_model = sd_models.switch_pipe(StableDiffusion3CFGZeroPipeline, MODELDATA.sd_model)
    if cls == 'HiDreamImagePipeline':
        from modules.cfgzero.hidream_pipeline import HiDreamImageCFGZeroPipeline
        MODELDATA.sd_model = sd_models.switch_pipe(HiDreamImageCFGZeroPipeline, MODELDATA.sd_model)
    if cls == 'WanPipeline':
        from modules.cfgzero.wan_t2v_pipeline import WanCFGZeroPipeline
        MODELDATA.sd_model = sd_models.switch_pipe(WanCFGZeroPipeline, MODELDATA.sd_model)
    if cls == 'HunyuanVideoPipeline':
        from modules.cfgzero.hunyuan_t2v_pipeline import HunyuanVideoCFGZeroPipeline
        MODELDATA.sd_model = sd_models.switch_pipe(HunyuanVideoCFGZeroPipeline, MODELDATA.sd_model)

    shared.log.debug(f'Apply CFGZero: cls={cls} init={shared.opts.cfgzero_enabled} star={shared.opts.cfgzero_star} steps={shared.opts.cfgzero_steps}')
    p.task_args['use_zero_init'] = shared.opts.cfgzero_enabled
    p.task_args['use_cfg_zero_star'] = shared.opts.cfgzero_star
    p.task_args['zero_steps'] = int(shared.opts.cfgzero_steps)
    p.extra_generation_params['CFGZero'] = True


def unapply():
    global orig_pipeline # pylint: disable=global-statement
    if orig_pipeline is not None:
        MODELDATA.sd_model = orig_pipeline
        orig_pipeline = None
    return MODELDATA.sd_model.__class__
