# reference: <https://github.com/WeichenFan/CFG-Zero-star>

from modules import shared, processing, sd_models
from modules.logger import log


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
    cls = shared.sd_model.__class__.__name__ if shared.sd_loaded else 'None'
    if 'CFGZero' in cls:
        unapply()
    if cls not in supported:
        return None
    global orig_pipeline # pylint: disable=global-statement
    orig_pipeline = shared.sd_model

    if cls == 'FluxPipeline':
        from diffusers import pipelines
        from modules.cfgzero.flux_pipeline import FluxCFGZeroPipeline
        shared.sd_model = sd_models.switch_pipe(FluxCFGZeroPipeline, shared.sd_model)
        pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["fluxcfgzero"] = FluxCFGZeroPipeline
        pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["fluxcfgzero"] = pipelines.FluxImg2ImgPipeline
        pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING["fluxcfgzero"] = pipelines.FluxInpaintPipeline
    if cls == 'CogView4Pipeline':
        from modules.cfgzero.cogview4_pipeline import CogView4CFGZeroPipeline
        shared.sd_model = sd_models.switch_pipe(CogView4CFGZeroPipeline, shared.sd_model)
    if cls == 'StableDiffusion3Pipeline':
        from modules.cfgzero.sd3_pipeline import StableDiffusion3CFGZeroPipeline
        shared.sd_model = sd_models.switch_pipe(StableDiffusion3CFGZeroPipeline, shared.sd_model)
    if cls == 'HiDreamImagePipeline':
        from modules.cfgzero.hidream_pipeline import HiDreamImageCFGZeroPipeline
        shared.sd_model = sd_models.switch_pipe(HiDreamImageCFGZeroPipeline, shared.sd_model)
    if cls == 'WanPipeline':
        from modules.cfgzero.wan_t2v_pipeline import WanCFGZeroPipeline
        shared.sd_model = sd_models.switch_pipe(WanCFGZeroPipeline, shared.sd_model)
    if cls == 'HunyuanVideoPipeline':
        from modules.cfgzero.hunyuan_t2v_pipeline import HunyuanVideoCFGZeroPipeline
        shared.sd_model = sd_models.switch_pipe(HunyuanVideoCFGZeroPipeline, shared.sd_model)

    log.debug(f'Apply CFGZero: cls={cls} init={shared.opts.cfgzero_enabled} star={shared.opts.cfgzero_star} steps={shared.opts.cfgzero_steps}')
    p.task_args['use_zero_init'] = shared.opts.cfgzero_enabled
    p.task_args['use_cfg_zero_star'] = shared.opts.cfgzero_star
    p.task_args['zero_steps'] = int(shared.opts.cfgzero_steps)
    p.extra_generation_params['CFGZero'] = True


def unapply():
    global orig_pipeline # pylint: disable=global-statement
    if orig_pipeline is not None:
        shared.sd_model = orig_pipeline
        orig_pipeline = None
    return shared.sd_model.__class__
