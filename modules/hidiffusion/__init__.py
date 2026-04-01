# Original: https://github.com/megvii-research/HiDiffusion

import time
from modules import shared
from modules.logger import log
from modules.hidiffusion import hidiffusion


def apply(p, model_type):
    if model_type not in ['sd', 'sdxl'] and p.hidiffusion:
        log.warning(f'HiDiffusion: class={shared.sd_model.__class__.__name__} not supported')
        return
    unapply()
    pipe = shared.sd_model.pipe if hasattr(shared.sd_model, 'pipe') else shared.sd_model
    if getattr(p, 'hidiffusion', False) is True:
        t0 = time.time()
        hidiffusion.is_aggressive_raunet = shared.opts.hidiffusion_steps > 0
        hidiffusion.aggressive_step = shared.opts.hidiffusion_steps
        if shared.opts.hidiffusion_t1 >= 0:
            t1 = shared.opts.hidiffusion_t1
            hidiffusion.switching_threshold_ratio_dict['sd15_1024']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sd15_2048']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sdxl_2048']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sdxl_4096']['T1_ratio'] = t1
            hidiffusion.switching_threshold_ratio_dict['sdxl_turbo_1024']['T1_ratio'] = t1
            p.extra_generation_params['HiDiffusion Ratios'] = f'{shared.opts.hidiffusion_t1}/{shared.opts.hidiffusion_t2}'
        if shared.opts.hidiffusion_t2 >= 0:
            t2 =shared.opts.hidiffusion_t2
            hidiffusion.switching_threshold_ratio_dict['sd15_1024']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sd15_2048']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sdxl_2048']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sdxl_4096']['T2_ratio'] = t2
            hidiffusion.switching_threshold_ratio_dict['sdxl_turbo_1024']['T2_ratio'] = t2
            p.extra_generation_params['HiDiffusion Ratios'] = f'{shared.opts.hidiffusion_t1}/{shared.opts.hidiffusion_t2}'
        hidiffusion.apply_hidiffusion(pipe, apply_raunet=shared.opts.hidiffusion_raunet, apply_window_attn=shared.opts.hidiffusion_attn, model_type=model_type, steps=p.steps)
        p.extra_generation_params['HiDiffusion'] = f'{shared.opts.hidiffusion_raunet}/{shared.opts.hidiffusion_attn}/{shared.opts.hidiffusion_steps > 0}:{shared.opts.hidiffusion_steps}'
        t1 = time.time()
        log.debug(f'Applying HiDiffusion: raunet={shared.opts.hidiffusion_raunet} attn={shared.opts.hidiffusion_attn} aggressive={shared.opts.hidiffusion_steps > 0}:{shared.opts.hidiffusion_steps} t1={shared.opts.hidiffusion_t1} t2={shared.opts.hidiffusion_t2} time={t1-t0:.2f} type={shared.sd_model_type} width={p.width} height={p.height}')
    elif hasattr(pipe, 'unet') and getattr(pipe.unet, 'hidiffusion', False):
        log.warning('HiDiffusion: model reload recomended')


def unapply():
    pipe = shared.sd_model.pipe if hasattr(shared.sd_model, 'pipe') else shared.sd_model
    if hasattr(pipe, 'unet') and pipe.unet is not None:
        hidiffusion.remove_hidiffusion(pipe)
