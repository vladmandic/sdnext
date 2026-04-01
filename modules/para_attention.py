from modules.logger import log
from modules import shared


supported_models = ['Flux', 'HunyuanVideo', 'CogVideoX', 'Mochi']


def apply_first_block_cache():
    if not shared.opts.para_cache_enabled:
        return
    if not any(shared.sd_model.__class__.__name__.startswith(x) for x in supported_models):
        return
    from installer import install
    install('para_attn')
    try:
        if 'Nunchaku' in shared.sd_model.transformer.__class__.__name__:
            from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
            log.info(f'Transformers cache: type=nunchaku rdt={shared.opts.para_diff_threshold} cls={shared.sd_model.transformer.__class__.__name__}')
        else:
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            log.info(f'Transformers cache: type=paraattn rdt={shared.opts.para_diff_threshold} cls={shared.sd_model.transformer.__class__.__name__}')
        apply_cache_on_pipe(shared.sd_model, residual_diff_threshold=shared.opts.para_diff_threshold)
    except Exception as e:
        log.error(f'Transformers cache: type=paraattn {e}')
        return
