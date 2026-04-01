import os
from installer import install
from modules import shared
from modules.logger import log


def apply_cache_dit(pipe):
    if not shared.opts.cache_dit_enabled:
        return
    install('git+https://github.com/vipshop/cache-dit', 'cache_dit')
    os.environ.setdefault("CACHE_DIT_LOG_LEVEL", "error")
    try:
        import cache_dit
    except Exception as e:
        log.error(f'Cache-DIT: {e}')
        return
    _, supported = cache_dit.supported_pipelines()
    supported = [s.replace('*', '') for s in supported]
    if not any(pipe.__class__.__name__.startswith(s) for s in supported):
        log.error(f'Cache-DiT: pipeline={pipe.__class__.__name__} unsupported')
        return

    if getattr(pipe, 'has_cache_dit', False):
        unapply_cache_dir(pipe)

    config_args = {}
    if shared.opts.cache_dit_fcompute >= 0:
        config_args['Fn_compute_blocks'] = int(shared.opts.cache_dit_fcompute)
    if shared.opts.cache_dit_bcompute >= 0:
        config_args['Bn_compute_blocks'] = int(shared.opts.cache_dit_bcompute)
    if shared.opts.cache_dit_threshold >= 0:
        config_args['residual_diff_threshold'] = float(shared.opts.cache_dit_threshold)
    if shared.opts.cache_dit_warmup >= 0:
        config_args['max_warmup_steps'] = int(shared.opts.cache_dit_warmup)
    cache_config = cache_dit.BasicCacheConfig(**config_args)
    if shared.opts.cache_dit_calibrator == "TaylorSeer":
        calibrator_config = cache_dit.TaylorSeerCalibratorConfig(taylorseer_order=1)
    elif shared.opts.cache_dit_calibrator == "FoCa":
        calibrator_config = cache_dit.FoCaCalibratorConfig()
    else:
        calibrator_config = None
    log.info(f'Apply Cache-DiT: config="{cache_config.strify()}" calibrator="{calibrator_config.strify() if calibrator_config else "None"}"')
    try:
        cache_dit.enable_cache(
            pipe,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
        )
        shared.sd_model.has_cache_dit = True
    except Exception as e:
        log.error(f'Cache-DiT: {e}')
        return


def unapply_cache_dir(pipe):
    if not shared.opts.cache_dit_enabled or not getattr(pipe, 'has_cache_dit', False):
        return
    try:
        import cache_dit
        # stats = cache_dit.summary(pipe)
        # log.critical(f'Unapply Cache-DiT: {stats}')
        cache_dit.disable_cache(pipe)
        pipe.has_cache_dit = False
    except Exception:
        return
