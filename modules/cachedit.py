from installer import install
from modules import shared


def apply_cache_dit(pipe):
    if not shared.opts.cache_dit_enabled:
        return
    install('cache_dit')
    try:
        import cache_dit
    except Exception as e:
        shared.log.error(f'Cache-DIT: {e}')
        return
    _, supported = cache_dit.supported_pipelines()
    supported = [s.replace('*', '') for s in supported]
    if not any(pipe.__class__.__name__.startswith(s) for s in supported):
        shared.log.error(f'Cache-DiT: pipeline={pipe.__class__.__name__} unsupported')
        return

    if getattr(pipe, 'has_cache_dit', False):
        unapply_cache_dir(pipe)

    cache_config = cache_dit.BasicCacheConfig()
    if shared.opts.cache_dit_calibrator == "TaylorSeer":
        calibrator_config = cache_dit.TaylorSeerCalibratorConfig(taylorseer_order=1)
    elif shared.opts.cache_dit_calibrator == "FoCa":
        calibrator_config = cache_dit.FoCaCalibratorConfig()
    else:
        calibrator_config = None
    """
        Fn_compute_blocks=shared.opts.cache_dit_Fn_compute_blocks, # 8
        Bn_compute_blocks=shared.opts.cache_dit_Bn_compute_blocks, # 0
        residual_diff_threshold=shared.opts.cache_dit_residual_diff_threshold, # 0.08
        max_warmup_steps=shared.opts.cache_dit_max_warmup_steps, # 8
        max_cached_steps=shared.opts.cache_dit_max_cached_steps, # -1
        max_continuous_cached_steps=shared.opts.cache_dit_max_continuous_cached_steps, # -1
        enable_separate_cfg=shared.opts.cache_dit_enable_separate_cfg, # False
        cfg_compute_first=shared.opts.cache_dit_cfg_compute_first, # False
    )
    """
    shared.log.info(f'Apply Cache-DiT: config="{cache_config.strify()}" calibrator="{calibrator_config.strify() if calibrator_config else "None"}"')
    cache_dit.enable_cache(
        pipe,
        cache_config=cache_config,
        calibrator_config=calibrator_config,
    )
    shared.sd_model.has_cache_dit = True


def unapply_cache_dir(pipe):
    if not shared.opts.cache_dit_enabled or not getattr(pipe, 'has_cache_dit', False):
        return
    try:
        import cache_dit
        cache_dit.disable_cache(pipe)
        pipe.has_cache_dit = False
    except Exception:
        return
