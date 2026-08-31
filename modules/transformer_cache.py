import os
import diffusers
from modules import shared, errors, processing, devices
from modules.sd_offload_utils import get_module_names
from modules.logger import log


debug = log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None


def get_transformers():
    if not shared.sd_loaded:
        return None
    for module_name in get_module_names(shared.sd_model):
        module = getattr(shared.sd_model, module_name, None)
        if (module is not None) and ('transformer' in module_name or 'Transformer' in module.__class__.__name__):
            yield module


def set_cache(p: processing.StableDiffusionProcessing):
    if not shared.sd_loaded:
        return
    for module in get_transformers():
        try:

            if shared.opts.fc_enabled:
                config = diffusers.hooks.FasterCacheConfig(
                    spatial_attention_block_skip_range=int(shared.opts.fc_spacial_skip_range),
                    spatial_attention_timestep_skip_range=(int(shared.opts.fc_spacial_skip_start), int(shared.opts.fc_spacial_skip_end)),
                    unconditional_batch_skip_range=int(shared.opts.fc_uncond_skip_range),
                    unconditional_batch_timestep_skip_range=(int(shared.opts.fc_uncond_skip_start), int(shared.opts.fc_uncond_skip_end)),
                    attention_weight_callback=lambda _: float(shared.opts.fc_attention_weight),
                    tensor_format=str(shared.opts.fc_tensor_format),
                    is_guidance_distilled=bool(shared.opts.fc_guidance_distilled),
                    current_timestep_callback=lambda: shared.sd_model.current_timestep,
                )
                if getattr(shared.sd_model, 'cache_applied', None) == config:
                    return
                if hasattr(module, 'disable_cache'):
                    module.disable_cache()
                shared.sd_model.cache_applied = config
                if not hasattr(shared.sd_model, 'current_timestep'):
                    log.warning(f'Transformer cache: method=FasterCache cls={shared.sd_model.__class__.__name__} not compatible')
                else:
                    diffusers.hooks.apply_faster_cache(module, config)
                    log.debug(f'Transformer cache: method=FasterCache module={module.__class__.__name__} config={config}')

            if shared.opts.pab_enabled:
                config = diffusers.hooks.PyramidAttentionBroadcastConfig(
                    spatial_attention_block_skip_range=int(shared.opts.pab_spacial_skip_range) if shared.opts.pab_spacial_skip_range > 0 else None,
                    spatial_attention_timestep_skip_range=(int(shared.opts.pab_spacial_skip_start), int(shared.opts.pab_spacial_skip_end)),
                    current_timestep_callback=lambda: shared.sd_model.current_timestep,
                )
                if getattr(shared.sd_model, 'cache_applied', None) == config:
                    return
                if hasattr(module, 'disable_cache'):
                    module.disable_cache()
                shared.sd_model.cache_applied = config
                if not hasattr(shared.sd_model, 'current_timestep'):
                    log.warning(f'Transformer cache: method=PyramidAttentionBroadcast cls={shared.sd_model.__class__.__name__} not compatible')
                else:
                    diffusers.hooks.apply_pyramid_attention_broadcast(module, config)
                    log.debug(f'Transformer cache: method=PyramidAttentionBroadcast module={module.__class__.__name__} config={config}')

            if shared.opts.ls_enabled:
                config = diffusers.hooks.LayerSkipConfig(
                    indices=[int(i.strip()) for i in shared.opts.ls_indices.split(',') if i.strip().isnumeric()],
                    fqn=str(shared.opts.ls_fqn),
                    skip_attention=bool(shared.opts.ls_skip_attention),
                    skip_attention_scores=bool(shared.opts.ls_skip_attention_scores),
                    skip_ff=bool(shared.opts.ls_skip_ff),
                    dropout=float(shared.opts.ls_dropout),
                )
                if getattr(shared.sd_model, 'cache_applied', None) == config:
                    return
                if hasattr(module, 'disable_cache'):
                    module.disable_cache()
                shared.sd_model.cache_applied = config
                diffusers.hooks.apply_layer_skip(module, config)
                log.debug(f'Transformer cache: method=LayerSkip module={module.__class__.__name__} config={config}')

            if shared.opts.mc_enabled:
                config = diffusers.hooks.MagCacheConfig(
                    threshold=float(shared.opts.mc_threshold),
                    max_skip_steps=int(shared.opts.mc_max_skip_steps),
                    retention_ratio=float(shared.opts.mc_retention_ratio),
                    num_inference_steps=int(p.steps)
                )
                if getattr(shared.sd_model, 'cache_applied', None) == config:
                    return
                if hasattr(module, 'disable_cache'):
                    module.disable_cache()
                shared.sd_model.cache_applied = config
                diffusers.hooks.apply_mag_cache(module, config)
                log.debug(f'Transformer cache: method=MagCache module={module.__class__.__name__} config={config}')

            if shared.opts.ts_enabled:
                config = diffusers.hooks.TaylorSeerCacheConfig(
                    cache_interval=int(shared.opts.ts_cache_interval),
                    disable_cache_before_step=int(shared.opts.ts_disable_cache_before_step),
                    disable_cache_after_step=int(shared.opts.ts_disable_cache_after_step),
                    max_order=int(shared.opts.ts_max_order),
                    taylor_factors_dtype=devices.dtype,
                    skip_predict_identifiers=[i.strip() for i in shared.opts.ts_skip_predict_identifiers.split(',') if i.strip()],
                    cache_identifiers=[i.strip() for i in shared.opts.ts_cache_identifiers.split(',') if i.strip()],
                    use_lite_mode=bool(shared.opts.ts_use_lite_mode),
                )
                if getattr(shared.sd_model, 'cache_applied', None) == config:
                    return
                if hasattr(module, 'disable_cache'):
                    module.disable_cache()
                shared.sd_model.cache_applied = config
                diffusers.hooks.apply_taylorseer_cache(module, config)
                log.debug(f'Transformer cache: method=TaylorSeerCache module={module.__class__.__name__} config={config}')

            if shared.opts.fb_enabled:
                config = diffusers.hooks.FirstBlockCacheConfig(
                    threshold=float(shared.opts.fb_threshold),
                )
                if getattr(shared.sd_model, 'cache_applied', None) == config:
                    return
                if hasattr(module, 'disable_cache'):
                    module.disable_cache()
                shared.sd_model.cache_applied = config
                diffusers.hooks.apply_first_block_cache(module, config)
                log.debug(f'Transformer cache: method=FirstBlockCache module={module.__class__.__name__} config={config}')

        except Exception as e:
            log.error(f'Transformer cache: {e}')
            errors.display(e, 'Transformer cache')
