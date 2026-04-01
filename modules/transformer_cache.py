import os
import diffusers
from modules import shared, errors
from modules.logger import log


debug = log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None


def set_cache(faster_cache=None, pyramid_attention_broadcast=None):
    if not shared.sd_loaded or not hasattr(shared.sd_model, 'transformer'):
        return
    faster_cache = faster_cache if faster_cache is not None else shared.opts.faster_cache_enabled
    pyramid_attention_broadcast = pyramid_attention_broadcast if pyramid_attention_broadcast is not None else shared.opts.pab_enabled
    if (not faster_cache) and (not pyramid_attention_broadcast):
        return
    if (not hasattr(shared.sd_model.transformer, 'enable_cache')) or (not hasattr(shared.sd_model.transformer, 'disable_cache')):
        log.debug(f'Transformer cache: cls={shared.sd_model.transformer.__class__.__name__} fc={faster_cache} pab={pyramid_attention_broadcast} not supported')
        return
    try:
        if faster_cache: # https://github.com/huggingface/diffusers/pull/10163
            distilled = shared.opts.fc_guidance_distilled or shared.sd_model_type == 'f1'
            config = diffusers.FasterCacheConfig(
                spatial_attention_block_skip_range=shared.opts.fc_spacial_skip_range,
                spatial_attention_timestep_skip_range=(int(shared.opts.fc_spacial_skip_start), int(shared.opts.fc_spacial_skip_end)),
                unconditional_batch_skip_range=shared.opts.fc_uncond_skip_range,
                unconditional_batch_timestep_skip_range=(int(shared.opts.fc_uncond_skip_start), int(shared.opts.fc_uncond_skip_end)),
                attention_weight_callback=lambda _: shared.opts.fc_attention_weight,
                tensor_format=shared.opts.fc_tensor_format, # TODO fc: autodetect tensor format based on model
                is_guidance_distilled=distilled, # TODO fc: autodetect distilled based on model
                current_timestep_callback=lambda: shared.sd_model.current_timestep,
            )
            shared.sd_model.transformer.disable_cache()
            shared.sd_model.transformer.enable_cache(config)
            log.debug(f'Transformer cache: type={config.__class__.__name__}')
            debug(f'Transformer cache: {vars(config)}')
        elif pyramid_attention_broadcast: # https://github.com/huggingface/diffusers/pull/9562
            config = diffusers.PyramidAttentionBroadcastConfig(
                spatial_attention_block_skip_range=shared.opts.pab_spacial_skip_range,
                spatial_attention_timestep_skip_range=(int(shared.opts.pab_spacial_skip_start), int(shared.opts.pab_spacial_skip_end)),
                current_timestep_callback=lambda: shared.sd_model.current_timestep,
            )
            shared.sd_model.transformer.disable_cache()
            shared.sd_model.transformer.enable_cache(config)
            log.debug(f'Transformer cache: type={config.__class__.__name__}')
            debug(f'Transformer cache: {vars(config)}')
        else:
            debug('Transformer cache: not enabled')
            shared.sd_model.transformer.disable_cache()
    except Exception as e:
        log.error(f'Transformer cache: {e}')
        errors.display(e, 'Transformer cache')
