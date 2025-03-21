import os
import diffusers
from modules import shared, errors


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def set_cache(faster_cache=False, pyramid_attention_broadcast=False):
    if not shared.sd_loaded or not hasattr(shared.sd_model, 'transformer'):
        return
    if not hasattr(shared.sd_model.transformer, 'enable_cache'):
        debug(f'Video cache: cls={shared.sd_model.transformer.__class__.__name__} not supported')
        return
    try:
        if faster_cache: # https://github.com/huggingface/diffusers/pull/10163
            config = diffusers.FasterCacheConfig(
                spatial_attention_block_skip_range=2,
                spatial_attention_timestep_skip_range=(-1, 681),
                current_timestep_callback=lambda: shared.sd_model.current_timestep,
                attention_weight_callback=lambda _: 0.3,
                unconditional_batch_skip_range=5,
                unconditional_batch_timestep_skip_range=(-1, 781),
                tensor_format="BFCHW",
            )
            shared.sd_model.transformer.disable_cache()
            shared.sd_model.transformer.enable_cache(config)
            shared.log.debug(f'Video cache: type={config.__class__.__name__}')
            debug(f'Video cache: {vars(config)}')
        elif pyramid_attention_broadcast: # https://github.com/huggingface/diffusers/pull/9562
            config = diffusers.PyramidAttentionBroadcastConfig(
                spatial_attention_block_skip_range=2,
                spatial_attention_timestep_skip_range=(100, 800),
                current_timestep_callback=lambda: shared.sd_model.current_timestep,
            )
            shared.sd_model.transformer.disable_cache()
            shared.sd_model.transformer.enable_cache(config)
            shared.log.debug(f'Video cache: type={config.__class__.__name__}')
            debug(f'Video cache: {vars(config)}')
        else:
            debug('Video cache: not enabled')
            shared.sd_model.transformer.disable_cache()
    except Exception as e:
        shared.log.error(f'Video cache: error={e}')
        errors.display(e, 'video cache')
