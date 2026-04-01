from types import MethodType
import accelerate
from diffusers import FluxPipeline
from modules import shared, sd_models
from modules.logger import log


def apply_flux(pipe: FluxPipeline):
    if not hasattr(pipe, 'transformer') or not 'Nunchaku' in pipe.transformer.__class__.__name__:
        log.error('PuLID: flux support requires nunchaku')
        return pipe

    from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
    if not isinstance(pipe, PuLIDFluxPipeline):
        from nunchaku.models.pulid.pulid_forward import pulid_forward
        sd_models.clear_caches(full=True)
        accelerate.hooks.remove_hook_from_module(pipe.transformer, recurse=True)
        pipe = sd_models.switch_pipe(PuLIDFluxPipeline, pipe)
        pipe.transformer.orig_forward = pipe.transformer.forward
        pipe.transformer.forward = MethodType(pulid_forward, pipe.transformer)
        pipe = sd_models.apply_balanced_offload(pipe)
        pipe.pulid_model = sd_models.apply_balanced_offload(pipe.pulid_model)
        log.info(f'PuLID: flux applied cls={pipe.__class__.__name__} pipe={pipe.pulid_model.__class__.__name__}')
    return pipe


def unapply_flux(pipe: FluxPipeline):
    from nunchaku.pipeline.pipeline_flux_pulid import PuLIDFluxPipeline
    if isinstance(pipe, PuLIDFluxPipeline) and hasattr(pipe.transformer, 'orig_forward'):
        sd_models.clear_caches(full=True)
        accelerate.hooks.remove_hook_from_module(pipe.transformer, recurse=True)
        pipe.transformer.forward = MethodType(pipe.transformer.orig_forward, pipe.transformer)
        del pipe.transformer.orig_forward
        pipe = sd_models.switch_pipe(FluxPipeline, pipe)
        pipe = sd_models.apply_balanced_offload(pipe)
    return pipe
