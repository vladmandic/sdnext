import diffusers
from modules import shared
from modules.logger import log


def apply_teacache_patch(cls):
    if shared.opts.teacache_enabled and cls is not None:
        from modules import teacache
        log.debug(f'Transformers cache: type=teacache patch=forward cls={cls.__name__}')
        if cls.__name__ == 'LTXVideoTransformer3DModel':
            cls.forward = teacache.teacache_ltx_forward
        elif cls.__name__ == 'MochiTransformer3DModel':
            cls.forward = teacache.teacache_mochi_forward
        elif cls.__name__ == 'CogVideoXTransformer3DModel':
            cls.forward = teacache.teacache_cog_forward

        diffusers.FluxTransformer2DModel.forward = teacache.teacache_flux_forward
