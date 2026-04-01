import time
import diffusers
from modules import shared
from modules.logger import log


modular_map= {
    'StableDiffusionXLPipeline': 'StableDiffusionXLAutoBlocks',
    'StableDiffusionXLImg2ImgPipeline': 'StableDiffusionXLAutoBlocks',
    'StableDiffusionXLInpaintPipeline': 'StableDiffusionXLAutoBlocks',
    'FluxPipeline': 'FluxAutoBlocks',
    'FluxImg2ImgPipeline': 'FluxAutoBlocks',
    'FluxInpaintPipeline': 'FluxAutoBlocks',
    'WanPipeline': 'WanAutoBlocks',
    'WanImageToVideoPipeline': 'WanAutoBlocks',
    'QwenImagePipeline': 'QwenImageAutoBlocks',
    'QwenImageEditPipeline': 'QwenImageEditAutoBlocks',
}


def is_compatible(diffusion_pipeline: diffusers.DiffusionPipeline) -> bool:
    if not shared.opts.model_modular_enable:
        return False
    compatible = diffusion_pipeline.__class__.__name__ in modular_map
    if not compatible:
        log.debug(f'Modular: source={diffusion_pipeline.__class__.__name__} incompatible pipeline')
    return compatible


def is_guider(diffusion_pipeline: diffusers.DiffusionPipeline) -> bool:
    guider = getattr(diffusion_pipeline, 'guider', None)
    return guider is not None


def convert_to_modular(diffusion_pipeline: diffusers.DiffusionPipeline) -> diffusers.ModularPipeline:
    modular_pipe = None
    try:
        t0 = time.time()
        modular_cls = modular_map.get(diffusion_pipeline.__class__.__name__, None)
        if modular_cls is None:
            raise ValueError(f'unknown: cls={diffusion_pipeline.__class__.__name__}')
        modular_cls = getattr(diffusers, modular_cls, None)
        if modular_cls is None:
            raise ValueError(f'invalid: cls={diffusion_pipeline.__class__.__name__}')
        modular_blocks = modular_cls()
        modular_pipe = modular_blocks.init_pipeline()
        components_dct = {k: v for k, v in diffusion_pipeline.components.items() if v is not None}
        modular_pipe.update_components(**components_dct, **diffusion_pipeline.parameters)
        modular_pipe.original_pipe = diffusion_pipeline
        t1 = time.time()
        log.debug(f'Modular: source={diffusion_pipeline.__class__.__name__} target={modular_pipe.__class__.__name__} time={t1 - t0:.2f}')
        """
        for expected_input_param in modular_pipe.blocks.inputs:
            name = expected_input_param.name
            default = expected_input_param.default
            kwargs_type = expected_input_param.kwargs_type
            log.trace(f'Modular input: name={name} type={kwargs_type} default={default}')
        """

    except Exception as e:
        log.error(f'Modular: {e}')
        raise e
    return modular_pipe


def restore_standard(modular_pipe):
    if hasattr(modular_pipe, 'original_pipe'):
        log.debug(f'Modular: source={modular_pipe.__class__.__name__} target={modular_pipe.original_pipe.__class__.__name__}')
        return modular_pipe.original_pipe
    return modular_pipe
