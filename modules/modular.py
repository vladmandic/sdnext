import os
import diffusers
from modules import shared, sd_hijack_modular
from modules.logger import log


debug = os.environ.get('SD_MODULAR_DEBUG', None) is not None


def get_modular_class_name(diffusion_pipeline: diffusers.DiffusionPipeline) -> str:
    name = diffusion_pipeline.__class__.__name__
    name = name.replace('Pipeline', '').replace('Img2Img', '').replace('Inpaint', '').replace('ImageToVideo', '')
    name = f'{name}AutoBlocks'
    modular_cls = getattr(diffusers, name, None)
    if debug:
        log.trace(f'Modular lookup: key={name} source={diffusion_pipeline.__class__.__name__} target={modular_cls.__name__ if modular_cls else None}')
    return modular_cls


def is_compatible(diffusion_pipeline: diffusers.DiffusionPipeline) -> bool:
    if not shared.opts.model_modular_enable:
        return False
    compatible = get_modular_class_name(diffusion_pipeline) is not None
    if not compatible:
        log.debug(f'Modular: source={diffusion_pipeline.__class__.__name__} incompatible pipeline')
    return compatible


def is_guider(diffusion_pipeline: diffusers.DiffusionPipeline) -> bool:
    guider = getattr(diffusion_pipeline, 'guider', None)
    return guider is not None


def convert_to_modular(diffusion_pipeline: diffusers.DiffusionPipeline) -> diffusers.ModularPipeline:
    modular_pipe = None
    try:
        modular_cls = get_modular_class_name(diffusion_pipeline)
        if modular_cls is None:
            raise ValueError(f'unknown: cls={diffusion_pipeline.__class__.__name__}')
        modular_blocks = modular_cls()
        modular_pipe = modular_blocks.init_pipeline()
        components_dct = {k: v for k, v in diffusion_pipeline.components.items() if v is not None}
        modular_pipe.update_components(**components_dct, **diffusion_pipeline.parameters)
        modular_pipe.original_pipe = diffusion_pipeline
        log.debug(f'Modular convert: source={diffusion_pipeline.__class__.__name__} target={modular_pipe.__class__.__name__}')
    except Exception as e:
        log.error(f'Modular: {e}')
        raise e
    sd_hijack_modular.install_state_hook(modular_pipe)
    sd_hijack_modular.register_callbacks(modular_pipe)
    return modular_pipe


def restore_standard(modular_pipe):
    if hasattr(modular_pipe, 'original_pipe'):
        log.debug(f'Modular: source={modular_pipe.__class__.__name__} target={modular_pipe.original_pipe.__class__.__name__}')
        return modular_pipe.original_pipe
    return modular_pipe
