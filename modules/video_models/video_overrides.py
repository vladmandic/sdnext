import os
import torch
import diffusers
from modules import shared, processing
from modules.video_models.models_def import Model


debug = shared.log.trace if os.environ.get('SD_VIDEO_DEBUG', None) is not None else lambda *args, **kwargs: None


def load_override(selected: Model):
    kwargs = {}
    if selected.name == 'Allegro T2V':
        kwargs['vae'] = diffusers.AutoencoderKLAllegro.from_pretrained(selected.repo,
                                                                       subfolder="vae",
                                                                       torch_dtype=torch.float32,
                                                                       cache_dir=shared.opts.hfcache_dir)
        debug(f'Video overrides: model="{selected.name}" kwargs={list(kwargs)}')
    if selected.name == 'LTXVideo 0.9.5 I2V':
        kwargs['vae'] = diffusers.AutoencoderKLLTXVideo.from_pretrained(selected.repo,
                                                                       subfolder="vae",
                                                                       torch_dtype=torch.float32,
                                                                       cache_dir=shared.opts.hfcache_dir)
    return kwargs


def set_overrides(p: processing.StableDiffusionProcessingVideo, selected: Model):
    cls = shared.sd_model.__class__.__name__
    # Allegro
    if selected.name == 'Allegro T2V':
        shared.sd_model.vae.enable_tiling()
    # Latte
    if selected.name == 'Latte 1 T2V':
        p.task_args['enable_temporal_attentions'] = False
        p.task_args['video_length'] = p.frames
    # LTX
    if cls == 'LTXImageToVideoPipeline' or cls == 'LTXConditionPipeline':
        p.task_args['generator'] = None
    if cls == 'LTXConditionPipeline':
        p.task_args['strength'] = p.denoising_strength
    if 'LTX' in shared.sd_model.__class__.__name__:
        p.task_args['width'] = 32 * (p.width // 32)
        p.task_args['height'] = 32 * (p.height // 32)
